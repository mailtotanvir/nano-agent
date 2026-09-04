//! `rust-repair` CLI — Phase 0/5 entry point.
//!
//! Wires the rust-repair Verifier + a chosen model backend into the generic
//! nano-controller loop, and logs the trajectory as JSONL.

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use nano_controller::{run_episode, Config, Verifier, VerifyState};
use nano_model_client::{AzureClient, GeminiClient, LlamaCppClient, ModelClient, PROPOSAL_GBNF};
use nano_trajectory::TrajectoryLog;
use rust_repair_verifier::{cargo_check, CheckResult, Diagnostic};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, ValueEnum)]
enum Backend {
    /// Tiny model served by a local llama.cpp OpenAI-compatible server.
    Llama,
    /// Gemini frontier teacher / fallback (AI Studio API key).
    Gemini,
    /// Gemini via Vertex AI + ADC (billed to a GCP project; real quota).
    Vertex,
    /// Azure AI / OpenAI-compatible endpoint (AZURE_API_BASE + AZURE_API_KEY).
    Azure,
}

#[derive(Parser, Debug)]
#[command(
    name = "rust-repair",
    about = "Tiny verified Rust compiler-error repair agent"
)]
struct Cli {
    /// Path to the crate directory to repair (contains Cargo.toml).
    #[arg(long)]
    path: PathBuf,
    /// Model backend.
    #[arg(long, value_enum, default_value = "llama")]
    backend: Backend,
    /// Model name/id (llama server model name, or gemini model like gemini-2.5-flash).
    #[arg(long, default_value = "qwen2.5-coder-0.5b-instruct")]
    model: String,
    /// llama.cpp server base URL.
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    llama_url: String,
    /// Max repair attempts.
    #[arg(long, default_value_t = 4)]
    max_attempts: u32,
    /// Trajectory JSONL output path (appended).
    #[arg(long, default_value = "trajectories.jsonl")]
    log: PathBuf,
    /// Case identifier for logging.
    #[arg(long, default_value = "adhoc")]
    case_id: String,
    /// Disable the GBNF grammar constraint on the llama backend.
    #[arg(long)]
    no_grammar: bool,
    /// GCP project id for the vertex backend (falls back to the GCP_PROJECT env var).
    #[arg(long, default_value = "your-gcp-project")]
    gcp_project: String,
    /// Vertex AI location (use "global" for newest Gemini models).
    #[arg(long, default_value = "global")]
    gcp_location: String,
}

/// Rust-specific Verifier: builds model context from cargo diagnostics and
/// reads/writes files under the crate directory.
struct RustVerifier {
    root: PathBuf,
    last_primary_file: std::cell::RefCell<Option<String>>,
}

impl RustVerifier {
    fn new(root: PathBuf) -> Self {
        Self {
            root,
            last_primary_file: std::cell::RefCell::new(None),
        }
    }

    fn build_context(&self, res: &CheckResult) -> String {
        let Some(d) = res.primary() else {
            return "No errors.".into();
        };
        *self.last_primary_file.borrow_mut() = d.file.clone();
        let mut ctx = String::new();
        if let Some(f) = &d.file {
            ctx.push_str(&format!("file: {f}\n"));
        }
        // Full rendered primary diagnostic.
        if !d.rendered.is_empty() {
            ctx.push_str(&d.rendered);
            ctx.push('\n');
        } else {
            ctx.push_str(&format!(
                "error{}: {}\n",
                d.code
                    .as_deref()
                    .map(|c| format!("[{c}]"))
                    .unwrap_or_default(),
                d.message
            ));
        }
        // Code excerpt: window around the primary span.
        if let Some(f) = &d.file {
            if let Ok(src) = std::fs::read_to_string(self.root.join(f)) {
                let excerpt = window(&src, d.line_start, d.line_end, 20);
                ctx.push_str(&format!("--- {} (context) ---\n{}\n", f, excerpt));
            }
        }
        if res.error_count() > 1 {
            ctx.push_str(&format!(
                "\n(note: {} total errors; fix this one first)\n",
                res.error_count()
            ));
        }
        ctx
    }
}

fn window(src: &str, line_start: usize, line_end: usize, pad: usize) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let start = line_start.saturating_sub(pad).max(1);
    let end = (line_end + pad).min(lines.len());
    let mut out = String::new();
    for (i, l) in lines.iter().enumerate() {
        let n = i + 1;
        if n >= start && n <= end {
            out.push_str(&format!("{n:>4} | {l}\n"));
        }
    }
    out
}

impl Verifier for RustVerifier {
    fn command(&self) -> &str {
        "cargo check"
    }

    fn verify(&self) -> anyhow::Result<VerifyState> {
        let res = cargo_check(&self.root)?;
        let context = self.build_context(&res);
        let primary_code = res.primary().and_then(|d: &Diagnostic| d.code.clone());
        Ok(VerifyState {
            passed: res.passed,
            error_count: res.error_count(),
            primary_code,
            codes: res.codes(),
            context,
        })
    }

    fn read_file(&self, rel: &str) -> anyhow::Result<String> {
        std::fs::read_to_string(self.root.join(rel)).with_context(|| format!("reading {rel}"))
    }
    fn write_file(&self, rel: &str, content: &str) -> anyhow::Result<()> {
        std::fs::write(self.root.join(rel), content).with_context(|| format!("writing {rel}"))
    }
    fn default_file(&self) -> Option<String> {
        self.last_primary_file.borrow().clone()
    }
}

fn build_model(cli: &Cli) -> Result<Box<dyn ModelClient>> {
    match cli.backend {
        Backend::Llama => {
            let mut c = LlamaCppClient::new(&cli.llama_url, &cli.model);
            if !cli.no_grammar {
                c = c.with_grammar(PROPOSAL_GBNF);
            }
            Ok(Box::new(c))
        }
        Backend::Gemini => {
            let c = GeminiClient::from_env(&cli.model)?;
            Ok(Box::new(c))
        }
        Backend::Vertex => {
            let c = GeminiClient::from_vertex_adc(&cli.model, &cli.gcp_project, &cli.gcp_location)?;
            Ok(Box::new(c))
        }
        Backend::Azure => {
            let c = AzureClient::from_env(&cli.model)?;
            Ok(Box::new(c))
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let mut cli = Cli::parse();

    // Env fallback: use GCP_PROJECT when the flag was left at its placeholder default.
    if cli.gcp_project == "your-gcp-project" {
        if let Ok(p) = std::env::var("GCP_PROJECT") {
            if !p.is_empty() {
                cli.gcp_project = p;
            }
        }
    }

    if !cli.path.join("Cargo.toml").exists() {
        anyhow::bail!("no Cargo.toml at {}", cli.path.display());
    }

    let verifier = RustVerifier::new(cli.path.clone());
    let model = build_model(&cli)?;
    let actor = match cli.backend {
        Backend::Gemini | Backend::Vertex | Backend::Azure => nano_trajectory::Actor::Frontier,
        Backend::Llama => nano_trajectory::Actor::Tiny,
    };
    let cfg = Config {
        max_attempts: cli.max_attempts,
        actor,
        ..Default::default()
    };

    let traj_id = format!("{}-{}", cli.case_id, chrono_ts());
    let traj = run_episode(&verifier, model.as_ref(), &cfg, &cli.case_id, &traj_id)?;

    let mut log = TrajectoryLog::open(&cli.log)?;
    log.append(&traj)?;

    println!(
        "outcome={:?} attempts={} initial_errors={} model={} latency_ms={}",
        traj.outcome,
        traj.attempts,
        traj.initial_codes.len(),
        traj.model,
        traj.total_latency_ms
    );
    // Non-zero exit if not fixed, so scripts can branch.
    match traj.outcome {
        nano_trajectory::Outcome::Success => Ok(()),
        _ => std::process::exit(2),
    }
}

fn chrono_ts() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{}", d.as_millis())
}

#[allow(dead_code)]
fn _unused(_: &Path) {}
