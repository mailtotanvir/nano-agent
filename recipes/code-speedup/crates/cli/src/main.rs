//! CLI for the verifier-gated Python code-speed recipe.

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
use code_speedup_verifier::{BridgeConfig, SpeedupRecipe, DEFAULT_CANDIDATE_FILE};
use nano_controller::drive;
use nano_model_client::{GeminiClient, LlamaCppClient, ModelClient, PROPOSAL_GBNF};
use nano_trajectory::{Actor, Outcome, TrajectoryLog};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "code-speedup",
    about = "Local verified Python code-speed agent"
)]
struct Cli {
    /// Workspace containing the model-editable candidate.py file.
    #[arg(long)]
    workspace: PathBuf,
    /// Verifier-gated private JSONL corpus.
    #[arg(long)]
    dataset: PathBuf,
    /// One private problem record ID in --dataset.
    #[arg(long)]
    problem_id: String,
    /// Local Python executable used for the private harness bridge.
    #[arg(long, default_value = "python3")]
    python: PathBuf,
    /// Workspace-relative model-editable file.
    #[arg(long, default_value = DEFAULT_CANDIDATE_FILE)]
    candidate_file: String,
    /// Local llama.cpp OpenAI-compatible server URL.
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    llama_url: String,
    /// Model backend. Gemini uses GEMINI_API_KEY / GOOGLE_API_KEY directly.
    #[arg(long, value_enum, default_value = "llama")]
    backend: Backend,
    /// GCP project for the Gemini Vertex backend.
    #[arg(long)]
    vertex_project: Option<String>,
    /// Vertex AI location; global supports Gemini global endpoints.
    #[arg(long, default_value = "global")]
    vertex_location: String,
    /// Model name served by local llama.cpp.
    #[arg(long, default_value = "qwen2.5-coder-0.5b-instruct")]
    model: String,
    /// Maximum local rewrite proposals.
    #[arg(long, default_value_t = 4)]
    max_attempts: u32,
    /// Append trajectory JSONL locally.
    #[arg(long, default_value = "trajectories.jsonl")]
    log: PathBuf,
    /// Run the private verifier bridge in this privileged, network-disabled Docker image.
    #[arg(long)]
    bridge_container_image: Option<String>,
}

#[derive(Debug, Clone, ValueEnum)]
enum Backend {
    /// Local llama.cpp OpenAI-compatible server.
    Llama,
    /// Gemini Generative Language API using an API key from the environment.
    Gemini,
    /// Gemini on Vertex AI using Application Default Credentials.
    GeminiVertex,
}

fn recipe_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("cli crate is nested under recipes/code-speedup")
        .to_path_buf()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if !cli.workspace.join(&cli.candidate_file).is_file() {
        bail!("workspace must contain {}", cli.candidate_file);
    }
    if !cli.dataset.is_file() {
        bail!("dataset not found: {}", cli.dataset.display());
    }
    let root = recipe_root();
    let mut bridge = BridgeConfig::new(
        cli.python,
        root.join("harness/bridge.py"),
        root,
        cli.dataset,
        cli.problem_id.clone(),
    );
    bridge.candidate_file = cli.candidate_file;
    bridge.container_image = cli.bridge_container_image;
    let actor = match cli.backend {
        Backend::Llama => Actor::Tiny,
        Backend::Gemini | Backend::GeminiVertex => Actor::Frontier,
    };
    let recipe = SpeedupRecipe {
        bridge,
        max_attempts: cli.max_attempts,
        actor,
    };
    // Deliberately no Vertex/ADC, Azure, or escalation route exists here.
    let model: Box<dyn ModelClient> = match cli.backend {
        Backend::Llama => {
            Box::new(LlamaCppClient::new(cli.llama_url, cli.model).with_grammar(PROPOSAL_GBNF))
        }
        Backend::Gemini => Box::new(GeminiClient::from_env(cli.model)?),
        Backend::GeminiVertex => Box::new(GeminiClient::from_vertex_adc(
            cli.model,
            cli.vertex_project
                .ok_or_else(|| anyhow::anyhow!("--vertex-project is required"))?,
            cli.vertex_location,
        )?),
    };
    let trajectory = drive(
        &recipe,
        &cli.workspace,
        model.as_ref(),
        None,
        &cli.problem_id,
        &format!("{}-local", cli.problem_id),
    )?;
    TrajectoryLog::open(&cli.log)?.append(&trajectory)?;
    println!(
        "outcome={:?} attempts={} initial_codes={:?}",
        trajectory.outcome, trajectory.attempts, trajectory.initial_codes
    );
    if trajectory.outcome == Outcome::Success {
        Ok(())
    } else {
        std::process::exit(2)
    }
}
