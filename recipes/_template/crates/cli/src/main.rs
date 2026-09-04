//! Template recipe CLI — a compilable NO-OP skeleton.
//!
//! `cp -r recipes/_template recipes/<name>`, add the two crates to the root
//! `Cargo.toml` workspace members, then implement the TODOs. This binary already
//! builds and runs: pointed at any directory it verifies (no-op = passes) and
//! prints the outcome.

use anyhow::Result;
use clap::Parser;
use nano_controller::{drive, Config, EscalationPolicy, RecipeSpec, Verifier};
use nano_model_client::{LlamaCppClient, ModelClient, PROPOSAL_GBNF};
use nano_template_verifier::TemplateVerifier;
use std::path::{Path, PathBuf};

/// TODO(4): a domain-specific system prompt. The generic contract (JSON +
/// SEARCH/REPLACE) is described by `nano_controller::GENERIC_SYSTEM_PROMPT`;
/// specialize the first sentence for your tool, e.g. "You are a SQL query
/// repair agent."
const TEMPLATE_SYSTEM_PROMPT: &str = nano_controller::GENERIC_SYSTEM_PROMPT;

#[derive(Parser, Debug)]
#[command(
    name = "nano-template",
    about = "Template nano-agent recipe (no-op skeleton)"
)]
struct Cli {
    /// Path to the workspace directory to repair.
    #[arg(long)]
    path: PathBuf,
    /// llama.cpp server base URL (tiny model).
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    llama_url: String,
    /// Tiny model name served by llama.cpp.
    #[arg(long, default_value = "qwen2.5-coder-0.5b-instruct")]
    model: String,
    /// Max repair attempts.
    #[arg(long, default_value_t = 4)]
    max_attempts: u32,
    /// Case id for logging.
    #[arg(long, default_value = "adhoc")]
    case_id: String,
}

/// The template recipe: implements `RecipeSpec` so the generic `drive()` runs it.
struct TemplateRecipe {
    max_attempts: u32,
}

impl RecipeSpec for TemplateRecipe {
    fn name(&self) -> &str {
        "template"
    }
    fn verifier(&self, workspace: &Path) -> anyhow::Result<Box<dyn Verifier>> {
        Ok(Box::new(TemplateVerifier::new(workspace.to_path_buf())))
    }
    fn system_prompt(&self) -> String {
        TEMPLATE_SYSTEM_PROMPT.to_string()
    }
    fn escalation_policy(&self) -> EscalationPolicy {
        // TODO: enable + configure escalation for your recipe if desired.
        EscalationPolicy::default()
    }
    fn base_config(&self) -> Config {
        Config {
            max_attempts: self.max_attempts,
            ..Default::default()
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let recipe = TemplateRecipe {
        max_attempts: cli.max_attempts,
    };

    // Tiny model over a local llama.cpp server, grammar-constrained.
    let tiny = LlamaCppClient::new(&cli.llama_url, &cli.model).with_grammar(PROPOSAL_GBNF);
    let tiny: Box<dyn ModelClient> = Box::new(tiny);

    let traj_id = format!("{}-template", cli.case_id);
    let traj = drive(
        &recipe,
        &cli.path,
        tiny.as_ref(),
        None, // TODO: pass an escalation model if you enable escalation
        &cli.case_id,
        &traj_id,
    )?;

    println!(
        "outcome={:?} attempts={} escalated={}",
        traj.outcome, traj.attempts, traj.escalated
    );
    Ok(())
}
