//! Deterministic closed-loop repair controller (spec §6), recipe-agnostic.
//!
//! The controller owns the loop; recipes supply a [`Verifier`] (how to build and
//! read the tool's diagnostics) and a [`ModelClient`]. The controller never knows
//! it is repairing Rust — the same loop will drive SQL/Terraform recipes.

use nano_model_client::{Message, ModelClient};
use nano_patch::{apply_block, parse_blocks, EditBlock, PatchLimits};
use nano_trajectory::{Actor, Outcome, Proposal, Step, Trajectory};
use std::time::Instant;

/// A snapshot of verifier state at one point in the loop.
pub struct VerifyState {
    pub passed: bool,
    pub error_count: usize,
    /// Primary error code, if any (e.g. "E0308").
    pub primary_code: Option<String>,
    /// All error codes (for logging).
    pub codes: Vec<String>,
    /// The model-facing context string describing the primary failure.
    pub context: String,
}

/// Recipe-specific verifier + context builder. Implemented by e.g. rust-repair.
pub trait Verifier {
    /// Human-readable command, e.g. "cargo check".
    fn command(&self) -> &str;
    /// Run the tool against the current workspace and summarize.
    fn verify(&self) -> anyhow::Result<VerifyState>;
    /// Read a file from the workspace (relative path).
    fn read_file(&self, rel: &str) -> anyhow::Result<String>;
    /// Write a file to the workspace (relative path).
    fn write_file(&self, rel: &str, content: &str) -> anyhow::Result<()>;
    /// Default target file when a patch block omits its own hint.
    fn default_file(&self) -> Option<String>;
}

/// Controller configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub max_attempts: u32,
    pub patch_limits: PatchLimits,
    /// Revert a patch if it increases the error count.
    pub regression_guard: bool,
    /// System prompt for the tiny model.
    pub system_prompt: String,
    /// Which actor this run represents (tiny model vs frontier teacher), for
    /// correct trajectory provenance.
    pub actor: Actor,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            patch_limits: PatchLimits::default(),
            regression_guard: true,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            actor: Actor::Tiny,
        }
    }
}

pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a Rust compiler-error repair agent. You are given one compiler error and the relevant source. Respond with ONLY a single JSON object matching this contract: {\"action\": \"patch\"|\"no_fix\"|\"escalate\", \"patch\": \"<SEARCH/REPLACE blocks or null>\", \"reason\": \"<short>\", \"confidence\": <0.0-1.0>}. For a patch, the \"patch\" field must contain one or more blocks in this exact format:\nfile: <path>\n<<<<<<< SEARCH\n<exact existing lines>\n=======\n<replacement lines>\n>>>>>>> REPLACE\nMake the smallest change that fixes the error. If you cannot fix it, use action \"no_fix\" or \"escalate\".";

/// Run one repair episode. Mutates the workspace through the verifier.
pub fn run_episode(
    verifier: &dyn Verifier,
    model: &dyn ModelClient,
    cfg: &Config,
    case_id: &str,
    traj_id: &str,
) -> anyhow::Result<Trajectory> {
    let ep_start = Instant::now();
    let mut traj = Trajectory::new(
        traj_id,
        case_id,
        model.name().to_string(),
        verifier.command().to_string(),
    );

    // Initial verify.
    let mut state = verifier.verify()?;
    traj.initial_codes = state.codes.clone();
    if state.passed {
        traj.outcome = Outcome::Success;
        traj.total_latency_ms = ep_start.elapsed().as_millis() as u64;
        return Ok(traj);
    }

    let mut attempt = 0u32;
    while attempt < cfg.max_attempts {
        // Build context from the current primary failure.
        let mut messages = vec![Message::system(cfg.system_prompt.clone())];
        // Include prior step summaries as assistant/user turns for recovery context.
        for prev in &traj.steps {
            if let Some(p) = &prev.proposal.patch {
                messages.push(Message::assistant(p.clone()));
            }
        }
        messages.push(Message::user(state.context.clone()));

        let resp = model.propose(&messages);
        let (proposal, latency_ms, tokens_out, ok) = match resp {
            Ok(r) => (r.proposal, r.latency_ms, r.tokens_out, true),
            Err(e) => {
                tracing::warn!("model error on attempt {attempt}: {e}");
                (
                    Proposal {
                        action: "no_fix".into(),
                        patch: None,
                        reason: Some(format!("model_error: {e}")),
                        confidence: 0.0,
                    },
                    0,
                    None,
                    false,
                )
            }
        };

        // Terminal non-patch actions.
        if proposal.action == "escalate" {
            push_step(
                &mut traj,
                cfg.actor,
                attempt,
                state.primary_code.clone(),
                state.error_count,
                &state.context,
                proposal,
                false,
                false,
                None,
                false,
                latency_ms,
                tokens_out,
            );
            traj.outcome = Outcome::Escalate;
            traj.attempts = attempt + 1;
            traj.total_latency_ms = ep_start.elapsed().as_millis() as u64;
            return Ok(traj);
        }
        if proposal.action == "no_fix" || !ok {
            push_step(
                &mut traj,
                cfg.actor,
                attempt,
                state.primary_code.clone(),
                state.error_count,
                &state.context,
                proposal,
                false,
                false,
                None,
                false,
                latency_ms,
                tokens_out,
            );
            attempt += 1;
            if !ok {
                continue;
            }
            traj.outcome = Outcome::NoFix;
            traj.attempts = attempt;
            traj.total_latency_ms = ep_start.elapsed().as_millis() as u64;
            return Ok(traj);
        }

        // action == "patch": parse and apply.
        let patch_text = proposal.patch.clone().unwrap_or_default();
        let blocks = match parse_blocks(&patch_text, verifier.default_file().as_deref()) {
            Ok(b) if !b.is_empty() => b,
            _ => {
                push_step(
                    &mut traj,
                    cfg.actor,
                    attempt,
                    state.primary_code.clone(),
                    state.error_count,
                    &state.context,
                    proposal,
                    false,
                    false,
                    None,
                    false,
                    latency_ms,
                    tokens_out,
                );
                attempt += 1;
                continue;
            }
        };

        // Snapshot touched files for possible revert.
        let mut snapshots: Vec<(String, String)> = Vec::new();
        let mut applied = true;
        let mut fuzzy_used = false;
        for b in &blocks {
            let orig = match verifier.read_file(&b.file) {
                Ok(c) => c,
                Err(_) => {
                    applied = false;
                    break;
                }
            };
            if !snapshots.iter().any(|(f, _)| f == &b.file) {
                snapshots.push((b.file.clone(), orig.clone()));
            }
            match apply_block(&orig, b, &cfg.patch_limits) {
                Ok(outcome) => {
                    fuzzy_used |= outcome.fuzzy;
                    if verifier.write_file(&b.file, &outcome.new_content).is_err() {
                        applied = false;
                        break;
                    }
                }
                Err(_) => {
                    applied = false;
                    break;
                }
            }
        }

        if !applied {
            revert(verifier, &snapshots);
            push_step(
                &mut traj,
                cfg.actor,
                attempt,
                state.primary_code.clone(),
                state.error_count,
                &state.context,
                proposal,
                false,
                fuzzy_used,
                None,
                false,
                latency_ms,
                tokens_out,
            );
            attempt += 1;
            continue;
        }

        // Re-verify.
        let new_state = verifier.verify()?;
        let regressed = cfg.regression_guard && new_state.error_count > state.error_count;
        if regressed {
            revert(verifier, &snapshots);
            push_step(
                &mut traj,
                cfg.actor,
                attempt,
                state.primary_code.clone(),
                state.error_count,
                &state.context,
                proposal,
                true,
                fuzzy_used,
                Some(new_state.error_count),
                true,
                latency_ms,
                tokens_out,
            );
            attempt += 1;
            // state unchanged (reverted)
            continue;
        }

        push_step(
            &mut traj,
            cfg.actor,
            attempt,
            state.primary_code.clone(),
            state.error_count,
            &state.context,
            proposal,
            true,
            fuzzy_used,
            Some(new_state.error_count),
            false,
            latency_ms,
            tokens_out,
        );
        attempt += 1;

        if new_state.passed {
            traj.outcome = Outcome::Success;
            traj.attempts = attempt;
            traj.total_latency_ms = ep_start.elapsed().as_millis() as u64;
            return Ok(traj);
        }
        state = new_state;
    }

    // Budget exhausted -> escalate.
    traj.outcome = Outcome::Escalate;
    traj.attempts = attempt;
    traj.escalated = false; // caller wires actual frontier fallback
    traj.total_latency_ms = ep_start.elapsed().as_millis() as u64;
    Ok(traj)
}

fn revert(verifier: &dyn Verifier, snapshots: &[(String, String)]) {
    for (file, content) in snapshots {
        let _ = verifier.write_file(file, content);
    }
}

#[allow(clippy::too_many_arguments)]
fn push_step(
    traj: &mut Trajectory,
    actor: Actor,
    attempt: u32,
    primary_code: Option<String>,
    error_count: usize,
    context: &str,
    proposal: Proposal,
    patch_applied: bool,
    patch_fuzzy: bool,
    error_count_after: Option<usize>,
    reverted: bool,
    model_latency_ms: u64,
    tokens_out: Option<u32>,
) {
    traj.steps.push(Step {
        attempt,
        actor,
        primary_code,
        error_count,
        context: context.to_string(),
        proposal,
        patch_applied,
        patch_fuzzy,
        error_count_after,
        reverted,
        model_latency_ms,
        tokens_out,
    });
}

/// Convenience: verify whether a patch string, applied to given file contents,
/// changes anything. Used by tests and datagen.
pub fn preview_apply(
    files: &[(String, String)],
    patch_text: &str,
    default_file: Option<&str>,
    limits: &PatchLimits,
) -> anyhow::Result<Vec<(String, String)>> {
    let blocks: Vec<EditBlock> = parse_blocks(patch_text, default_file)?;
    let mut out: Vec<(String, String)> = files.to_vec();
    for b in &blocks {
        let idx = out
            .iter()
            .position(|(f, _)| f == &b.file)
            .ok_or_else(|| anyhow::anyhow!("patch targets unknown file {}", b.file))?;
        let applied = apply_block(&out[idx].1, b, limits)?;
        out[idx].1 = applied.new_content;
    }
    Ok(out)
}
