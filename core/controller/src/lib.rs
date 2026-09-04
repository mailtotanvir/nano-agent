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

/// Everything a recipe owns. The generic [`drive`] entrypoint turns a
/// `RecipeSpec` plus model client(s) into a running episode. Implement this once
/// per recipe; the loop, patch engine, model backends, and trajectory schema are
/// all provided by `core/`.
pub trait RecipeSpec {
    /// Recipe id, e.g. "rust-repair".
    fn name(&self) -> &str;
    /// Build the recipe's [`Verifier`] for a workspace path.
    fn verifier(&self, workspace: &std::path::Path) -> anyhow::Result<Box<dyn Verifier>>;
    /// The recipe's domain-specific system prompt.
    fn system_prompt(&self) -> String;
    /// Escalation policy for this recipe (default: disabled).
    fn escalation_policy(&self) -> EscalationPolicy {
        EscalationPolicy::default()
    }
    /// Base controller config (max_attempts, patch_limits, etc.). The driver
    /// overlays `system_prompt` and `escalation` from this spec on top.
    fn base_config(&self) -> Config {
        Config::default()
    }
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
    /// Recipe-configurable frontier escalation policy.
    pub escalation: EscalationPolicy,
}

/// Recipe-configurable frontier escalation. When `enabled`, the controller runs
/// up to `max_frontier_attempts` frontier proposals AFTER the tiny loop's budget
/// is exhausted, using the escalation model supplied to
/// [`run_episode_with_escalation`].
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    pub enabled: bool,
    pub max_frontier_attempts: u32,
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_frontier_attempts: 1,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            patch_limits: PatchLimits::default(),
            regression_guard: true,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            actor: Actor::Tiny,
            escalation: EscalationPolicy::default(),
        }
    }
}

pub const DEFAULT_SYSTEM_PROMPT: &str = GENERIC_SYSTEM_PROMPT;

/// Domain-neutral system prompt. Describes ONLY the response contract and the
/// SEARCH/REPLACE patch format — no mention of any specific language or tool.
/// Recipes override `Config.system_prompt` with a domain-specific prompt.
pub const GENERIC_SYSTEM_PROMPT: &str = "You are a code-repair agent. You are given one error diagnostic and the relevant source. Respond with ONLY a single JSON object matching this contract: {\"action\": \"patch\"|\"no_fix\"|\"escalate\", \"patch\": \"<SEARCH/REPLACE blocks or null>\", \"reason\": \"<short>\", \"confidence\": <0.0-1.0>}. For a patch, the \"patch\" field must contain one or more blocks in this exact format:\nfile: <path>\n<<<<<<< SEARCH\n<exact existing lines>\n=======\n<replacement lines>\n>>>>>>> REPLACE\nMake the smallest change that fixes the reported error. If you cannot fix it, use action \"no_fix\" or \"escalate\".";

/// Run one repair episode (tiny model only, no escalation). Mutates the
/// workspace through the verifier. Thin wrapper over
/// [`run_episode_with_escalation`] with no escalation model.
pub fn run_episode(
    verifier: &dyn Verifier,
    model: &dyn ModelClient,
    cfg: &Config,
    case_id: &str,
    traj_id: &str,
) -> anyhow::Result<Trajectory> {
    run_episode_with_escalation(verifier, model, None, cfg, case_id, traj_id)
}

/// Outcome of a single proposing phase (tiny or frontier).
enum PhaseOutcome {
    /// The phase reached a terminal state; `traj.outcome` carries it.
    Terminal(Outcome),
    /// The phase used its whole budget without resolving the failure.
    Exhausted,
}

/// Run the tiny model, then (if the recipe's escalation policy is enabled AND an
/// escalation model is supplied) run a bounded frontier phase when the tiny loop
/// exhausts its budget. Each frontier attempt is recorded as a `Step` with
/// `Actor::Frontier`; `traj.escalated` is set true iff the frontier phase ran.
pub fn run_episode_with_escalation(
    verifier: &dyn Verifier,
    model: &dyn ModelClient,
    escalation_model: Option<&dyn ModelClient>,
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

    // Tiny phase.
    let phase = run_phase(
        verifier,
        model,
        cfg,
        cfg.actor,
        cfg.max_attempts,
        0,
        &mut traj,
        &mut state,
    )?;
    if let PhaseOutcome::Terminal(outcome) = phase {
        traj.outcome = outcome;
        traj.total_latency_ms = ep_start.elapsed().as_millis() as u64;
        return Ok(traj);
    }

    // Tiny budget exhausted. Escalate to a frontier model if configured.
    if cfg.escalation.enabled {
        if let Some(front) = escalation_model {
            let base = traj.attempts;
            let phase2 = run_phase(
                verifier,
                front,
                cfg,
                Actor::Frontier,
                cfg.escalation.max_frontier_attempts,
                base,
                &mut traj,
                &mut state,
            )?;
            traj.escalated = true;
            traj.outcome = match phase2 {
                PhaseOutcome::Terminal(outcome) => outcome,
                PhaseOutcome::Exhausted => Outcome::Escalate,
            };
            traj.total_latency_ms = ep_start.elapsed().as_millis() as u64;
            return Ok(traj);
        }
    }

    // No escalation wired: preserve the original "caller escalates" contract.
    traj.outcome = Outcome::Escalate;
    traj.escalated = false;
    traj.total_latency_ms = ep_start.elapsed().as_millis() as u64;
    Ok(traj)
}

/// Drive one episode for a recipe. Builds the recipe's verifier for `workspace`,
/// overlays the recipe's system prompt and escalation policy onto its base
/// config, and runs the tiny model (escalating to `escalation_model` if the
/// recipe's policy is enabled). This is the one-call entrypoint a recipe CLI uses.
pub fn drive(
    spec: &dyn RecipeSpec,
    workspace: &std::path::Path,
    tiny: &dyn ModelClient,
    escalation_model: Option<&dyn ModelClient>,
    case_id: &str,
    traj_id: &str,
) -> anyhow::Result<Trajectory> {
    let verifier = spec.verifier(workspace)?;
    let mut cfg = spec.base_config();
    cfg.system_prompt = spec.system_prompt();
    cfg.escalation = spec.escalation_policy();
    run_episode_with_escalation(
        verifier.as_ref(),
        tiny,
        escalation_model,
        &cfg,
        case_id,
        traj_id,
    )
}

/// Run one bounded proposing phase against the current verifier state, pushing a
/// `Step` per attempt. Shared by the tiny and frontier phases. Step `attempt`
/// numbers continue from `base_attempt`. Sets `traj.attempts` on every return.
#[allow(clippy::too_many_arguments)]
fn run_phase(
    verifier: &dyn Verifier,
    model: &dyn ModelClient,
    cfg: &Config,
    actor: Actor,
    budget: u32,
    base_attempt: u32,
    traj: &mut Trajectory,
    state: &mut VerifyState,
) -> anyhow::Result<PhaseOutcome> {
    let mut local = 0u32;
    while local < budget {
        let attempt = base_attempt + local;

        // Build context from the current primary failure.
        let mut messages = vec![Message::system(cfg.system_prompt.clone())];
        // Include prior step summaries as assistant turns for recovery context.
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
                traj,
                actor,
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
            traj.attempts = attempt + 1;
            return Ok(PhaseOutcome::Terminal(Outcome::Escalate));
        }
        if proposal.action == "no_fix" || !ok {
            push_step(
                traj,
                actor,
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
            local += 1;
            traj.attempts = base_attempt + local;
            if !ok {
                continue;
            }
            return Ok(PhaseOutcome::Terminal(Outcome::NoFix));
        }

        // action == "patch": parse and apply.
        let patch_text = proposal.patch.clone().unwrap_or_default();
        let blocks = match parse_blocks(&patch_text, verifier.default_file().as_deref()) {
            Ok(b) if !b.is_empty() => b,
            _ => {
                push_step(
                    traj,
                    actor,
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
                local += 1;
                traj.attempts = base_attempt + local;
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
                traj,
                actor,
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
            local += 1;
            traj.attempts = base_attempt + local;
            continue;
        }

        // Re-verify.
        let new_state = verifier.verify()?;
        let regressed = cfg.regression_guard && new_state.error_count > state.error_count;
        if regressed {
            revert(verifier, &snapshots);
            push_step(
                traj,
                actor,
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
            local += 1;
            traj.attempts = base_attempt + local;
            // state unchanged (reverted)
            continue;
        }

        push_step(
            traj,
            actor,
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
        local += 1;
        traj.attempts = base_attempt + local;

        if new_state.passed {
            return Ok(PhaseOutcome::Terminal(Outcome::Success));
        }
        *state = new_state;
    }

    traj.attempts = base_attempt + local;
    Ok(PhaseOutcome::Exhausted)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escalation_policy_defaults_to_disabled() {
        let p = EscalationPolicy::default();
        assert!(!p.enabled);
        assert_eq!(p.max_frontier_attempts, 1);
    }

    #[test]
    fn config_default_has_disabled_escalation() {
        let c = Config::default();
        assert!(!c.escalation.enabled);
    }

    #[test]
    fn generic_prompt_is_domain_neutral() {
        let p = GENERIC_SYSTEM_PROMPT.to_lowercase();
        assert!(!p.contains("rust"));
        assert!(!p.contains("cargo"));
        assert!(GENERIC_SYSTEM_PROMPT.contains("SEARCH"));
        assert!(GENERIC_SYSTEM_PROMPT.contains("REPLACE"));
    }

    use nano_model_client::{ModelError, ModelResponse};
    use std::cell::RefCell;

    /// In-memory verifier: one file "f", passes only once its content contains
    /// the marker "FRONTIER_FIX".
    struct MockVerifier {
        content: RefCell<String>,
    }
    impl Verifier for MockVerifier {
        fn command(&self) -> &str {
            "mock-check"
        }
        fn verify(&self) -> anyhow::Result<VerifyState> {
            let passed = self.content.borrow().contains("FRONTIER_FIX");
            Ok(VerifyState {
                passed,
                error_count: if passed { 0 } else { 1 },
                primary_code: if passed { None } else { Some("E_MOCK".into()) },
                codes: if passed {
                    vec![]
                } else {
                    vec!["E_MOCK".into()]
                },
                context: "mock failure".into(),
            })
        }
        fn read_file(&self, _rel: &str) -> anyhow::Result<String> {
            Ok(self.content.borrow().clone())
        }
        fn write_file(&self, _rel: &str, content: &str) -> anyhow::Result<()> {
            *self.content.borrow_mut() = content.to_string();
            Ok(())
        }
        fn default_file(&self) -> Option<String> {
            Some("f".into())
        }
    }

    /// Model that always returns a fixed patch string.
    struct MockModel {
        name: String,
        patch: String,
    }
    impl ModelClient for MockModel {
        fn name(&self) -> &str {
            &self.name
        }
        fn propose(&self, _messages: &[Message]) -> Result<ModelResponse, ModelError> {
            Ok(ModelResponse {
                proposal: Proposal {
                    action: "patch".into(),
                    patch: Some(self.patch.clone()),
                    reason: None,
                    confidence: 1.0,
                },
                raw: String::new(),
                latency_ms: 0,
                tokens_out: None,
            })
        }
    }

    fn block(search: &str, replace: &str) -> String {
        format!("file: f\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE")
    }

    #[test]
    fn escalation_runs_frontier_after_budget_and_records_truthfully() {
        let verifier = MockVerifier {
            content: RefCell::new("START".into()),
        };
        // Tiny proposes a no-op that applies but never introduces the marker.
        let tiny = MockModel {
            name: "tiny".into(),
            patch: block("START", "STILL_BROKEN"),
        };
        // Frontier proposes the winning patch (matches whatever tiny left).
        let frontier = MockModel {
            name: "frontier".into(),
            patch: block("STILL_BROKEN", "FRONTIER_FIX"),
        };
        let cfg = Config {
            max_attempts: 2,
            escalation: EscalationPolicy {
                enabled: true,
                max_frontier_attempts: 1,
            },
            ..Default::default()
        };
        let traj =
            run_episode_with_escalation(&verifier, &tiny, Some(&frontier), &cfg, "c", "t").unwrap();
        assert!(traj.escalated, "frontier phase should have run");
        assert_eq!(traj.outcome, Outcome::Success);
        assert_eq!(
            traj.steps.last().unwrap().actor,
            Actor::Frontier,
            "final step must be the frontier attempt"
        );
    }

    #[test]
    fn no_escalation_when_disabled_returns_escalate_outcome() {
        let verifier = MockVerifier {
            content: RefCell::new("START".into()),
        };
        let tiny = MockModel {
            name: "tiny".into(),
            patch: block("START", "STILL_BROKEN"),
        };
        let frontier = MockModel {
            name: "frontier".into(),
            patch: block("STILL_BROKEN", "FRONTIER_FIX"),
        };
        // escalation disabled (default)
        let cfg = Config {
            max_attempts: 2,
            ..Default::default()
        };
        let traj =
            run_episode_with_escalation(&verifier, &tiny, Some(&frontier), &cfg, "c", "t").unwrap();
        assert!(!traj.escalated);
        assert_eq!(traj.outcome, Outcome::Escalate);
        assert!(traj.steps.iter().all(|s| s.actor == Actor::Tiny));
    }

    /// Minimal RecipeSpec whose verifier passes immediately (no-op recipe).
    struct DummyRecipe;
    struct PassingVerifier;
    impl Verifier for PassingVerifier {
        fn command(&self) -> &str {
            "dummy"
        }
        fn verify(&self) -> anyhow::Result<VerifyState> {
            Ok(VerifyState {
                passed: true,
                error_count: 0,
                primary_code: None,
                codes: vec![],
                context: "ok".into(),
            })
        }
        fn read_file(&self, _rel: &str) -> anyhow::Result<String> {
            Ok(String::new())
        }
        fn write_file(&self, _rel: &str, _content: &str) -> anyhow::Result<()> {
            Ok(())
        }
        fn default_file(&self) -> Option<String> {
            None
        }
    }
    impl RecipeSpec for DummyRecipe {
        fn name(&self) -> &str {
            "dummy"
        }
        fn verifier(&self, _workspace: &std::path::Path) -> anyhow::Result<Box<dyn Verifier>> {
            Ok(Box::new(PassingVerifier))
        }
        fn system_prompt(&self) -> String {
            "dummy prompt".into()
        }
    }

    #[test]
    fn recipe_spec_defaults_and_drive_run() {
        let spec = DummyRecipe;
        assert_eq!(spec.name(), "dummy");
        assert!(!spec.system_prompt().is_empty());
        assert!(!spec.escalation_policy().enabled);

        // drive() on an already-passing workspace yields immediate success.
        let tiny = MockModel {
            name: "tiny".into(),
            patch: String::new(),
        };
        let traj = drive(&spec, std::path::Path::new("."), &tiny, None, "c", "t").unwrap();
        assert_eq!(traj.outcome, Outcome::Success);
    }
}
