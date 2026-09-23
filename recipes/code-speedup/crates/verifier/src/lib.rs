//! Rust adapter for the private Python code-speed harness.
//!
//! Rust owns workspace patching and model-facing context. The Python bridge owns
//! all private problem metadata and executes the existing sandbox/correctness/
//! Cachegrind oracle. The bridge output is intentionally a small, sanitized JSON
//! response rather than a serialized problem record.

use anyhow::{bail, Context, Result};
use nano_controller::{Config, RecipeSpec, Verifier, VerifyState};
use nano_trajectory::Actor;
use serde::Deserialize;
use serde_json::Value;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

pub const DEFAULT_CANDIDATE_FILE: &str = "candidate.py";

/// All inputs needed to start one private Python bridge process.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub python: PathBuf,
    /// Absolute path to `recipes/code-speedup/harness/bridge.py`.
    pub bridge: PathBuf,
    /// Recipe root added to `PYTHONPATH`, never supplied by a model.
    pub pythonpath: PathBuf,
    pub dataset: PathBuf,
    pub problem_id: String,
    pub candidate_file: String,
    /// Optional privileged, network-disabled container for the private bridge.
    /// Required on hosts that cannot create Bubblewrap network namespaces.
    pub container_image: Option<String>,
}

impl BridgeConfig {
    pub fn new(
        python: PathBuf,
        bridge: PathBuf,
        pythonpath: PathBuf,
        dataset: PathBuf,
        problem_id: String,
    ) -> Self {
        Self {
            python,
            bridge,
            pythonpath,
            dataset,
            problem_id,
            candidate_file: DEFAULT_CANDIDATE_FILE.to_owned(),
            container_image: None,
        }
    }
}

#[derive(Debug, Clone)]
struct PublicProblem {
    family_id: String,
    function_name: String,
    reference_source: String,
}

#[derive(Debug, Deserialize)]
struct BridgeResponse {
    passed: bool,
    status: String,
    correct: bool,
    reward: Option<f64>,
    reference_instruction_count: Option<u64>,
    candidate_instruction_count: Option<u64>,
}

/// A verifier bound to a candidate workspace and one immutable private record.
pub struct SpeedupVerifier {
    root: PathBuf,
    bridge: BridgeConfig,
    public_problem: PublicProblem,
}

impl SpeedupVerifier {
    pub fn new(root: PathBuf, bridge: BridgeConfig) -> Result<Self> {
        validate_relative(&bridge.candidate_file)?;
        let public_problem = load_public_problem(&bridge.dataset, &bridge.problem_id)?;
        Ok(Self {
            root,
            bridge,
            public_problem,
        })
    }

    fn candidate_path(&self) -> Result<PathBuf> {
        workspace_path(&self.root, &self.bridge.candidate_file)
    }

    fn run_bridge(&self) -> Result<BridgeResponse> {
        let candidate = self.candidate_path()?;
        let output = if let Some(image) = &self.bridge.container_image {
            let dataset_rel = self
                .bridge
                .dataset
                .strip_prefix(&self.bridge.pythonpath)
                .context("container bridge dataset must be inside recipe root")?;
            let bridge_rel = self
                .bridge
                .bridge
                .strip_prefix(&self.bridge.pythonpath)
                .context("container bridge script must be inside recipe root")?;
            Command::new("docker")
                .args([
                    "run",
                    "--rm",
                    "--privileged",
                    "--network",
                    "none",
                    "-e",
                    "PYTHONPATH=/repo",
                    "-v",
                ])
                .arg(format!("{}:/repo:ro", self.bridge.pythonpath.display()))
                .arg("-v")
                .arg(format!(
                    "{}:/candidate:ro",
                    candidate
                        .parent()
                        .context("candidate must have a parent directory")?
                        .display()
                ))
                .arg(image)
                .arg("python3")
                .arg(format!("/repo/{}", bridge_rel.display()))
                .arg("--dataset")
                .arg(format!("/repo/{}", dataset_rel.display()))
                .arg("--problem-id")
                .arg(&self.bridge.problem_id)
                .arg("--candidate")
                .arg(format!("/candidate/{}", self.bridge.candidate_file))
                .output()
                .context("starting containerized private Python speedup bridge")?
        } else {
            Command::new(&self.bridge.python)
                .arg(&self.bridge.bridge)
                .arg("--dataset")
                .arg(&self.bridge.dataset)
                .arg("--problem-id")
                .arg(&self.bridge.problem_id)
                .arg("--candidate")
                .arg(candidate)
                .env("PYTHONPATH", &self.bridge.pythonpath)
                .output()
                .context("starting private Python speedup bridge")?
        };
        // The bridge returns a sanitized JSON status even on harness failure.
        // Never surface stderr: it can contain private sandbox/oracle details.
        let parsed: BridgeResponse = serde_json::from_slice(&output.stdout)
            .map_err(|_| anyhow::anyhow!("private bridge returned no valid result"))?;
        Ok(parsed)
    }

    fn context(&self, candidate: &str, response: &BridgeResponse) -> String {
        let status = safe_status(&response.status);
        format!(
            "family: {}\nfunction: {}\nstatus: {}\ncorrect: {}\nreward: {}\n\
reference_instruction_count: {}\ncandidate_instruction_count: {}\n\
--- slow reference ---\n{}\n--- candidate.py ---\n{}\n\
Return one JSON proposal with action \"patch\" and exact SEARCH/REPLACE blocks. \
Rewrite candidate.py to be correct and use fewer instruction references.",
            self.public_problem.family_id,
            self.public_problem.function_name,
            status,
            response.correct,
            response
                .reward
                .map_or_else(|| "unavailable".to_owned(), |value| value.to_string()),
            response
                .reference_instruction_count
                .map_or_else(|| "unavailable".to_owned(), |value| value.to_string()),
            response
                .candidate_instruction_count
                .map_or_else(|| "unavailable".to_owned(), |value| value.to_string()),
            self.public_problem.reference_source,
            candidate,
        )
    }
}

impl Verifier for SpeedupVerifier {
    fn command(&self) -> &str {
        "python harness/bridge.py (sandbox + correctness + cachegrind)"
    }

    fn verify(&self) -> Result<VerifyState> {
        let candidate =
            std::fs::read_to_string(self.candidate_path()?).context("reading candidate.py")?;
        let response = self.run_bridge()?;
        let status = safe_status(&response.status);
        let passed = response.passed && status == "passed";
        Ok(VerifyState {
            passed,
            // The generic controller reverts only when this score increases.
            // Preserve a correct-but-slow candidate (1) over an invalid or
            // incorrect rewrite (2), and rank harness failures worst (3).
            error_count: status_severity(status, response.correct),
            primary_code: (!passed).then(|| format!("SPEEDUP_{status}")),
            codes: (!passed)
                .then(|| format!("SPEEDUP_{status}"))
                .into_iter()
                .collect(),
            context: self.context(&candidate, &response),
        })
    }

    fn read_file(&self, rel: &str) -> Result<String> {
        Ok(std::fs::read_to_string(workspace_path(&self.root, rel)?)?)
    }

    fn write_file(&self, rel: &str, content: &str) -> Result<()> {
        std::fs::write(workspace_path(&self.root, rel)?, content)?;
        Ok(())
    }

    fn default_file(&self) -> Option<String> {
        Some(self.bridge.candidate_file.clone())
    }
}

/// `RecipeSpec` implementation used by the CLI and fake-model tests.
pub struct SpeedupRecipe {
    pub bridge: BridgeConfig,
    pub max_attempts: u32,
    pub actor: Actor,
}

impl RecipeSpec for SpeedupRecipe {
    fn name(&self) -> &str {
        "code-speedup"
    }

    fn verifier(&self, workspace: &Path) -> Result<Box<dyn Verifier>> {
        Ok(Box::new(SpeedupVerifier::new(
            workspace.to_path_buf(),
            self.bridge.clone(),
        )?))
    }

    fn system_prompt(&self) -> String {
        include_str!("../../../prompt/system.txt")
            .trim_end()
            .to_owned()
    }

    fn base_config(&self) -> Config {
        Config {
            max_attempts: self.max_attempts,
            actor: self.actor,
            ..Default::default()
        }
    }
}

fn load_public_problem(dataset: &Path, problem_id: &str) -> Result<PublicProblem> {
    let contents = std::fs::read_to_string(dataset)
        .with_context(|| format!("reading dataset {}", dataset.display()))?;
    for line in contents.lines().filter(|line| !line.trim().is_empty()) {
        let row: Value = serde_json::from_str(line).context("parsing dataset JSONL")?;
        if row.get("id").and_then(Value::as_str) != Some(problem_id) {
            continue;
        }
        if row.pointer("/verification/status").and_then(Value::as_str) != Some("passed") {
            bail!("selected problem is not verifier-gated")
        }
        return Ok(PublicProblem {
            family_id: string_field(&row, "family_id")?,
            function_name: string_field(&row, "function_name")?,
            reference_source: string_field(&row, "reference_source")?,
        });
    }
    bail!("problem id not found in dataset")
}

fn string_field(row: &Value, name: &str) -> Result<String> {
    row.get(name)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("selected problem has no {name}"))
}

fn safe_status(value: &str) -> &str {
    match value {
        "passed"
        | "no_speedup"
        | "mismatch"
        | "candidate_error"
        | "infrastructure_error"
        | "bridge_error" => value,
        _ => "bridge_error",
    }
}

fn status_severity(status: &str, correct: bool) -> usize {
    match status {
        "passed" if correct => 0,
        "no_speedup" if correct => 1,
        "mismatch" | "candidate_error" => 2,
        _ => 3,
    }
}

fn validate_relative(rel: &str) -> Result<()> {
    let path = Path::new(rel);
    if path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir))
    {
        bail!("candidate file must be a workspace-relative path")
    }
    Ok(())
}

fn workspace_path(root: &Path, rel: &str) -> Result<PathBuf> {
    validate_relative(rel)?;
    Ok(root.join(rel))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_candidate_path_traversal() {
        assert!(validate_relative("../candidate.py").is_err());
        assert!(validate_relative("/tmp/candidate.py").is_err());
        assert!(validate_relative("candidate.py").is_ok());
    }

    #[test]
    fn status_is_allowlisted() {
        assert_eq!(safe_status("no_speedup"), "no_speedup");
        assert_eq!(safe_status("private input leaked"), "bridge_error");
    }

    #[test]
    fn invalid_rewrites_rank_worse_than_correct_slow_code() {
        assert_eq!(status_severity("passed", true), 0);
        assert_eq!(status_severity("no_speedup", true), 1);
        assert_eq!(status_severity("mismatch", false), 2);
        assert_eq!(status_severity("candidate_error", false), 2);
        assert_eq!(status_severity("bridge_error", false), 3);
    }

    #[test]
    fn shared_prompt_declares_the_patch_and_sandbox_contracts() {
        let prompt = include_str!("../../../prompt/system.txt").trim_end();
        assert!(prompt.contains("<<<<<<< SEARCH"));
        assert!(prompt.contains(">>>>>>> REPLACE"));
        assert!(prompt.contains("All imports are forbidden"));
    }
}
