//! Template recipe verifier — a NO-OP skeleton.
//!
//! Copy `recipes/_template` to `recipes/<your-recipe>/` and implement the four
//! TODOs below. This crate compiles and runs as-is (it reports "no errors"), so
//! you always have a working baseline to edit from.

use nano_controller::{Verifier, VerifyState};
use std::path::{Path, PathBuf};

/// A recipe Verifier bound to one workspace directory.
///
/// TODO(1): replace the no-op `verify()` with a call to YOUR tool (compiler,
/// linter, type checker, test runner, schema/policy validator, ...).
pub struct TemplateVerifier {
    #[allow(dead_code)]
    root: PathBuf,
}

impl TemplateVerifier {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

impl Verifier for TemplateVerifier {
    fn command(&self) -> &str {
        // TODO(2): the human-readable name of your tool, e.g. "sqlfluff lint".
        "template-check"
    }

    fn verify(&self) -> anyhow::Result<VerifyState> {
        // TODO(1): run your tool against `self.root`, parse its diagnostics, and
        // build the model-facing `context` string from the PRIMARY failure.
        // The no-op below always reports success so the skeleton runs end to end.
        Ok(VerifyState {
            passed: true,
            error_count: 0,
            primary_code: None,
            codes: vec![],
            context: "no errors (template no-op verifier)".into(),
        })
    }

    fn read_file(&self, rel: &str) -> anyhow::Result<String> {
        Ok(std::fs::read_to_string(self.root.join(rel))?)
    }

    fn write_file(&self, rel: &str, content: &str) -> anyhow::Result<()> {
        Ok(std::fs::write(self.root.join(rel), content)?)
    }

    fn default_file(&self) -> Option<String> {
        // TODO(3): the file a patch block targets when it omits its own hint
        // (usually the file of the primary diagnostic from `verify()`).
        None
    }
}

/// Convenience for callers that only have a path.
pub fn verifier_for(root: &Path) -> TemplateVerifier {
    TemplateVerifier::new(root.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_verifier_reports_passing() {
        let v = TemplateVerifier::new(PathBuf::from("."));
        let state = v.verify().unwrap();
        assert!(state.passed);
        assert_eq!(state.error_count, 0);
    }
}
