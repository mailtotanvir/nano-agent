//! Rust compiler verifier: runs `cargo check --message-format=json` and parses
//! the diagnostic stream into a structured, controller-friendly form.
//!
//! The verifier — not the model — decides whether a repair succeeded (spec §5).

use serde::Deserialize;
use std::path::Path;
use std::process::Command;

/// One compiler diagnostic reduced to what the controller/model need.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// e.g. "E0308". None for diagnostics without a code (some syntax errors).
    pub code: Option<String>,
    pub message: String,
    /// Primary source file (relative to the crate root when possible).
    pub file: Option<String>,
    pub line_start: usize,
    pub line_end: usize,
    /// The rendered diagnostic as rustc would print it (used for model context).
    pub rendered: String,
}

/// Outcome of one verifier run.
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub errors: Vec<Diagnostic>,
    pub warnings: usize,
    /// True when there are zero error-level diagnostics.
    pub passed: bool,
}

impl CheckResult {
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
    /// Error codes in order (with duplicates), for trajectory logging.
    pub fn codes(&self) -> Vec<String> {
        self.errors.iter().filter_map(|d| d.code.clone()).collect()
    }
    /// First error diagnostic, treated as the primary target.
    pub fn primary(&self) -> Option<&Diagnostic> {
        self.errors.first()
    }
}

// --- cargo JSON message shapes (subset) ---

#[derive(Deserialize)]
struct CargoMessage {
    reason: String,
    #[serde(default)]
    message: Option<RustcDiagnostic>,
}

#[derive(Deserialize)]
struct RustcDiagnostic {
    #[serde(default)]
    message: String,
    #[serde(default)]
    code: Option<RustcCode>,
    #[serde(default)]
    level: String,
    #[serde(default)]
    spans: Vec<RustcSpan>,
    #[serde(default)]
    rendered: Option<String>,
}

#[derive(Deserialize)]
struct RustcCode {
    code: String,
}

#[derive(Deserialize)]
struct RustcSpan {
    #[serde(default)]
    file_name: String,
    #[serde(default)]
    is_primary: bool,
    #[serde(default)]
    line_start: usize,
    #[serde(default)]
    line_end: usize,
}

/// Resolve the cargo binary: prefer PATH, else fall back to the standard rustup
/// location so non-login shells (cron, ssh subprocesses) still work.
fn cargo_bin() -> String {
    if let Ok(explicit) = std::env::var("CARGO") {
        if !explicit.is_empty() {
            return explicit;
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let candidate = std::path::Path::new(&home).join(".cargo/bin/cargo");
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    "cargo".to_string()
}

/// Run `cargo check --message-format=json` in `crate_dir` and parse results.
pub fn cargo_check(crate_dir: impl AsRef<Path>) -> anyhow::Result<CheckResult> {
    let output = Command::new(cargo_bin())
        .args(["check", "--message-format=json", "--quiet"])
        .current_dir(crate_dir.as_ref())
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(parse_cargo_json(&stdout))
}

/// Parse a captured cargo JSON stream (one JSON object per line).
pub fn parse_cargo_json(stream: &str) -> CheckResult {
    let mut errors = Vec::new();
    let mut warnings = 0usize;
    for line in stream.lines() {
        let line = line.trim();
        if line.is_empty() || !line.starts_with('{') {
            continue;
        }
        let msg: CargoMessage = match serde_json::from_str(line) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if msg.reason != "compiler-message" {
            continue;
        }
        let Some(diag) = msg.message else { continue };
        match diag.level.as_str() {
            "error" => {
                let primary = diag
                    .spans
                    .iter()
                    .find(|s| s.is_primary)
                    .or_else(|| diag.spans.first());
                errors.push(Diagnostic {
                    code: diag.code.map(|c| c.code),
                    message: diag.message,
                    file: primary.map(|s| s.file_name.clone()),
                    line_start: primary.map(|s| s.line_start).unwrap_or(0),
                    line_end: primary.map(|s| s.line_end).unwrap_or(0),
                    rendered: diag.rendered.unwrap_or_default(),
                });
            }
            "warning" => warnings += 1,
            _ => {}
        }
    }
    let passed = errors.is_empty();
    CheckResult {
        errors,
        warnings,
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_type_mismatch() {
        // Minimal representative cargo compiler-message line for E0308.
        let line = r#"{"reason":"compiler-message","message":{"message":"mismatched types","code":{"code":"E0308"},"level":"error","spans":[{"file_name":"src/main.rs","is_primary":true,"line_start":2,"line_end":2}],"rendered":"error[E0308]: mismatched types\n --> src/main.rs:2:18\n"}}"#;
        let res = parse_cargo_json(line);
        assert!(!res.passed);
        assert_eq!(res.error_count(), 1);
        let d = res.primary().unwrap();
        assert_eq!(d.code.as_deref(), Some("E0308"));
        assert_eq!(d.file.as_deref(), Some("src/main.rs"));
        assert_eq!(d.line_start, 2);
        assert!(d.rendered.contains("mismatched types"));
    }

    #[test]
    fn passes_when_no_errors() {
        let stream = r#"{"reason":"compiler-message","message":{"message":"unused variable","code":{"code":"unused_variables"},"level":"warning","spans":[]}}
{"reason":"compiler-artifact","package_id":"x"}"#;
        let res = parse_cargo_json(stream);
        assert!(res.passed);
        assert_eq!(res.warnings, 1);
        assert_eq!(res.error_count(), 0);
    }

    #[test]
    fn ignores_non_json_lines() {
        let stream = "Compiling foo v0.1.0\n   Checking...\n";
        let res = parse_cargo_json(stream);
        assert!(res.passed);
    }
}
