//! End-to-end controller test: a scripted mock model drives the real
//! Verifier + patch engine over a real broken crate on disk, and the loop
//! must produce a compiling crate. Requires `cargo` on PATH.

use nano_controller::{run_episode, Config, Verifier, VerifyState};
use nano_model_client::{Message, ModelClient, ModelError, ModelResponse};
use nano_trajectory::{Outcome, Proposal};
use rust_repair_verifier::cargo_check;
use std::cell::RefCell;
use std::path::PathBuf;

/// Minimal rust-repair Verifier bound to a temp crate dir.
struct TestVerifier {
    root: PathBuf,
    last_file: RefCell<Option<String>>,
}

impl Verifier for TestVerifier {
    fn command(&self) -> &str {
        "cargo check"
    }
    fn verify(&self) -> anyhow::Result<VerifyState> {
        let res = cargo_check(&self.root)?;
        let ctx = match res.primary() {
            Some(d) => {
                *self.last_file.borrow_mut() = d.file.clone();
                format!(
                    "file: {}\n{}\n",
                    d.file.clone().unwrap_or_default(),
                    if d.rendered.is_empty() {
                        d.message.clone()
                    } else {
                        d.rendered.clone()
                    }
                )
            }
            None => "No errors.".to_string(),
        };
        Ok(VerifyState {
            passed: res.passed,
            error_count: res.error_count(),
            primary_code: res.primary().and_then(|d| d.code.clone()),
            codes: res.codes(),
            context: ctx,
        })
    }
    fn read_file(&self, rel: &str) -> anyhow::Result<String> {
        Ok(std::fs::read_to_string(self.root.join(rel))?)
    }
    fn write_file(&self, rel: &str, content: &str) -> anyhow::Result<()> {
        Ok(std::fs::write(self.root.join(rel), content)?)
    }
    fn default_file(&self) -> Option<String> {
        self.last_file.borrow().clone()
    }
}

/// Mock model that returns pre-scripted proposals in order.
struct ScriptedModel {
    responses: RefCell<Vec<Proposal>>,
}
impl ModelClient for ScriptedModel {
    fn name(&self) -> &str {
        "scripted-mock"
    }
    fn propose(&self, _messages: &[Message]) -> Result<ModelResponse, ModelError> {
        let mut r = self.responses.borrow_mut();
        let p = if r.is_empty() {
            Proposal {
                action: "no_fix".into(),
                patch: None,
                reason: None,
                confidence: 0.0,
            }
        } else {
            r.remove(0)
        };
        Ok(ModelResponse {
            proposal: p,
            raw: String::new(),
            latency_ms: 1,
            tokens_out: Some(1),
        })
    }
}

fn write_crate(dir: &std::path::Path, main_rs: &str) {
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"fixme\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[[bin]]\nname = \"fixme\"\npath = \"src/main.rs\"\n",
    )
    .unwrap();
    std::fs::write(dir.join("src/main.rs"), main_rs).unwrap();
}

#[test]
fn repairs_type_mismatch_end_to_end() {
    // Skip gracefully if cargo is unavailable.
    if std::process::Command::new("cargo")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("cargo not found; skipping e2e test");
        return;
    }

    let tmp = std::env::temp_dir().join(format!("nano_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    // Broken: assigning a &str to an i32 -> E0308.
    write_crate(
        &tmp,
        "fn main() {\n    let x: i32 = \"5\";\n    println!(\"{}\", x);\n}\n",
    );

    let verifier = TestVerifier {
        root: tmp.clone(),
        last_file: RefCell::new(None),
    };
    let model = ScriptedModel {
        responses: RefCell::new(vec![Proposal {
            action: "patch".into(),
            patch: Some(
                "file: src/main.rs\n<<<<<<< SEARCH\n    let x: i32 = \"5\";\n=======\n    let x: i32 = 5;\n>>>>>>> REPLACE\n"
                    .into(),
            ),
            reason: Some("string literal should be integer".into()),
            confidence: 0.9,
        }]),
    };

    let cfg = Config {
        max_attempts: 4,
        ..Default::default()
    };
    let traj = run_episode(&verifier, &model, &cfg, "e2e-case", "e2e-1").unwrap();

    assert_eq!(
        traj.outcome,
        Outcome::Success,
        "expected repair to succeed; steps={:?}",
        traj.steps.len()
    );
    assert_eq!(
        traj.initial_codes.first().map(|s| s.as_str()),
        Some("E0308")
    );
    // Confirm the file on disk actually compiles now.
    let final_check = cargo_check(&tmp).unwrap();
    assert!(final_check.passed);

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn regression_guard_reverts_bad_patch() {
    if std::process::Command::new("cargo")
        .arg("--version")
        .output()
        .is_err()
    {
        return;
    }
    let tmp = std::env::temp_dir().join(format!("nano_e2e_reg_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    write_crate(
        &tmp,
        "fn main() {\n    let x: i32 = \"5\";\n    println!(\"{}\", x);\n}\n",
    );

    let verifier = TestVerifier {
        root: tmp.clone(),
        last_file: RefCell::new(None),
    };
    // First proposal makes things worse (introduces a new error), then a good one.
    let model = ScriptedModel {
        responses: RefCell::new(vec![
            Proposal {
                action: "patch".into(),
                patch: Some(
                    "file: src/main.rs\n<<<<<<< SEARCH\n    let x: i32 = \"5\";\n=======\n    let x: i32 = \"5\";\n    let y: i32 = \"bad\";\n>>>>>>> REPLACE\n".into(),
                ),
                reason: Some("makes it worse".into()),
                confidence: 0.5,
            },
            Proposal {
                action: "patch".into(),
                patch: Some(
                    "file: src/main.rs\n<<<<<<< SEARCH\n    let x: i32 = \"5\";\n=======\n    let x: i32 = 5;\n>>>>>>> REPLACE\n".into(),
                ),
                reason: Some("real fix".into()),
                confidence: 0.9,
            },
        ]),
    };
    let cfg = Config {
        max_attempts: 4,
        regression_guard: true,
        ..Default::default()
    };
    let traj = run_episode(&verifier, &model, &cfg, "reg-case", "reg-1").unwrap();

    // Step 0 should have been reverted; final outcome success on step 1.
    assert!(traj.steps[0].reverted, "first bad patch must be reverted");
    assert_eq!(traj.outcome, Outcome::Success);
    std::fs::remove_dir_all(&tmp).ok();
}
