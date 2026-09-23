//! A fake local model drives the generic controller through the real recipe
//! verifier boundary. The bridge is scripted here so this test stays CPU/local
//! and does not require Bubblewrap or Cachegrind; Python bridge behavior has its
//! own focused tests.

use code_speedup_verifier::{BridgeConfig, SpeedupRecipe};
use nano_controller::drive;
use nano_model_client::{Message, ModelClient, ModelError, ModelResponse};
use nano_trajectory::{Outcome, Proposal};
use std::cell::RefCell;
use std::path::Path;

struct FakeModel {
    seen_context: RefCell<String>,
}

impl ModelClient for FakeModel {
    fn name(&self) -> &str {
        "fake-local-model"
    }

    fn propose(&self, messages: &[Message]) -> Result<ModelResponse, ModelError> {
        *self.seen_context.borrow_mut() = messages.last().expect("user context").content.clone();
        Ok(ModelResponse {
            proposal: Proposal {
                action: "patch".into(),
                patch: Some(
                    "file: candidate.py\n<<<<<<< SEARCH\ndef solve(xs):\n    for i in range(len(xs)):\n        for j in range(i):\n            if xs[i] == xs[j]:\n                return True\n    return False\n=======\ndef solve(xs):\n    return len(set(xs)) != len(xs)\n>>>>>>> REPLACE\n".into(),
                ),
                reason: Some("use a set".into()),
                confidence: 1.0,
            },
            raw: String::new(),
            latency_ms: 0,
            tokens_out: None,
        })
    }
}

fn write(path: &Path, content: &str) {
    std::fs::write(path, content).unwrap();
}

#[test]
fn fake_model_drive_patches_candidate_and_succeeds_without_private_context() {
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        return;
    }
    let root = std::env::temp_dir().join(format!("code_speedup_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let slow = "def solve(xs):\n    for i in range(len(xs)):\n        for j in range(i):\n            if xs[i] == xs[j]:\n                return True\n    return False\n";
    write(&root.join("candidate.py"), slow);
    write(
        &root.join("records.jsonl"),
        &format!("{{\"id\":\"fixture\",\"family_id\":\"hash-membership\",\"function_name\":\"solve\",\"reference_source\":{},\"edge_inputs\":[[[999]]],\"benchmark_variants\":[{{\"inputs\":[[123]]}}],\"verification\":{{\"status\":\"passed\"}}}}\n", serde_json::to_string(slow).unwrap()),
    );
    let bridge = root.join("fake_bridge.py");
    write(
        &bridge,
        r#"import json, sys
candidate = open(sys.argv[sys.argv.index('--candidate') + 1]).read()
passed = 'return len(set(xs)) != len(xs)' in candidate
print(json.dumps({'passed': passed, 'status': 'passed' if passed else 'no_speedup', 'correct': True, 'reward': 0.5 if passed else 0.0, 'reference_instruction_count': 100, 'candidate_instruction_count': 50}))
"#,
    );
    let config = BridgeConfig::new(
        "python3".into(),
        bridge,
        root.clone(),
        root.join("records.jsonl"),
        "fixture".into(),
    );
    let recipe = SpeedupRecipe {
        bridge: config,
        max_attempts: 1,
        actor: nano_trajectory::Actor::Tiny,
    };
    let model = FakeModel {
        seen_context: RefCell::new(String::new()),
    };
    let trajectory = drive(&recipe, &root, &model, None, "fixture", "fixture-e2e").unwrap();
    assert_eq!(trajectory.outcome, Outcome::Success);
    assert_eq!(trajectory.initial_codes, vec!["SPEEDUP_no_speedup"]);
    assert!(std::fs::read_to_string(root.join("candidate.py"))
        .unwrap()
        .contains("len(set(xs))"));
    let context = model.seen_context.borrow();
    assert!(context.contains("reference_instruction_count: 100"));
    assert!(!context.contains("edge_inputs"));
    assert!(!context.contains("999"));
    assert!(!context.contains("123"));
    assert!(!context.contains("benchmark_variants"));
    std::fs::remove_dir_all(root).ok();
}
