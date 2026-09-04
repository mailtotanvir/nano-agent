//! Language-agnostic search/replace patch engine for nano-agent recipes.
//!
//! Tiny models produce malformed unified-diff headers constantly. We instead use
//! exact-match SEARCH/REPLACE blocks (aider-style), which are trivial to validate
//! and apply, with a whitespace-normalized fuzzy fallback for near misses.
//!
//! Block format:
//! ```text
//! <<<<<<< SEARCH
//! <old text>
//! =======
//! <new text>
//! >>>>>>> REPLACE
//! ```

use serde::{Deserialize, Serialize};

const MARK_SEARCH: &str = "<<<<<<< SEARCH";
const MARK_DIVIDER: &str = "=======";
const MARK_REPLACE: &str = ">>>>>>> REPLACE";

/// A single search/replace edit targeting one file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EditBlock {
    pub file: String,
    pub search: String,
    pub replace: String,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PatchError {
    #[error("search text not found in {file}")]
    NotFound { file: String },
    #[error("search text is ambiguous in {file}: {count} occurrences")]
    Ambiguous { file: String, count: usize },
    #[error("search and replace are identical in {file}")]
    NoOp { file: String },
    #[error("patch exceeds size cap: {changed} changed lines > {cap}")]
    TooLarge { changed: usize, cap: usize },
    #[error("malformed edit block: {0}")]
    Malformed(String),
}

/// Limits applied when validating/applying a patch.
#[derive(Debug, Clone, Copy)]
pub struct PatchLimits {
    /// Max changed (search+replace) lines per patch. Default 60.
    pub max_changed_lines: usize,
    /// Allow the whitespace-normalized fuzzy fallback when exact match fails.
    pub allow_fuzzy: bool,
}

impl Default for PatchLimits {
    fn default() -> Self {
        Self {
            max_changed_lines: 60,
            allow_fuzzy: true,
        }
    }
}

/// Parse zero or more SEARCH/REPLACE blocks out of raw model text.
///
/// `default_file` is used when a block does not carry its own `file:` hint on the
/// line immediately preceding the SEARCH marker. Returns blocks in order.
pub fn parse_blocks(text: &str, default_file: Option<&str>) -> Result<Vec<EditBlock>, PatchError> {
    let mut blocks = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim_end() == MARK_SEARCH {
            // Look back one line for an optional "file: <path>" hint.
            let file = infer_file(&lines, i, default_file)?;
            let mut j = i + 1;
            let mut search = Vec::new();
            while j < lines.len() && lines[j].trim_end() != MARK_DIVIDER {
                if lines[j].trim_end() == MARK_SEARCH || lines[j].trim_end() == MARK_REPLACE {
                    return Err(PatchError::Malformed(
                        "nested/unterminated SEARCH before divider".into(),
                    ));
                }
                search.push(lines[j]);
                j += 1;
            }
            if j >= lines.len() {
                return Err(PatchError::Malformed("missing ======= divider".into()));
            }
            j += 1; // skip divider
            let mut replace = Vec::new();
            while j < lines.len() && lines[j].trim_end() != MARK_REPLACE {
                if lines[j].trim_end() == MARK_SEARCH || lines[j].trim_end() == MARK_DIVIDER {
                    return Err(PatchError::Malformed(
                        "unexpected marker inside REPLACE body".into(),
                    ));
                }
                replace.push(lines[j]);
                j += 1;
            }
            if j >= lines.len() {
                return Err(PatchError::Malformed("missing >>>>>>> REPLACE".into()));
            }
            blocks.push(EditBlock {
                file,
                search: search.join("\n"),
                replace: replace.join("\n"),
            });
            i = j + 1;
        } else {
            i += 1;
        }
    }
    Ok(blocks)
}

fn infer_file(
    lines: &[&str],
    search_idx: usize,
    default_file: Option<&str>,
) -> Result<String, PatchError> {
    if search_idx > 0 {
        let prev = lines[search_idx - 1].trim();
        if let Some(rest) = prev.strip_prefix("file:") {
            let f = rest.trim();
            if !f.is_empty() {
                return Ok(f.to_string());
            }
        }
    }
    default_file
        .map(|s| s.to_string())
        .ok_or_else(|| PatchError::Malformed("no file hint and no default_file".into()))
}

/// Result of applying one block to a file's content.
#[derive(Debug, Clone)]
pub struct ApplyOutcome {
    pub new_content: String,
    pub fuzzy: bool,
}

/// Apply a single edit block to `content`, honoring limits. Returns the new content.
pub fn apply_block(
    content: &str,
    block: &EditBlock,
    limits: &PatchLimits,
) -> Result<ApplyOutcome, PatchError> {
    if block.search == block.replace {
        return Err(PatchError::NoOp {
            file: block.file.clone(),
        });
    }
    let changed = block.search.lines().count() + block.replace.lines().count();
    if changed > limits.max_changed_lines {
        return Err(PatchError::TooLarge {
            changed,
            cap: limits.max_changed_lines,
        });
    }

    // Exact match path.
    let matches: Vec<usize> = find_all(content, &block.search);
    match matches.len() {
        1 => {
            let start = matches[0];
            let mut out = String::with_capacity(content.len() + block.replace.len());
            out.push_str(&content[..start]);
            out.push_str(&block.replace);
            out.push_str(&content[start + block.search.len()..]);
            return Ok(ApplyOutcome {
                new_content: out,
                fuzzy: false,
            });
        }
        n if n > 1 => {
            return Err(PatchError::Ambiguous {
                file: block.file.clone(),
                count: n,
            })
        }
        _ => {}
    }

    // Fuzzy fallback: match on whitespace-normalized lines.
    if limits.allow_fuzzy {
        if let Some(out) = fuzzy_apply(content, &block.search, &block.replace) {
            return Ok(ApplyOutcome {
                new_content: out,
                fuzzy: true,
            });
        }
    }
    Err(PatchError::NotFound {
        file: block.file.clone(),
    })
}

fn find_all(haystack: &str, needle: &str) -> Vec<usize> {
    if needle.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        out.push(start + pos);
        start += pos + 1;
    }
    out
}

/// Whitespace-normalized line-block matcher. Finds a contiguous run of lines in
/// `content` whose trimmed forms equal the trimmed search lines, then swaps in
/// `replace` while preserving the file's surrounding text.
fn fuzzy_apply(content: &str, search: &str, replace: &str) -> Option<String> {
    let content_lines: Vec<&str> = content.split_inclusive('\n').collect();
    let search_norm: Vec<String> = search.lines().map(normalize).collect();
    if search_norm.is_empty() {
        return None;
    }
    let n = search_norm.len();
    let mut hit: Option<usize> = None;
    let plain: Vec<String> = content_lines.iter().map(|l| normalize(l)).collect();
    for i in 0..=plain.len().saturating_sub(n) {
        if plain[i..i + n] == search_norm[..] {
            if hit.is_some() {
                return None; // ambiguous under fuzzy match too; refuse
            }
            hit = Some(i);
        }
    }
    let i = hit?;
    let mut out = String::new();
    out.push_str(&content_lines[..i].concat());
    // Preserve trailing newline if the matched region had one.
    let matched_has_nl = content_lines[i + n - 1].ends_with('\n');
    out.push_str(replace);
    if matched_has_nl && !replace.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&content_lines[i + n..].concat());
    Some(out)
}

fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_apply() {
        let content = "fn main() {\n    let x: i32 = \"5\";\n}\n";
        let block = EditBlock {
            file: "src/main.rs".into(),
            search: "    let x: i32 = \"5\";".into(),
            replace: "    let x: i32 = 5;".into(),
        };
        let out = apply_block(content, &block, &PatchLimits::default()).unwrap();
        assert!(!out.fuzzy);
        assert_eq!(out.new_content, "fn main() {\n    let x: i32 = 5;\n}\n");
    }

    #[test]
    fn not_found() {
        let content = "fn main() {}\n";
        let block = EditBlock {
            file: "src/main.rs".into(),
            search: "nonexistent".into(),
            replace: "whatever".into(),
        };
        let err = apply_block(
            content,
            &block,
            &PatchLimits {
                allow_fuzzy: false,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            err,
            PatchError::NotFound {
                file: "src/main.rs".into()
            }
        );
    }

    #[test]
    fn ambiguous() {
        let content = "let a = 1;\nlet a = 1;\n";
        let block = EditBlock {
            file: "f.rs".into(),
            search: "let a = 1;".into(),
            replace: "let a = 2;".into(),
        };
        let err = apply_block(
            content,
            &block,
            &PatchLimits {
                allow_fuzzy: false,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            err,
            PatchError::Ambiguous {
                file: "f.rs".into(),
                count: 2
            }
        );
    }

    #[test]
    fn noop_rejected() {
        let content = "x\n";
        let block = EditBlock {
            file: "f.rs".into(),
            search: "x".into(),
            replace: "x".into(),
        };
        let err = apply_block(content, &block, &PatchLimits::default()).unwrap_err();
        assert_eq!(
            err,
            PatchError::NoOp {
                file: "f.rs".into()
            }
        );
    }

    #[test]
    fn size_cap() {
        let content = "a\n";
        let big: String = (0..100).map(|_| "x\n").collect();
        let block = EditBlock {
            file: "f.rs".into(),
            search: "a".into(),
            replace: big,
        };
        let err = apply_block(
            content,
            &block,
            &PatchLimits {
                max_changed_lines: 60,
                ..Default::default()
            },
        )
        .unwrap_err();
        matches!(err, PatchError::TooLarge { .. });
    }

    #[test]
    fn fuzzy_whitespace() {
        // Search has different INTERNAL spacing than the file, so exact-match
        // fails and only the whitespace-normalized fuzzy path can apply it.
        let content = "fn main() {\n    let x   =   1;\n}\n";
        let block = EditBlock {
            file: "f.rs".into(),
            search: "let x = 1;".into(),
            replace: "let x = 2;".into(),
        };
        let out = apply_block(content, &block, &PatchLimits::default()).unwrap();
        assert!(out.fuzzy);
        assert!(out.new_content.contains("let x = 2;"));
    }

    #[test]
    fn parse_single_block() {
        let text = "file: src/main.rs\n<<<<<<< SEARCH\nlet x: i32 = \"5\";\n=======\nlet x: i32 = 5;\n>>>>>>> REPLACE\n";
        let blocks = parse_blocks(text, None).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].file, "src/main.rs");
        assert_eq!(blocks[0].search, "let x: i32 = \"5\";");
        assert_eq!(blocks[0].replace, "let x: i32 = 5;");
    }

    #[test]
    fn parse_uses_default_file() {
        let text = "<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n";
        let blocks = parse_blocks(text, Some("src/lib.rs")).unwrap();
        assert_eq!(blocks[0].file, "src/lib.rs");
    }

    #[test]
    fn parse_malformed_missing_divider() {
        let text = "<<<<<<< SEARCH\na\n>>>>>>> REPLACE\n";
        // ">>>>>>> REPLACE" appears while scanning for divider -> malformed
        assert!(parse_blocks(text, Some("f.rs")).is_err());
    }
}
