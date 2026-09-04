//! Model client abstraction: one trait, multiple backends.
//!
//! - `LlamaCppClient` talks to a llama.cpp server's OpenAI-compatible
//!   `/v1/chat/completions` endpoint (the tiny CPU-served model on the OCI box),
//!   optionally passing a GBNF grammar so sub-1B models are forced to emit valid
//!   JSON matching the output contract.
//! - `GeminiClient` talks to Google's Generative Language API
//!   (`generateContent`) for the frontier teacher / fallback (Gemini 3.6 Flash).
//!
//! Swapping the tiny model backend never touches the controller (spec section 12).

use nano_trajectory::Proposal;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// One chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String, // "system" | "user" | "assistant"
    pub content: String,
}

impl Message {
    pub fn system(c: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: c.into(),
        }
    }
    pub fn user(c: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: c.into(),
        }
    }
    pub fn assistant(c: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: c.into(),
        }
    }
}

/// What a model returns for one call.
pub struct ModelResponse {
    pub proposal: Proposal,
    pub raw: String,
    pub latency_ms: u64,
    pub tokens_out: Option<u32>,
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("model returned non-JSON or contract-violating output: {0}")]
    BadOutput(String),
    #[error("api error {status}: {body}")]
    Api { status: u16, body: String },
}

/// The core abstraction the controller depends on.
pub trait ModelClient {
    /// Human-readable identifier recorded in trajectories.
    fn name(&self) -> &str;
    /// Produce a proposal from a chat context.
    fn propose(&self, messages: &[Message]) -> Result<ModelResponse, ModelError>;
}

/// Extract the first balanced JSON object from arbitrary model text and parse it
/// as a `Proposal`. Tolerant of leading prose / code fences.
pub fn parse_proposal(text: &str) -> Result<Proposal, ModelError> {
    let json = extract_json_object(text).ok_or_else(|| {
        ModelError::BadOutput(format!("no JSON object found in: {}", truncate(text, 200)))
    })?;
    let p: Proposal = serde_json::from_str(&json)
        .map_err(|e| ModelError::BadOutput(format!("{e}; json was: {}", truncate(&json, 200))))?;
    match p.action.as_str() {
        "patch" | "no_fix" | "escalate" => Ok(p),
        other => Err(ModelError::BadOutput(format!("invalid action: {other}"))),
    }
}

fn extract_json_object(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let start = text.find('{')?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for i in start..bytes.len() {
        let c = bytes[i] as char;
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
        } else {
            match c {
                '"' => in_str = true,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(text[start..=i].to_string());
                    }
                }
                _ => {}
            }
        }
    }
    None
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}...", &s[..n])
    }
}

// ---------------------------------------------------------------------------
// llama.cpp (OpenAI-compatible) backend
// ---------------------------------------------------------------------------

/// Client for a llama.cpp server (`--host`/`--port`, OpenAI-compatible route).
pub struct LlamaCppClient {
    base_url: String,
    model_name: String,
    http: reqwest::blocking::Client,
    /// Optional GBNF grammar to constrain output to the JSON contract.
    grammar: Option<String>,
    max_tokens: u32,
    temperature: f32,
}

impl LlamaCppClient {
    pub fn new(base_url: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model_name: model_name.into(),
            http: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("http client"),
            grammar: None,
            max_tokens: 512,
            temperature: 0.2,
        }
    }
    pub fn with_grammar(mut self, gbnf: impl Into<String>) -> Self {
        self.grammar = Some(gbnf.into());
        self
    }
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
}

#[derive(Serialize)]
struct OpenAiRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    max_tokens: u32,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    grammar: Option<&'a str>,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}
#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMsg,
}
#[derive(Deserialize)]
struct OpenAiMsg {
    content: String,
}
#[derive(Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    completion_tokens: u32,
}

impl ModelClient for LlamaCppClient {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn propose(&self, messages: &[Message]) -> Result<ModelResponse, ModelError> {
        let url = format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        );
        let body = OpenAiRequest {
            model: &self.model_name,
            messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            grammar: self.grammar.as_deref(),
        };
        let start = Instant::now();
        let resp = self.http.post(&url).json(&body).send()?;
        let status = resp.status();
        let text = resp.text()?;
        if !status.is_success() {
            return Err(ModelError::Api {
                status: status.as_u16(),
                body: truncate(&text, 500),
            });
        }
        let parsed: OpenAiResponse = serde_json::from_str(&text)
            .map_err(|e| ModelError::BadOutput(format!("bad server json: {e}")))?;
        let content = parsed
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| ModelError::BadOutput("no choices".into()))?;
        let proposal = parse_proposal(&content)?;
        Ok(ModelResponse {
            proposal,
            raw: content,
            latency_ms: start.elapsed().as_millis() as u64,
            tokens_out: parsed.usage.map(|u| u.completion_tokens),
        })
    }
}

// ---------------------------------------------------------------------------
// Azure OpenAI backend (frontier comparison: gpt-5.6, grok, deepseek via Azure)
// ---------------------------------------------------------------------------

/// Talks to an Azure AI / OpenAI-compatible `/chat/completions` endpoint using
/// an `api-key` header. Base URL should already include the `openai/v1` suffix
/// (e.g. `https://<res>.services.ai.azure.com/openai/v1`). Uses
/// `max_completion_tokens` (required by the gpt-5 series) rather than
/// `max_tokens`, and omits temperature (some frontier models reject non-default).
pub struct AzureClient {
    base_url: String,
    model_name: String,
    api_key: String,
    http: reqwest::blocking::Client,
    max_tokens: u32,
}

impl AzureClient {
    pub fn new(
        base_url: impl Into<String>,
        model_name: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            model_name: model_name.into(),
            api_key: api_key.into(),
            http: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(180))
                .build()
                .expect("http client"),
            max_tokens: 1024,
        }
    }

    /// Build from env: AZURE_API_BASE + AZURE_API_KEY.
    pub fn from_env(model_name: impl Into<String>) -> Result<Self, ModelError> {
        let base = std::env::var("AZURE_API_BASE")
            .map_err(|_| ModelError::BadOutput("AZURE_API_BASE not set".into()))?;
        let key = std::env::var("AZURE_API_KEY")
            .map_err(|_| ModelError::BadOutput("AZURE_API_KEY not set".into()))?;
        Ok(Self::new(base, model_name, key))
    }
}

#[derive(Serialize)]
struct AzureRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    max_completion_tokens: u32,
}

impl ModelClient for AzureClient {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn propose(&self, messages: &[Message]) -> Result<ModelResponse, ModelError> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body = AzureRequest {
            model: &self.model_name,
            messages,
            max_completion_tokens: self.max_tokens,
        };
        let start = Instant::now();
        let resp = self
            .http
            .post(&url)
            .header("api-key", &self.api_key)
            .json(&body)
            .send()?;
        let status = resp.status();
        let text = resp.text()?;
        if !status.is_success() {
            return Err(ModelError::Api {
                status: status.as_u16(),
                body: truncate(&text, 500),
            });
        }
        let parsed: OpenAiResponse = serde_json::from_str(&text)
            .map_err(|e| ModelError::BadOutput(format!("bad azure json: {e}")))?;
        let content = parsed
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| ModelError::BadOutput("no choices".into()))?;
        let proposal = parse_proposal(&content)?;
        Ok(ModelResponse {
            proposal,
            raw: content,
            latency_ms: start.elapsed().as_millis() as u64,
            tokens_out: parsed.usage.map(|u| u.completion_tokens),
        })
    }
}

// ---------------------------------------------------------------------------
// Gemini backend (frontier teacher / fallback)
// ---------------------------------------------------------------------------

/// Client for Google Generative Language API `generateContent`.
/// Reads the API key from the `GEMINI_API_KEY` env var by default.
/// Auth + endpoint mode for Gemini.
enum GeminiAuth {
    /// AI Studio API key (generativelanguage endpoint). Often free-tier limited.
    ApiKey(String),
    /// Vertex AI via ADC bearer token, billed to a GCP project. `location` is
    /// usually "global"; `project` is the GCP project id.
    VertexAdc {
        project: String,
        location: String,
        token: String,
    },
}

pub struct GeminiClient {
    model: String,
    auth: GeminiAuth,
    http: reqwest::blocking::Client,
    temperature: f32,
}

impl GeminiClient {
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .map_err(|_| anyhow::anyhow!("GEMINI_API_KEY / GOOGLE_API_KEY not set"))?;
        Ok(Self {
            model: model.into(),
            auth: GeminiAuth::ApiKey(api_key),
            http: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()?,
            temperature: 0.2,
        })
    }

    /// Vertex AI backend using Application Default Credentials. The bearer token
    /// is fetched via `gcloud auth application-default print-access-token`, so the
    /// caller must have ADC configured. Billed to `project`'s GCP account (real
    /// quota, not AI Studio free tier).
    pub fn from_vertex_adc(
        model: impl Into<String>,
        project: impl Into<String>,
        location: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let token = fetch_adc_token()?;
        Ok(Self {
            model: model.into(),
            auth: GeminiAuth::VertexAdc {
                project: project.into(),
                location: location.into(),
                token,
            },
            http: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()?,
            temperature: 0.2,
        })
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    fn endpoint(&self) -> String {
        match &self.auth {
            GeminiAuth::ApiKey(key) => format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                self.model, key
            ),
            GeminiAuth::VertexAdc {
                project, location, ..
            } => {
                let host = if location == "global" {
                    "aiplatform.googleapis.com".to_string()
                } else {
                    format!("{location}-aiplatform.googleapis.com")
                };
                format!(
                    "https://{host}/v1/projects/{project}/locations/{location}/publishers/google/models/{}:generateContent",
                    self.model
                )
            }
        }
    }
}

/// Fetch an ADC access token: prefer the `GOOGLE_VERTEX_TOKEN` env var (so hosts
/// without gcloud can still use Vertex), else shell out to gcloud.
fn fetch_adc_token() -> anyhow::Result<String> {
    if let Ok(tok) = std::env::var("GOOGLE_VERTEX_TOKEN") {
        let tok = tok.trim().to_string();
        if !tok.is_empty() {
            return Ok(tok);
        }
    }
    let out = std::process::Command::new("gcloud")
        .args(["auth", "application-default", "print-access-token"])
        .output()
        .map_err(|e| anyhow::anyhow!("no GOOGLE_VERTEX_TOKEN and failed to run gcloud: {e}"))?;
    if !out.status.success() {
        anyhow::bail!(
            "gcloud ADC token failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(rename = "generationConfig")]
    generation_config: GeminiGenConfig,
}
#[derive(Serialize, Deserialize)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}
#[derive(Serialize, Deserialize)]
struct GeminiPart {
    text: String,
}
#[derive(Serialize)]
struct GeminiGenConfig {
    temperature: f32,
    #[serde(rename = "responseMimeType")]
    response_mime_type: String,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}
#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

impl ModelClient for GeminiClient {
    fn name(&self) -> &str {
        &self.model
    }

    fn propose(&self, messages: &[Message]) -> Result<ModelResponse, ModelError> {
        // Split system messages out; Gemini wants systemInstruction separate.
        let mut system = String::new();
        let mut contents = Vec::new();
        for m in messages {
            if m.role == "system" {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(&m.content);
            } else {
                let role = if m.role == "assistant" {
                    "model"
                } else {
                    "user"
                };
                contents.push(GeminiContent {
                    role: Some(role.into()),
                    parts: vec![GeminiPart {
                        text: m.content.clone(),
                    }],
                });
            }
        }
        let req = GeminiRequest {
            contents,
            system_instruction: if system.is_empty() {
                None
            } else {
                Some(GeminiContent {
                    role: None,
                    parts: vec![GeminiPart { text: system }],
                })
            },
            generation_config: GeminiGenConfig {
                temperature: self.temperature,
                response_mime_type: "application/json".into(),
            },
        };
        let url = self.endpoint();
        let start = Instant::now();
        let mut builder = self.http.post(&url).json(&req);
        if let GeminiAuth::VertexAdc { token, .. } = &self.auth {
            builder = builder.bearer_auth(token);
        }
        let resp = builder.send()?;
        let status = resp.status();
        let text = resp.text()?;
        if !status.is_success() {
            return Err(ModelError::Api {
                status: status.as_u16(),
                body: truncate(&text, 500),
            });
        }
        let parsed: GeminiResponse = serde_json::from_str(&text).map_err(|e| {
            ModelError::BadOutput(format!(
                "bad gemini json: {e}; body: {}",
                truncate(&text, 300)
            ))
        })?;
        let content = parsed
            .candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .map(|p| p.text.clone())
            .ok_or_else(|| ModelError::BadOutput("no gemini candidate text".into()))?;
        let proposal = parse_proposal(&content)?;
        Ok(ModelResponse {
            proposal,
            raw: content,
            latency_ms: start.elapsed().as_millis() as u64,
            tokens_out: None,
        })
    }
}

/// GBNF grammar constraining output to the JSON proposal contract.
/// Used with llama.cpp to guarantee valid JSON from sub-1B models.
pub const PROPOSAL_GBNF: &str = r##"
root   ::= "{" ws "\"action\"" ws ":" ws action ws "," ws "\"patch\"" ws ":" ws patch ws "," ws "\"reason\"" ws ":" ws string ws "," ws "\"confidence\"" ws ":" ws number ws "}"
action ::= "\"patch\"" | "\"no_fix\"" | "\"escalate\""
patch  ::= string | "null"
string ::= "\"" ( [^"\\] | "\\" . )* "\""
number ::= "0" | "0." [0-9]+ | "1" | "1.0"
ws     ::= [ \t\n]*
"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_plain_json() {
        let p = parse_proposal(r#"{"action":"patch","patch":"x","reason":"y","confidence":0.5}"#)
            .unwrap();
        assert_eq!(p.action, "patch");
        assert_eq!(p.patch.as_deref(), Some("x"));
    }

    #[test]
    fn extract_json_with_prose_and_fence() {
        let text = "Sure, here is the fix:\n```json\n{\"action\": \"no_fix\", \"patch\": null, \"reason\": \"cannot\", \"confidence\": 0.1}\n```\n";
        let p = parse_proposal(text).unwrap();
        assert_eq!(p.action, "no_fix");
        assert!(p.patch.is_none());
    }

    #[test]
    fn nested_braces_in_string() {
        let text =
            r#"{"action":"patch","patch":"fn f() { let x = 1; }","reason":"r","confidence":0.9}"#;
        let p = parse_proposal(text).unwrap();
        assert!(p.patch.unwrap().contains("{ let x = 1; }"));
    }

    #[test]
    fn invalid_action_rejected() {
        let err = parse_proposal(r#"{"action":"delete","patch":null,"reason":"","confidence":0}"#)
            .unwrap_err();
        matches!(err, ModelError::BadOutput(_));
    }

    #[test]
    fn no_json_rejected() {
        assert!(parse_proposal("I cannot help with that.").is_err());
    }
}
