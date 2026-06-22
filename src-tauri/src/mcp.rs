use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{
    env,
    error::Error,
    fmt,
    path::{Path, PathBuf},
    process::Stdio,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    time::{timeout, Duration},
};

const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
const EVERYTHING_PACKAGE: &str = "@modelcontextprotocol/server-everything";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpSettings {
    #[serde(default)]
    pub everything_enabled: bool,
}

impl Default for McpSettings {
    fn default() -> Self {
        Self {
            everything_enabled: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolCallInfo {
    pub tool_name: String,
    pub arguments: String,
    pub result: String,
    pub is_error: bool,
}

#[derive(Debug, Clone)]
pub struct McpRuntimeContext {
    pub enabled: bool,
    pub status: String,
    pub tools: Vec<McpTool>,
    pub tool_call: Option<McpToolCallInfo>,
}

impl McpRuntimeContext {
    fn disabled() -> Self {
        Self {
            enabled: false,
            status: "disabled".to_owned(),
            tools: Vec::new(),
            tool_call: None,
        }
    }
}

#[derive(Debug)]
enum McpError {
    Start(String),
    Protocol(String),
    Io(String),
    Json(String),
    Timeout(String),
}

impl fmt::Display for McpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McpError::Start(message)
            | McpError::Protocol(message)
            | McpError::Io(message)
            | McpError::Json(message)
            | McpError::Timeout(message) => write!(formatter, "{message}"),
        }
    }
}

impl Error for McpError {}

impl From<std::io::Error> for McpError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

impl From<serde_json::Error> for McpError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error.to_string())
    }
}

#[derive(Debug, Deserialize)]
struct RpcError {
    code: i64,
    message: String,
    #[serde(default)]
    data: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct RpcResponse {
    #[serde(default)]
    id: Option<Value>,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<RpcError>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolsListResult {
    #[serde(default)]
    tools: Vec<McpTool>,
    #[serde(default)]
    next_cursor: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolCallResult {
    #[serde(default)]
    content: Vec<ToolContent>,
    #[serde(default)]
    structured_content: Option<Value>,
    #[serde(default)]
    is_error: bool,
}

#[derive(Debug, Deserialize)]
struct ToolContent {
    #[serde(default, rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default, rename = "mimeType")]
    mime_type: Option<String>,
    #[serde(default)]
    uri: Option<String>,
    #[serde(default)]
    resource: Option<Value>,
}

struct McpToolCallRequest {
    tool_name: String,
    arguments: Value,
}

struct McpStdioSession {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl McpStdioSession {
    async fn connect() -> Result<Self, McpError> {
        let mut command = everything_command();
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);

        let mut child = command
            .spawn()
            .map_err(|error| McpError::Start(format!("failed to start Everything MCP: {error}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Start("Everything MCP stdin is unavailable.".to_owned()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Start("Everything MCP stdout is unavailable.".to_owned()))?;

        let mut session = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        };
        session.initialize().await?;
        Ok(session)
    }

    async fn initialize(&mut self) -> Result<(), McpError> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "chatbot-ai-desktop",
                    "version": "0.1.0"
                }
            }),
        )
        .await?;

        self.notification("notifications/initialized", json!({}))
            .await
    }

    async fn list_tools(&mut self) -> Result<Vec<McpTool>, McpError> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
            let params = cursor
                .as_ref()
                .map(|value| json!({ "cursor": value }))
                .unwrap_or_else(|| json!({}));
            let result = self.request("tools/list", params).await?;
            let parsed = serde_json::from_value::<ToolsListResult>(result)?;
            tools.extend(parsed.tools);

            match parsed.next_cursor.filter(|value| !value.trim().is_empty()) {
                Some(next_cursor) => cursor = Some(next_cursor),
                None => break,
            }
        }

        Ok(tools)
    }

    async fn call_tool(&mut self, name: &str, arguments: Value) -> Result<ToolCallResult, McpError> {
        let result = self
            .request(
                "tools/call",
                json!({
                    "name": name,
                    "arguments": arguments
                }),
            )
            .await?;

        serde_json::from_value::<ToolCallResult>(result).map_err(Into::into)
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = self.next_id;
        self.next_id += 1;

        self.write_message(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        }))
        .await?;

        self.read_response(id).await
    }

    async fn notification(&mut self, method: &str, params: Value) -> Result<(), McpError> {
        self.write_message(json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }))
        .await
    }

    async fn write_message(&mut self, message: Value) -> Result<(), McpError> {
        let mut line = serde_json::to_string(&message)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_response(&mut self, id: u64) -> Result<Value, McpError> {
        let id_value = json!(id);

        loop {
            let mut line = String::new();
            let bytes_read = timeout(REQUEST_TIMEOUT, self.stdout.read_line(&mut line))
                .await
                .map_err(|_| McpError::Timeout(format!("MCP request {id} timed out.")))??;

            if bytes_read == 0 {
                return Err(McpError::Protocol(
                    "Everything MCP closed stdout before responding.".to_owned(),
                ));
            }

            let Ok(value) = serde_json::from_str::<Value>(line.trim()) else {
                continue;
            };

            if value.get("method").is_some() && value.get("id").is_some() {
                self.reply_client_feature_not_implemented(&value).await?;
                continue;
            }

            let response = serde_json::from_value::<RpcResponse>(value)?;
            if response.id.as_ref() != Some(&id_value) {
                continue;
            }

            if let Some(error) = response.error {
                let data = error
                    .data
                    .map(|value| format!(" data={}", compact_json(&value)))
                    .unwrap_or_default();
                return Err(McpError::Protocol(format!(
                    "MCP error {}: {}{}",
                    error.code, error.message, data
                )));
            }

            return response
                .result
                .ok_or_else(|| McpError::Protocol("MCP response has no result.".to_owned()));
        }
    }

    async fn reply_client_feature_not_implemented(&mut self, request: &Value) -> Result<(), McpError> {
        let Some(id) = request.get("id").cloned() else {
            return Ok(());
        };

        self.write_message(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": -32601,
                "message": "Client-side MCP feature is not implemented in this app."
            }
        }))
        .await
    }

    async fn close(self) {
        let mut child = self.child;
        let mut stdin = self.stdin;
        let _ = stdin.shutdown().await;
        drop(stdin);

        if timeout(Duration::from_secs(2), child.wait()).await.is_err() {
            let _ = child.kill().await;
        }
    }
}

pub async fn prepare_everything_context(
    user_message: &str,
    settings: &McpSettings,
) -> McpRuntimeContext {
    if !settings.everything_enabled {
        return McpRuntimeContext::disabled();
    }

    let mut context = McpRuntimeContext {
        enabled: true,
        status: "connecting".to_owned(),
        tools: Vec::new(),
        tool_call: None,
    };

    let mut session = match McpStdioSession::connect().await {
        Ok(session) => session,
        Err(error) => {
            context.status = format!("unavailable: {error}");
            return context;
        }
    };

    match session.list_tools().await {
        Ok(tools) => {
            context.status = "connected".to_owned();
            context.tools = tools;
        }
        Err(error) => {
            context.status = format!("connected, but tools/list failed: {error}");
            session.close().await;
            return context;
        }
    }

    if let Some(call_request) = build_tool_call_request(user_message, &context.tools) {
        let arguments_text = compact_json(&call_request.arguments);
        match session
            .call_tool(&call_request.tool_name, call_request.arguments)
            .await
        {
            Ok(result) => {
                context.tool_call = Some(McpToolCallInfo {
                    tool_name: call_request.tool_name,
                    arguments: arguments_text,
                    result: format_tool_result(&result),
                    is_error: result.is_error,
                });
            }
            Err(error) => {
                context.tool_call = Some(McpToolCallInfo {
                    tool_name: call_request.tool_name,
                    arguments: arguments_text,
                    result: error.to_string(),
                    is_error: true,
                });
            }
        }
    }

    session.close().await;
    context
}

pub fn build_mcp_instruction(context: &McpRuntimeContext) -> String {
    if !context.enabled {
        return String::new();
    }

    let mut sections = Vec::new();
    sections.push(format!(
        "Everything MCP context:\nconnection_status: {}",
        preview_text(&context.status, 500)
    ));

    if context.tools.is_empty() {
        sections.push(
            "Available Everything MCP tools: none returned. If the user asked for MCP tools, explain that the MCP connection did not provide a list."
                .to_owned(),
        );
    } else {
        let tools = context
            .tools
            .iter()
            .take(40)
            .map(|tool| {
                format!(
                    "- {}: {} inputSchema={}",
                    tool.name,
                    preview_text(tool.description.as_deref().unwrap_or("No description."), 500),
                    compact_json(&tool.input_schema)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        sections.push(format!(
            "Available Everything MCP tools:\n{tools}\n\n\
             Use this list when the user asks what Everything MCP can do. \
             Do not claim a tool was invoked unless an mcp_tool_call_result block is present."
        ));
    }

    if let Some(call) = &context.tool_call {
        sections.push(format!(
            "mcp_tool_call_result:\n\
             tool: {}\n\
             arguments: {}\n\
             is_error: {}\n\
             result:\n{}",
            call.tool_name,
            call.arguments,
            call.is_error,
            preview_text(&call.result, 6000)
        ));
    } else {
        sections.push(
            "mcp_tool_call_result: none. If the user wants a direct MCP call, ask them to use /mcp <toolName> {jsonArgs} or name a listed tool and arguments."
                .to_owned(),
        );
    }

    sections.join("\n\n")
}

fn build_tool_call_request(user_message: &str, tools: &[McpTool]) -> Option<McpToolCallRequest> {
    if tools.is_empty() {
        return None;
    }

    let message = user_message.trim();
    let lower = message.to_lowercase();
    let wants_mcp = lower.starts_with("/mcp")
        || lower.contains("mcp")
        || lower.contains("everything")
        || lower.contains("model context protocol");

    if !wants_mcp {
        return None;
    }

    let has_call_intent = lower.starts_with("/mcp")
        || [
            "call",
            "invoke",
            "execute",
            "run",
            "\u{0432}\u{044b}\u{0437}\u{043e}\u{0432}",
            "\u{0437}\u{0430}\u{043f}\u{0443}\u{0441}\u{0442}",
            "\u{0432}\u{044b}\u{043f}\u{043e}\u{043b}\u{043d}",
        ]
        .iter()
        .any(|word| lower.contains(word));

    if !has_call_intent {
        return None;
    }

    let tool = find_requested_tool(message, &lower, tools)?;
    let arguments = parse_json_arguments(message)
        .unwrap_or_else(|| infer_arguments_from_message(message, &lower, tool));

    Some(McpToolCallRequest {
        tool_name: tool.name.clone(),
        arguments,
    })
}

fn find_requested_tool<'a>(
    message: &str,
    lower: &str,
    tools: &'a [McpTool],
) -> Option<&'a McpTool> {
    if lower.starts_with("/mcp") {
        if let Some(name) = message
            .split_whitespace()
            .nth(1)
            .map(|value| value.trim_matches(|character: char| !is_tool_name_character(character)))
            .filter(|value| !value.is_empty())
        {
            if let Some(tool) = tools
                .iter()
                .find(|tool| tool.name.eq_ignore_ascii_case(name))
            {
                return Some(tool);
            }
        }
    }

    tools
        .iter()
        .find(|tool| contains_word(lower, &tool.name.to_lowercase()))
}

fn parse_json_arguments(message: &str) -> Option<Value> {
    let start = message.find('{')?;
    let end = message.rfind('}')?;
    if end <= start {
        return None;
    }

    let value = serde_json::from_str::<Value>(&message[start..=end]).ok()?;
    value.is_object().then_some(value)
}

fn infer_arguments_from_message(message: &str, lower: &str, tool: &McpTool) -> Value {
    let Some(properties) = tool
        .input_schema
        .get("properties")
        .and_then(Value::as_object)
    else {
        return json!({});
    };

    let property_names = preferred_property_names(&tool.input_schema, properties);
    if property_names.is_empty() {
        return json!({});
    }

    let mut arguments = Map::new();
    let numbers = extract_numbers(message);

    if property_names
        .iter()
        .all(|name| property_type(properties, name).is_some_and(|kind| kind == "number" || kind == "integer"))
        && numbers.len() >= property_names.len()
    {
        for (name, value) in property_names.iter().zip(numbers.iter()) {
            arguments.insert(name.clone(), number_to_json(*value));
        }

        return Value::Object(arguments);
    }

    if property_names.len() == 1 {
        let name = &property_names[0];
        let value = match property_type(properties, name) {
            Some("number") | Some("integer") => numbers
                .first()
                .map(|value| number_to_json(*value))
                .unwrap_or(Value::Null),
            Some("boolean") => Value::Bool(lower.contains("true")),
            _ => Value::String(extract_text_argument(message, lower, &tool.name)),
        };

        if !value.is_null() {
            arguments.insert(name.clone(), value);
        }
    }

    Value::Object(arguments)
}

fn preferred_property_names(schema: &Value, properties: &Map<String, Value>) -> Vec<String> {
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .filter(|name| properties.contains_key(*name))
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if required.is_empty() {
        properties.keys().cloned().collect()
    } else {
        required
    }
}

fn property_type<'a>(properties: &'a Map<String, Value>, name: &str) -> Option<&'a str> {
    properties
        .get(name)
        .and_then(|schema| schema.get("type"))
        .and_then(Value::as_str)
}

fn extract_text_argument(message: &str, lower: &str, tool_name: &str) -> String {
    let text = lower
        .find(&tool_name.to_lowercase())
        .and_then(|index| message.get(index + tool_name.len()..))
        .unwrap_or(message)
        .trim();

    let cleaned = text
        .trim_start_matches(|character: char| {
            character.is_whitespace() || matches!(character, ':' | '=' | '-' | ',' | '.')
        })
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim();

    if cleaned.is_empty() {
        message.to_owned()
    } else {
        cleaned.to_owned()
    }
}

fn extract_numbers(message: &str) -> Vec<f64> {
    message
        .split(|character: char| {
            !(character.is_ascii_digit() || character == '.' || character == '-')
        })
        .filter_map(|part| part.parse::<f64>().ok())
        .collect()
}

fn number_to_json(value: f64) -> Value {
    if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
        return Value::Number((value as i64).into());
    }

    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn format_tool_result(result: &ToolCallResult) -> String {
    let mut parts = result
        .content
        .iter()
        .map(|content| match content.kind.as_str() {
            "text" => content.text.clone().unwrap_or_default(),
            "resource_link" => format!(
                "[resource_link: {}]",
                content.uri.as_deref().unwrap_or("no uri")
            ),
            "resource" => format!(
                "[resource: {}]",
                content
                    .resource
                    .as_ref()
                    .map(compact_json)
                    .unwrap_or_else(|| "no resource payload".to_owned())
            ),
            other => format!(
                "[{other} content{}]",
                content
                    .mime_type
                    .as_deref()
                    .map(|mime_type| format!(" mimeType={mime_type}"))
                    .unwrap_or_default()
            ),
        })
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>();

    if let Some(structured) = &result.structured_content {
        parts.push(format!("structuredContent={}", compact_json(structured)));
    }

    if parts.is_empty() {
        "Tool returned no content.".to_owned()
    } else {
        parts.join("\n")
    }
}

fn contains_word(haystack: &str, needle: &str) -> bool {
    haystack
        .split(|character: char| !is_tool_name_character(character))
        .any(|part| part == needle)
}

fn is_tool_name_character(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '_' || character == '-'
}

fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_owned())
}

fn preview_text(value: &str, max_chars: usize) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        return compact;
    }

    let mut preview = compact.chars().take(max_chars).collect::<String>();
    preview.push_str("...");
    preview
}

fn everything_command() -> Command {
    if let Some(local_bin) = find_local_everything_bin() {
        if cfg!(windows) {
            let mut command = Command::new("cmd.exe");
            command.arg("/C").arg(local_bin);
            return command;
        }

        return Command::new(local_bin);
    }

    if cfg!(windows) {
        let mut command = Command::new("cmd.exe");
        command.args(["/C", "npx", "-y", EVERYTHING_PACKAGE]);
        command
    } else {
        let mut command = Command::new("npx");
        command.args(["-y", EVERYTHING_PACKAGE]);
        command
    }
}

fn find_local_everything_bin() -> Option<PathBuf> {
    let bin_name = if cfg!(windows) {
        "mcp-server-everything.CMD"
    } else {
        "mcp-server-everything"
    };

    let mut roots = Vec::new();
    if let Ok(current_dir) = env::current_dir() {
        roots.extend(current_dir.ancestors().map(Path::to_path_buf));
    }

    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        let manifest_path = Path::new(manifest_dir);
        roots.extend(manifest_path.ancestors().map(Path::to_path_buf));
    }

    roots
        .into_iter()
        .map(|root| root.join("node_modules").join(".bin").join(bin_name))
        .find(|candidate| candidate.is_file())
}
