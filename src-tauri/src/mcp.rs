use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE, WWW_AUTHENTICATE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{collections::BTreeMap, error::Error, fmt, process::Stdio};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    time::{timeout, Duration},
};

const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
const EVERYTHING_PACKAGE: &str = "@modelcontextprotocol/server-everything";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpSettings {
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
    // Backward compatibility with settings saved before multi-server support.
    #[serde(default)]
    pub everything_enabled: bool,
}

impl McpSettings {
    pub fn enabled_servers(&self) -> Vec<McpServerConfig> {
        if !self.servers.is_empty() {
            return self
                .servers
                .iter()
                .filter(|server| server.enabled)
                .cloned()
                .collect();
        }

        self.everything_enabled
            .then(default_everything_server)
            .into_iter()
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerConfig {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub transport: McpTransport,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum McpTransport {
    #[default]
    Stdio,
    StreamableHttp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpTool {
    #[serde(default)]
    pub server_id: String,
    #[serde(default)]
    pub server_name: String,
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
    pub server_id: String,
    pub server_name: String,
    pub tool_name: String,
    pub arguments: String,
    pub result: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct McpConnectionTestResult {
    pub server_id: String,
    pub server_name: String,
    pub connected: bool,
    pub status: String,
    pub tools: Vec<McpTool>,
}

#[derive(Debug, Clone)]
pub struct McpRuntimeContext {
    pub enabled: bool,
    pub status: String,
    pub tools: Vec<McpTool>,
    pub tool_call: Option<McpToolCallInfo>,
    pub tool_calls: Vec<McpToolCallInfo>,
}

impl McpRuntimeContext {
    fn disabled() -> Self {
        Self {
            enabled: false,
            status: "disabled".to_owned(),
            tools: Vec::new(),
            tool_call: None,
            tool_calls: Vec::new(),
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
    Http(String),
}

impl fmt::Display for McpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McpError::Start(message)
            | McpError::Protocol(message)
            | McpError::Io(message)
            | McpError::Json(message)
            | McpError::Timeout(message)
            | McpError::Http(message) => write!(formatter, "{message}"),
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
    server_id: String,
    tool_name: String,
    arguments: Value,
}

struct McpStdioSession {
    server_name: String,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl McpStdioSession {
    async fn connect(config: &McpServerConfig) -> Result<Self, McpError> {
        let mut command = configured_command(config)?;
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);

        if let Some(cwd) = config
            .cwd
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            command.current_dir(cwd);
        }
        command.envs(&config.env);

        let mut child = command.spawn().map_err(|error| {
            McpError::Start(format!(
                "failed to start MCP server '{}': {error}",
                config.name
            ))
        })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            McpError::Start(format!("MCP server '{}' stdin is unavailable", config.name))
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            McpError::Start(format!(
                "MCP server '{}' stdout is unavailable",
                config.name
            ))
        })?;

        let mut session = Self {
            server_name: display_server_name(config),
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
                    "version": "0.2.0"
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

    async fn call_tool(
        &mut self,
        name: &str,
        arguments: Value,
    ) -> Result<ToolCallResult, McpError> {
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
                .map_err(|_| {
                    McpError::Timeout(format!(
                        "MCP server '{}' request {id} timed out",
                        self.server_name
                    ))
                })??;

            if bytes_read == 0 {
                return Err(McpError::Protocol(format!(
                    "MCP server '{}' closed stdout before responding",
                    self.server_name
                )));
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
                .ok_or_else(|| McpError::Protocol("MCP response has no result".to_owned()));
        }
    }

    async fn reply_client_feature_not_implemented(
        &mut self,
        request: &Value,
    ) -> Result<(), McpError> {
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

struct McpHttpSession {
    server_name: String,
    client: reqwest::Client,
    url: String,
    headers: HeaderMap,
    session_id: Option<String>,
    next_id: u64,
}

impl McpHttpSession {
    async fn connect(config: &McpServerConfig) -> Result<Self, McpError> {
        let url = config
            .url
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                McpError::Start(format!(
                    "remote MCP server '{}' has an empty URL",
                    display_server_name(config)
                ))
            })?;
        reqwest::Url::parse(url)
            .map_err(|error| McpError::Start(format!("invalid MCP URL: {error}")))?;

        let mut headers = HeaderMap::new();
        for (name, value) in &config.headers {
            let name = HeaderName::from_bytes(name.trim().as_bytes()).map_err(|error| {
                McpError::Start(format!("invalid HTTP header name '{name}': {error}"))
            })?;
            let value = HeaderValue::from_str(value).map_err(|error| {
                McpError::Start(format!("invalid value for HTTP header '{name}': {error}"))
            })?;
            headers.insert(name, value);
        }

        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|error| McpError::Http(error.to_string()))?;
        let mut session = Self {
            server_name: display_server_name(config),
            client,
            url: url.to_owned(),
            headers,
            session_id: None,
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
                    "version": "0.3.0"
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

    async fn call_tool(
        &mut self,
        name: &str,
        arguments: Value,
    ) -> Result<ToolCallResult, McpError> {
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
        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });
        let response = self.send_message(message).await?;
        parse_http_response(&response, id)
    }

    async fn notification(&mut self, method: &str, params: Value) -> Result<(), McpError> {
        let message = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });
        self.send_message(message).await.map(|_| ())
    }

    async fn send_message(&mut self, message: Value) -> Result<String, McpError> {
        let mut request = self
            .client
            .post(&self.url)
            .headers(self.headers.clone())
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json, text/event-stream")
            .header("MCP-Protocol-Version", MCP_PROTOCOL_VERSION)
            .json(&message);
        if let Some(session_id) = &self.session_id {
            request = request.header("Mcp-Session-Id", session_id);
        }

        let response = request.send().await.map_err(|error| {
            McpError::Http(format!(
                "remote MCP server '{}' request failed: {error}",
                self.server_name
            ))
        })?;
        let status = response.status();
        if self.session_id.is_none() {
            self.session_id = response
                .headers()
                .get("Mcp-Session-Id")
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned);
        }
        let authentication = response
            .headers()
            .get(WWW_AUTHENTICATE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let body = response
            .text()
            .await
            .map_err(|error| McpError::Http(error.to_string()))?;

        if !status.is_success() {
            let auth_hint = if status.as_u16() == 401 || status.as_u16() == 403 {
                format!(
                    " Authorization is required. Add a Bearer/API-key header or complete the server OAuth flow.{}",
                    authentication
                        .map(|value| format!(" WWW-Authenticate: {value}."))
                        .unwrap_or_default()
                )
            } else {
                String::new()
            };
            return Err(McpError::Http(format!(
                "remote MCP server '{}' returned HTTP {}.{} {}",
                self.server_name,
                status.as_u16(),
                auth_hint,
                preview_text(&body, 800)
            )));
        }

        Ok(body)
    }

    async fn close(self) {
        if self.session_id.is_none() {
            return;
        }
        let mut request = self
            .client
            .delete(&self.url)
            .headers(self.headers)
            .header("MCP-Protocol-Version", MCP_PROTOCOL_VERSION);
        if let Some(session_id) = self.session_id {
            request = request.header("Mcp-Session-Id", session_id);
        }
        let _ = request.send().await;
    }
}

enum McpSession {
    Stdio(Box<McpStdioSession>),
    Http(McpHttpSession),
}

impl McpSession {
    async fn connect(config: &McpServerConfig) -> Result<Self, McpError> {
        match config.transport {
            McpTransport::Stdio => McpStdioSession::connect(config)
                .await
                .map(Box::new)
                .map(Self::Stdio),
            McpTransport::StreamableHttp => McpHttpSession::connect(config).await.map(Self::Http),
        }
    }

    async fn list_tools(&mut self) -> Result<Vec<McpTool>, McpError> {
        match self {
            Self::Stdio(session) => session.list_tools().await,
            Self::Http(session) => session.list_tools().await,
        }
    }

    async fn call_tool(
        &mut self,
        name: &str,
        arguments: Value,
    ) -> Result<ToolCallResult, McpError> {
        match self {
            Self::Stdio(session) => session.call_tool(name, arguments).await,
            Self::Http(session) => session.call_tool(name, arguments).await,
        }
    }

    async fn close(self) {
        match self {
            Self::Stdio(session) => (*session).close().await,
            Self::Http(session) => session.close().await,
        }
    }
}

fn parse_http_response(body: &str, id: u64) -> Result<Value, McpError> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Err(McpError::Protocol(format!(
            "remote MCP response to request {id} was empty"
        )));
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return rpc_result_from_payload(value, id);
    }

    for event in trimmed.replace("\r\n", "\n").split("\n\n") {
        let data = event
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(str::trim_start)
            .collect::<Vec<_>>()
            .join("\n");
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<Value>(&data) {
            if let Ok(result) = rpc_result_from_payload(value, id) {
                return Ok(result);
            }
        }
    }

    Err(McpError::Protocol(format!(
        "remote MCP response did not contain JSON-RPC result for request {id}: {}",
        preview_text(trimmed, 800)
    )))
}

fn rpc_result_from_payload(value: Value, id: u64) -> Result<Value, McpError> {
    let values = match value {
        Value::Array(values) => values,
        value => vec![value],
    };
    let id_value = json!(id);

    for value in values {
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
            .ok_or_else(|| McpError::Protocol("MCP response has no result".to_owned()));
    }

    Err(McpError::Protocol(format!(
        "MCP response has no matching id {id}"
    )))
}

pub async fn test_mcp_server(config: McpServerConfig) -> McpConnectionTestResult {
    let server_id = config.id.clone();
    let server_name = display_server_name(&config);
    let mut session = match McpSession::connect(&config).await {
        Ok(session) => session,
        Err(error) => {
            return McpConnectionTestResult {
                server_id,
                server_name,
                connected: false,
                status: error.to_string(),
                tools: Vec::new(),
            };
        }
    };

    let result = match session.list_tools().await {
        Ok(mut tools) => {
            attach_server(&mut tools, &config);
            McpConnectionTestResult {
                server_id,
                server_name,
                connected: true,
                status: format!("connected, tools/list returned {} tool(s)", tools.len()),
                tools,
            }
        }
        Err(error) => McpConnectionTestResult {
            server_id,
            server_name,
            connected: false,
            status: format!("connected, but tools/list failed: {error}"),
            tools: Vec::new(),
        },
    };

    session.close().await;
    result
}

pub async fn prepare_mcp_context(user_message: &str, settings: &McpSettings) -> McpRuntimeContext {
    let configs = settings.enabled_servers();
    if configs.is_empty() {
        return McpRuntimeContext::disabled();
    }

    let total = configs.len();
    let mut context = McpRuntimeContext {
        enabled: true,
        status: "connecting".to_owned(),
        tools: Vec::new(),
        tool_call: None,
        tool_calls: Vec::new(),
    };
    let mut statuses = Vec::new();
    let mut sessions: Vec<(McpServerConfig, McpSession)> = Vec::new();

    for config in configs {
        let server_name = display_server_name(&config);
        let mut session = match McpSession::connect(&config).await {
            Ok(session) => session,
            Err(error) => {
                statuses.push(format!("{server_name}: unavailable ({error})"));
                continue;
            }
        };

        match session.list_tools().await {
            Ok(mut tools) => {
                attach_server(&mut tools, &config);
                statuses.push(format!("{server_name}: {} tool(s)", tools.len()));
                context.tools.extend(tools);
                sessions.push((config, session));
            }
            Err(error) => {
                statuses.push(format!("{server_name}: tools/list failed ({error})"));
                session.close().await;
            }
        }
    }

    context.status = format!(
        "{}/{} connected: {}",
        sessions.len(),
        total,
        statuses.join("; ")
    );

    if let Some(call_request) = build_tool_call_request(user_message, &context.tools) {
        let arguments_text = compact_json(&call_request.arguments);
        if let Some((config, session)) = sessions
            .iter_mut()
            .find(|(config, _)| config.id == call_request.server_id)
        {
            let server_id = config.id.clone();
            let server_name = display_server_name(config);
            let call = match session
                .call_tool(&call_request.tool_name, call_request.arguments)
                .await
            {
                Ok(result) => McpToolCallInfo {
                    server_id,
                    server_name,
                    tool_name: call_request.tool_name,
                    arguments: arguments_text,
                    result: format_tool_result(&result),
                    is_error: result.is_error,
                },
                Err(error) => McpToolCallInfo {
                    server_id,
                    server_name,
                    tool_name: call_request.tool_name,
                    arguments: arguments_text,
                    result: error.to_string(),
                    is_error: true,
                },
            };
            context.tool_call = Some(call.clone());
            context.tool_calls.push(call);
        }
    }

    for (_, session) in sessions {
        session.close().await;
    }
    context
}

pub async fn execute_mcp_tool(
    settings: &McpSettings,
    server_id: &str,
    tool_name: &str,
    arguments: Value,
) -> McpToolCallInfo {
    let arguments_text = compact_json(&arguments);
    let Some(config) = settings
        .enabled_servers()
        .into_iter()
        .find(|config| config.id == server_id)
    else {
        return McpToolCallInfo {
            server_id: server_id.to_owned(),
            server_name: server_id.to_owned(),
            tool_name: tool_name.to_owned(),
            arguments: arguments_text,
            result: "The selected MCP server is not enabled.".to_owned(),
            is_error: true,
        };
    };

    let server_name = display_server_name(&config);
    let mut session = match McpSession::connect(&config).await {
        Ok(session) => session,
        Err(error) => {
            return McpToolCallInfo {
                server_id: config.id,
                server_name,
                tool_name: tool_name.to_owned(),
                arguments: arguments_text,
                result: error.to_string(),
                is_error: true,
            };
        }
    };

    let result = match session.call_tool(tool_name, arguments).await {
        Ok(result) => McpToolCallInfo {
            server_id: config.id,
            server_name,
            tool_name: tool_name.to_owned(),
            arguments: arguments_text,
            result: format_tool_result(&result),
            is_error: result.is_error,
        },
        Err(error) => McpToolCallInfo {
            server_id: config.id,
            server_name,
            tool_name: tool_name.to_owned(),
            arguments: arguments_text,
            result: error.to_string(),
            is_error: true,
        },
    };
    session.close().await;
    result
}

pub fn build_mcp_instruction(context: &McpRuntimeContext) -> String {
    if !context.enabled {
        return String::new();
    }

    let mut sections = vec![format!(
        "MCP context:\nconnection_status: {}",
        preview_text(&context.status, 1200)
    )];

    if context.tools.is_empty() {
        sections.push(
            "Available MCP tools: none returned. If the user asked for MCP tools, explain that the enabled MCP connections did not provide a list."
                .to_owned(),
        );
    } else {
        let tools = context
            .tools
            .iter()
            .take(100)
            .map(|tool| {
                format!(
                    "- [{}] {}: {} inputSchema={}",
                    tool.server_name,
                    tool.name,
                    preview_text(
                        tool.description.as_deref().unwrap_or("No description."),
                        500
                    ),
                    compact_json(&tool.input_schema)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        sections.push(format!(
            "Available MCP tools:\n{tools}\n\n\
             Use this list when the user asks what MCP can do. MCP tools are available to an \
             automatic tool router; /mcp is only a manual fallback. For a write tool with a \
             confirmed parameter, ask for explicit confirmation after presenting the details. \
             After the user confirms, the router will execute it on the next turn. \
             Do not claim a tool was invoked unless an mcp_tool_call_result block is present."
        ));
    }

    if !context.tool_calls.is_empty() {
        let calls = context
            .tool_calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                format!(
                    "step: {}\nserver: {} ({})\ntool: {}\narguments: {}\nis_error: {}\nresult:\n{}",
                    index + 1,
                    call.server_name,
                    call.server_id,
                    call.tool_name,
                    call.arguments,
                    call.is_error,
                    preview_text(&call.result, 6000)
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        sections.push(format!("mcp_tool_call_results:\n{calls}"));
    } else {
        sections.push(
            "mcp_tool_call_result: none. A direct call can use /mcp <serverId> <toolName> {jsonArgs}; if a tool name is unique, /mcp <toolName> {jsonArgs} also works."
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
    let has_call_intent = lower.starts_with("/mcp")
        || [
            "call",
            "invoke",
            "execute",
            "run",
            "вызов",
            "вызови",
            "запуст",
            "выполн",
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
        server_id: tool.server_id.clone(),
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
        let parts = message
            .split_once('{')
            .map(|(prefix, _)| prefix)
            .unwrap_or(message)
            .split_whitespace()
            .skip(1)
            .map(clean_command_token)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();

        if parts.len() >= 2 {
            if let Some(tool) = tools.iter().find(|tool| {
                server_matches(tool, parts[0]) && tool.name.eq_ignore_ascii_case(parts[1])
            }) {
                return Some(tool);
            }
        }

        if let Some(name) = parts.first() {
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

fn server_matches(tool: &McpTool, value: &str) -> bool {
    tool.server_id.eq_ignore_ascii_case(value) || tool.server_name.eq_ignore_ascii_case(value)
}

fn clean_command_token(value: &str) -> &str {
    value.trim_matches(|character: char| !is_tool_name_character(character))
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

    if property_names.iter().all(|name| {
        property_type(properties, name).is_some_and(|kind| kind == "number" || kind == "integer")
    }) && numbers.len() >= property_names.len()
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

fn attach_server(tools: &mut [McpTool], config: &McpServerConfig) {
    let server_name = display_server_name(config);
    for tool in tools {
        tool.server_id = config.id.clone();
        tool.server_name = server_name.clone();
    }
}

fn display_server_name(config: &McpServerConfig) -> String {
    let name = config.name.trim();
    if name.is_empty() {
        config.id.clone()
    } else {
        name.to_owned()
    }
}

fn contains_word(haystack: &str, needle: &str) -> bool {
    haystack
        .split(|character: char| !is_tool_name_character(character))
        .any(|part| part == needle)
}

fn is_tool_name_character(character: char) -> bool {
    character.is_alphanumeric() || character == '_' || character == '-'
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

fn configured_command(config: &McpServerConfig) -> Result<Command, McpError> {
    let executable = config.command.trim();
    if executable.is_empty() {
        return Err(McpError::Start(format!(
            "MCP server '{}' has an empty command",
            display_server_name(config)
        )));
    }

    if cfg!(windows) && is_windows_command_shim(executable) {
        let mut command = Command::new("cmd.exe");
        command.args(["/D", "/C", executable]);
        command.args(&config.args);
        return Ok(command);
    }

    let mut command = Command::new(executable);
    command.args(&config.args);
    Ok(command)
}

fn is_windows_command_shim(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    lower.ends_with(".cmd")
        || lower.ends_with(".bat")
        || matches!(lower.as_str(), "npm" | "npx" | "pnpm" | "yarn")
}

fn default_everything_server() -> McpServerConfig {
    McpServerConfig {
        id: "everything".to_owned(),
        name: "Everything".to_owned(),
        enabled: true,
        transport: McpTransport::Stdio,
        command: "npx".to_owned(),
        args: vec!["-y".to_owned(), EVERYTHING_PACKAGE.to_owned()],
        env: BTreeMap::new(),
        cwd: None,
        url: None,
        headers: BTreeMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(server_id: &str, server_name: &str, name: &str) -> McpTool {
        McpTool {
            server_id: server_id.to_owned(),
            server_name: server_name.to_owned(),
            name: name.to_owned(),
            title: None,
            description: None,
            input_schema: json!({"type": "object"}),
        }
    }

    #[test]
    fn slash_command_can_select_server_and_tool() {
        let tools = vec![tool("one", "First", "echo"), tool("two", "Second", "echo")];
        let request = build_tool_call_request("/mcp two echo {\"message\":\"hi\"}", &tools)
            .expect("tool call should be parsed");

        assert_eq!(request.server_id, "two");
        assert_eq!(request.tool_name, "echo");
        assert_eq!(request.arguments, json!({"message": "hi"}));
    }

    #[test]
    fn unique_tool_keeps_short_slash_syntax() {
        let tools = vec![tool("one", "First", "add")];
        let request = build_tool_call_request("/mcp add {\"a\":2,\"b\":3}", &tools)
            .expect("tool call should be parsed");

        assert_eq!(request.server_id, "one");
        assert_eq!(request.tool_name, "add");
    }

    #[test]
    fn parses_streamable_http_sse_response() {
        let body =
            "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[]}}\n\n";
        let result = parse_http_response(body, 1).expect("SSE result should be parsed");

        assert_eq!(result, json!({"tools": []}));
    }

    #[test]
    fn instruction_contains_every_pipeline_tool_result() {
        let calls = vec![
            McpToolCallInfo {
                server_id: "tracker".to_owned(),
                server_name: "Tracker".to_owned(),
                tool_name: "search_tracker_issues".to_owned(),
                arguments: "{}".to_owned(),
                result: "search_id=search-1".to_owned(),
                is_error: false,
            },
            McpToolCallInfo {
                server_id: "tracker".to_owned(),
                server_name: "Tracker".to_owned(),
                tool_name: "summarize_tracker_issues".to_owned(),
                arguments: "{\"search_id\":\"search-1\"}".to_owned(),
                result: "summary_id=summary-1".to_owned(),
                is_error: false,
            },
        ];
        let context = McpRuntimeContext {
            enabled: true,
            status: "connected".to_owned(),
            tools: Vec::new(),
            tool_call: calls.last().cloned(),
            tool_calls: calls,
        };

        let instruction = build_mcp_instruction(&context);

        assert!(instruction.contains("search_tracker_issues"));
        assert!(instruction.contains("search_id=search-1"));
        assert!(instruction.contains("summarize_tracker_issues"));
        assert!(instruction.contains("summary_id=summary-1"));
    }

    #[test]
    #[ignore = "requires npm install and the local Everything MCP package"]
    fn everything_server_connects_and_lists_tools() {
        let result = tauri::async_runtime::block_on(test_mcp_server(default_everything_server()));

        assert!(result.connected, "{}", result.status);
        assert!(!result.tools.is_empty(), "{}", result.status);
        println!("{}", result.status);
        for tool in result.tools {
            println!("- [{}] {}", tool.server_name, tool.name);
        }
    }

    #[test]
    #[ignore = "requires network access to the public Cloudflare Docs MCP"]
    fn remote_http_server_connects_and_lists_tools() {
        let config = McpServerConfig {
            id: "cloudflare-docs".to_owned(),
            name: "Cloudflare Docs".to_owned(),
            enabled: true,
            transport: McpTransport::StreamableHttp,
            command: String::new(),
            args: Vec::new(),
            env: BTreeMap::new(),
            cwd: None,
            url: Some("https://docs.mcp.cloudflare.com/mcp".to_owned()),
            headers: BTreeMap::new(),
        };
        let result = tauri::async_runtime::block_on(test_mcp_server(config));

        assert!(result.connected, "{}", result.status);
        assert!(!result.tools.is_empty(), "{}", result.status);
        println!("{}", result.status);
        for tool in result.tools {
            println!("- [{}] {}", tool.server_name, tool.name);
        }
    }

    #[test]
    #[ignore = "requires YandexTrackerMCP running in mock mode on localhost:8788"]
    fn agent_calls_yandex_tracker_mcp_tool() {
        let mut headers = BTreeMap::new();
        headers.insert(
            "Authorization".to_owned(),
            "Bearer local-test-secret".to_owned(),
        );
        let settings = McpSettings {
            servers: vec![McpServerConfig {
                id: "yandex-tracker".to_owned(),
                name: "Yandex Tracker".to_owned(),
                enabled: true,
                transport: McpTransport::StreamableHttp,
                command: String::new(),
                args: Vec::new(),
                env: BTreeMap::new(),
                cwd: None,
                url: Some(
                    std::env::var("YANDEX_TRACKER_MCP_TEST_URL")
                        .unwrap_or_else(|_| "http://127.0.0.1:8788/mcp".to_owned()),
                ),
                headers,
            }],
            everything_enabled: false,
        };
        let message = "/mcp yandex-tracker create_issue {\
            \"summary\":\"Day 17 MCP demo\",\
            \"queue\":\"TEST\",\
            \"confirmed\":true,\
            \"unique\":\"day17-e2e-tracker-demo\"\
        }";

        let context = tauri::async_runtime::block_on(prepare_mcp_context(message, &settings));
        let tool = context
            .tools
            .iter()
            .find(|tool| tool.name == "create_issue")
            .expect("tools/list should return create_issue");
        let required = tool
            .input_schema
            .get("required")
            .and_then(Value::as_array)
            .expect("tool schema should declare required parameters");
        assert!(required.iter().any(|value| value == "summary"));
        assert!(required.iter().any(|value| value == "confirmed"));
        assert!(tool
            .description
            .as_deref()
            .unwrap_or_default()
            .contains("Create"));
        let call = context
            .tool_call
            .expect("agent should call create_calendar_event");

        assert_eq!(call.server_id, "yandex-tracker");
        assert_eq!(call.tool_name, "create_issue");
        assert!(!call.is_error, "{}", call.result);
        assert!(call.result.contains("Day 17 MCP demo"), "{}", call.result);
        assert!(call.result.contains("mock"), "{}", call.result);
        println!("MCP tool result:\n{}", call.result);

        let agentic_call = tauri::async_runtime::block_on(execute_mcp_tool(
            &settings,
            "yandex-tracker",
            "create_issue",
            json!({
                "summary": "Agent-selected MCP tool",
                "queue": "TEST",
                "confirmed": true,
                "unique": "day17-agent-selected-tool"
            }),
        ));
        assert!(!agentic_call.is_error, "{}", agentic_call.result);
        assert!(
            agentic_call.result.contains("Agent-selected MCP tool"),
            "{}",
            agentic_call.result
        );
    }
}
