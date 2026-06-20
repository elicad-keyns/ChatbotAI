mod agent;

use agent::{Agent, AgentReply, AgentRequest};
use serde::Serialize;
use tauri::{AppHandle, Emitter};

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentStreamDelta {
    request_id: String,
    delta: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentMemoryStarted {
    request_id: String,
}

#[tauri::command]
async fn send_agent_message(
    app_handle: AppHandle,
    request: AgentRequest,
) -> Result<AgentReply, String> {
    let agent = Agent::from_request(&request).map_err(|error| error.to_string())?;
    let stream_request_id = request.request_id.clone();
    let memory_request_id = request.request_id.clone();
    let stream_app_handle = app_handle.clone();

    agent
        .send_stream(
            request.messages,
            move |delta| {
                let _ = stream_app_handle.emit(
                    "agent_stream_delta",
                    AgentStreamDelta {
                        request_id: stream_request_id.clone(),
                        delta: delta.to_owned(),
                    },
                );
            },
            move || {
                let _ = app_handle.emit(
                    "agent_memory_started",
                    AgentMemoryStarted {
                        request_id: memory_request_id.clone(),
                    },
                );
            },
        )
        .await
        .map_err(|error| error.to_string())
}

pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![send_agent_message])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
