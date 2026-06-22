mod agent;
mod mcp;

use agent::{
    cancel_agent_request, clear_agent_request, Agent, AgentReply, AgentRequest, AgentStreamChunk,
    AgentSwarmStatus as AgentSwarmStatusPayload,
};
use serde::Serialize;
use tauri::{AppHandle, Emitter};

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentStreamDelta {
    request_id: String,
    delta: String,
    channel: String,
    actor: Option<String>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentMemoryStarted {
    request_id: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentSwarmStatus {
    request_id: String,
    actors: Vec<String>,
    active_actor: Option<String>,
    status: String,
}

#[tauri::command]
async fn send_agent_message(
    app_handle: AppHandle,
    request: AgentRequest,
) -> Result<AgentReply, String> {
    let agent = Agent::from_request(&request).map_err(|error| error.to_string())?;
    let request_id = request.request_id.clone();
    let stream_request_id = request.request_id.clone();
    let memory_request_id = request.request_id.clone();
    let swarm_request_id = request.request_id.clone();
    let stream_app_handle = app_handle.clone();
    let swarm_app_handle = app_handle.clone();

    let result = agent
        .send_stream(
            request.messages,
            move |chunk: AgentStreamChunk| {
                let _ = stream_app_handle.emit(
                    "agent_stream_delta",
                    AgentStreamDelta {
                        request_id: stream_request_id.clone(),
                        delta: chunk.delta,
                        channel: chunk.channel,
                        actor: chunk.actor,
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
            move |status: AgentSwarmStatusPayload| {
                let _ = swarm_app_handle.emit(
                    "agent_swarm_status",
                    AgentSwarmStatus {
                        request_id: swarm_request_id.clone(),
                        actors: status.actors,
                        active_actor: status.active_actor,
                        status: status.status,
                    },
                );
            },
        )
        .await
        .map_err(|error| error.to_string());

    clear_agent_request(&request_id);
    result
}

#[tauri::command]
fn cancel_agent_message(request_id: String) -> Result<(), String> {
    cancel_agent_request(&request_id);
    Ok(())
}

pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            send_agent_message,
            cancel_agent_message
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
