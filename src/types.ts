export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
  kind?: "text" | "swarm";
  swarm?: SwarmDiscussion;
  mcpSteps?: McpExecutionStep[];
}

export interface McpExecutionStep {
  serverName: string;
  toolName: string;
  label: string;
  status: "running" | "completed" | "failed";
}

export interface SwarmDiscussion {
  actors: string[];
  logs: Record<string, string>;
  status: string;
}

export interface ShortTermSummary {
  content: string;
  compressedTurnCount: number;
  compressedMessageCount: number;
  updatedAt: string;
}

export interface ShortTermCompressionSettings {
  enabled: boolean;
  maxUncompressedTurns: number;
}

export type MemoryLayerId = "shortTerm" | "working" | "longTerm" | "userProfile";

export interface UserProfile {
  id: string;
  name: string;
  style: string;
  format: string;
  constraints: string;
  context: string;
  createdAt: string;
  updatedAt: string;
}

export interface MemoryItem {
  id: string;
  content: string;
  createdAt: string;
  updatedAt: string;
  sourceChatId?: string;
  sourceMessage?: string;
}

export type TaskPhase = "planning" | "execution" | "validation" | "done";
export type OrchestratorAction =
  | "userMessage"
  | "approvePlan"
  | "approveSolution"
  | "disputeSolution"
  | "cancelTask"
  | "debugTransition";

export interface TaskState {
  phase: TaskPhase;
  task: string;
  step: number;
  totalSteps: number;
  draftPlan: string;
  approvedPlan: string;
  solution: string;
  validationReport: string;
  violations: string[];
  done: string[];
  currentStep: string;
  expectedAction: string;
  isPaused: boolean;
  isCancelled: boolean;
  updatedAt: string;
}

export interface OrchestrationSettings {
  enabled: boolean;
  action?: OrchestratorAction;
  validatorInvariants: string;
}

export interface McpSettings {
  servers: McpServerConfig[];
}

export interface McpServerConfig {
  id: string;
  name: string;
  enabled: boolean;
  transport: "stdio" | "streamableHttp";
  command: string;
  args: string[];
  env: Record<string, string>;
  cwd?: string;
  url?: string;
  headers: Record<string, string>;
}

export interface McpTool {
  serverId: string;
  serverName: string;
  name: string;
  title?: string;
  description?: string;
  inputSchema?: unknown;
}

export interface McpToolCallInfo {
  serverId: string;
  serverName: string;
  toolName: string;
  arguments: string;
  result: string;
  isError: boolean;
}

export interface McpConnectionTestResult {
  serverId: string;
  serverName: string;
  connected: boolean;
  status: string;
  tools: McpTool[];
}

export interface MemoryContext {
  activeProfile?: UserProfile;
  shortTerm: ChatMessage[];
  shortTermSummary?: ShortTermSummary;
  working: MemoryItem[];
  longTerm: MemoryItem[];
  taskState?: TaskState;
}

export interface MemoryDebugInfo {
  includedLayers: MemoryLayerId[];
  shortTermMessageCount: number;
  workingItemCount: number;
  longTermItemCount: number;
  memoryInstructionChars: number;
  inputMessageCount: number;
  shortTermVisibleMessageCount: number;
  shortTermInputMessageCount: number;
  shortTermSummaryChars: number;
  shortTermCompressedTurnCount: number;
  shortTermCompressionEnabled: boolean;
  shortTermCompressionLimit: number;
  shortTermCompressionTriggered: boolean;
  shortTermCompressionInput: string;
  shortTermCompressionRaw: string;
  activeProfileName?: string;
  activeProfileChars: number;
  promptPreview: string;
  memoryRouterInput: string;
  memoryRouterRaw: string;
  taskPhase?: TaskPhase;
  taskCurrentStep: string;
  taskExpectedAction: string;
  taskPaused: boolean;
  orchestratorEnabled: boolean;
  orchestratorAgent: string;
  orchestratorAction: string;
  validatorViolations: string[];
  mcpEnabled: boolean;
  mcpStatus: string;
  mcpToolCount: number;
  mcpTools: McpTool[];
  mcpToolCall?: McpToolCallInfo;
  mcpToolCalls: McpToolCallInfo[];
}

export interface MemoryDecision {
  layer: "working" | "longTerm";
  action: "save" | "skip";
  memoryText: string;
  reason: string;
}

export interface AgentReply {
  content: string;
  model: string;
  usage?: {
    inputTokens?: number;
    outputTokens?: number;
    totalTokens?: number;
  };
  shortTermSummary?: ShortTermSummary;
  debug?: MemoryDebugInfo;
  memoryDecisions?: MemoryDecision[];
  taskState?: TaskState;
}

export interface AgentRequest {
  requestId: string;
  apiKey?: string;
  model: string;
  systemPrompt: string;
  messages: ChatMessage[];
  memoryContext: MemoryContext;
  shortTermCompression: ShortTermCompressionSettings;
  orchestration?: OrchestrationSettings;
  mcp?: McpSettings;
}

export interface AgentStreamDelta {
  requestId: string;
  delta: string;
  channel?: "final" | "swarm" | "mcp";
  actor?: string;
  serverName?: string;
}

export interface AgentMemoryStarted {
  requestId: string;
}

export interface AgentSwarmStatus {
  requestId: string;
  actors: string[];
  activeActor?: string;
  status: string;
}
