export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
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

export type MemoryLayerId = "shortTerm" | "working" | "longTerm";

export interface MemoryItem {
  id: string;
  content: string;
  createdAt: string;
  updatedAt: string;
  sourceChatId?: string;
  sourceMessage?: string;
}

export interface MemoryContext {
  shortTerm: ChatMessage[];
  shortTermSummary?: ShortTermSummary;
  working: MemoryItem[];
  longTerm: MemoryItem[];
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
  promptPreview: string;
  memoryRouterInput: string;
  memoryRouterRaw: string;
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
}

export interface AgentRequest {
  requestId: string;
  apiKey?: string;
  model: string;
  systemPrompt: string;
  messages: ChatMessage[];
  memoryContext: MemoryContext;
  shortTermCompression: ShortTermCompressionSettings;
}

export interface AgentStreamDelta {
  requestId: string;
  delta: string;
}

export interface AgentMemoryStarted {
  requestId: string;
}
