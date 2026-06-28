import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  AgentReply,
  AgentMemoryStarted,
  AgentSwarmStatus,
  ChatMessage,
  MemoryContext,
  MemoryDecision,
  MemoryItem,
  MemoryLayerId,
  McpConnectionTestResult,
  McpExecutionStep,
  McpServerConfig,
  AgentStreamDelta,
  OrchestratorAction,
  ShortTermCompressionSettings,
  ShortTermSummary,
  SwarmDiscussion,
  TaskPhase,
  TaskState,
  UserProfile
} from "./types";

const DEFAULT_SYSTEM_PROMPT =
  "You are a helpful AI agent inside a desktop application. Answer clearly and keep context from the conversation.";

const SETTINGS_STORAGE_KEY = "chatbot-ai.settings.v1";
const CHATS_STORAGE_KEY = "chatbot-ai.chats.v1";
const WORKING_MEMORY_STORAGE_KEY = "chatbot-ai.memory.working.v1";
const TASK_STATE_STORAGE_KEY = "chatbot-ai.task-state.v1";
const USER_PROFILES_STORAGE_KEY = "chatbot-ai.user-profiles.v1";
const ACTIVE_PROFILE_STORAGE_KEY = "chatbot-ai.active-profile.v1";
const FIRST_PROFILE_NAME = "Профиль 1";
const GENERATED_PROFILE_DEFAULTS = {
  name: "Основной профиль",
  style: "Дружелюбно, ясно, по делу.",
  format: "Сначала краткий ответ, затем шаги или списки, если они помогают.",
  constraints: "Не перегружать ответ лишней теорией. Учитывать русский язык интерфейса.",
  context: "Пользователь работает с desktop AI-agent и учебными задачами по stateful-агентам."
};
const DEFAULT_VALIDATOR_INVARIANTS = `# Кодовые проверки валидатора, по одной на строку
# Форматы: must: текст, forbid: текст
forbid: RxJava
forbid: AsyncTask`;

const ICONS = {
  ai: new URL("../src-tauri/new_icons/ai_100.png", import.meta.url).href,
  arrowDown: new URL("../src-tauri/new_icons/arrow_down_100.png", import.meta.url).href,
  arrowLeft: new URL("../src-tauri/new_icons/arrow_left_100.png", import.meta.url).href,
  arrowRight: new URL("../src-tauri/new_icons/arrow_right_100.png", import.meta.url).href,
  cross: new URL("../src-tauri/new_icons/cross_100.png", import.meta.url).href,
  plus: new URL("../src-tauri/new_icons/plus_100.png", import.meta.url).href,
  sendLetter: new URL("../src-tauri/new_icons/send_letter_100.png", import.meta.url).href,
  settings: new URL("../src-tauri/new_icons/settings_100.png", import.meta.url).href
};

const MODEL_OPTIONS = [
  { id: "gpt-5.5", label: "GPT-5.5" },
  { id: "gpt-5.5-pro", label: "GPT-5.5 Pro" },
  { id: "gpt-5.4", label: "GPT-5.4" },
  { id: "gpt-5.4-pro", label: "GPT-5.4 Pro" },
  { id: "gpt-5.4-mini", label: "GPT-5.4 mini" },
  { id: "gpt-5.4-nano", label: "GPT-5.4 nano" },
  { id: "gpt-5.2", label: "GPT-5.2" },
  { id: "gpt-5.2-pro", label: "GPT-5.2 Pro" },
  { id: "gpt-5.1", label: "GPT-5.1" },
  { id: "gpt-5", label: "GPT-5" },
  { id: "gpt-5-pro", label: "GPT-5 Pro" },
  { id: "gpt-5-mini", label: "GPT-5 mini" },
  { id: "gpt-5-nano", label: "GPT-5 nano" },
  { id: "o3-pro", label: "o3-pro" },
  { id: "o3", label: "o3" },
  { id: "gpt-4.1", label: "GPT-4.1" },
  { id: "gpt-4.1-mini", label: "GPT-4.1 mini" },
  { id: "gpt-4o-mini", label: "GPT-4o mini (legacy)" },
  { id: "gpt-4o", label: "GPT-4o (legacy)" }
];

const MEMORY_LAYER_LABELS: Record<
  MemoryLayerId,
  { title: string; scope: string; description: string }
> = {
  shortTerm: {
    title: "Краткосрочная",
    scope: "текущий чат",
    description: "История сообщений только выбранного чата. Очищается вместе с чатом."
  },
  working: {
    title: "Рабочая",
    scope: "все чаты",
    description: "Только временный контекст активной задачи: фича, баг, файлы, ограничения, решения."
  },
  longTerm: {
    title: "Долгосрочная",
    scope: "все чаты",
    description: "Профиль пользователя: навыки, профессия, языки, цели, проекты и предпочтения."
  },
  userProfile: {
    title: "Профиль",
    scope: "активный пользователь",
    description: "Явные предпочтения ответа: стиль, формат, ограничения и контекст пользователя."
  }
};

const TASK_PHASES: { id: TaskPhase; label: string; description: string }[] = [
  {
    id: "planning",
    label: "Planning",
    description: "план"
  },
  {
    id: "execution",
    label: "Execution",
    description: "работа"
  },
  {
    id: "validation",
    label: "Validation",
    description: "проверка"
  },
  {
    id: "done",
    label: "Done",
    description: "готово"
  }
];

const TASK_PHASE_DEFAULTS: Record<
  TaskPhase,
  { currentStep: string; expectedAction: string }
> = {
  planning: {
    currentStep: "Сформировать план задачи",
    expectedAction: "Опишите цель или подтвердите план переходом к execution."
  },
  execution: {
    currentStep: "Выполнить согласованный план",
    expectedAction: "Дайте команду на реализацию или уточните рабочие ограничения."
  },
  validation: {
    currentStep: "Проверить результат",
    expectedAction: "Запустите проверку, ревью или укажите, что нужно исправить."
  },
  done: {
    currentStep: "Задача завершена",
    expectedAction: "Напишите новый запрос, чтобы начать новую задачу."
  }
};

const TASK_CANCELLED_DEFAULTS = {
  currentStep: "Задача отменена пользователем.",
  expectedAction: "Напишите новый запрос, чтобы начать новую задачу."
};
const TASK_CANCEL_MESSAGE =
  "🛑 Задача отменена. Оркестратор больше не будет продолжать planning, execution или validation для этой задачи.";

const TASK_PHASE_ORDER = TASK_PHASES.map((phase) => phase.id);
const TASK_ALLOWED_TRANSITIONS: Record<TaskPhase, TaskPhase[]> = {
  planning: ["execution"],
  execution: ["planning", "validation"],
  validation: ["execution", "done"],
  done: []
};

type ThemeMode = "light" | "dark";
type PersistentMemoryLayer = "working" | "longTerm";
type MemoryAction = "saved" | "deleted" | "skipped";
type AgentPhase = "idle" | "compressing" | "streaming" | "memory";

interface AppSettings {
  apiKey: string;
  model: string;
  systemPrompt: string;
  theme: ThemeMode;
  autoScroll: boolean;
  shortTermCompressionEnabled: boolean;
  shortTermCompressionTurnLimit: number;
  orchestrationEnabled: boolean;
  mcpServers: McpServerConfig[];
  validatorInvariants: string;
  debugManualStateControls: boolean;
}

interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  shortTermSummary?: ShortTermSummary;
  createdAt: string;
  updatedAt: string;
}

interface StoredUserProfile extends UserProfile {
  longTermMemory: MemoryItem[];
}

interface MemoryWrite {
  id: string;
  action: MemoryAction;
  layer: MemoryLayerId;
  content: string;
  createdAt: string;
  reason: string;
}

function ButtonIcon({ className = "", src }: { className?: string; src: string }) {
  return (
    <img
      className={`button-icon ${className}`.trim()}
      src={src}
      alt=""
      aria-hidden="true"
      draggable={false}
    />
  );
}

const DEFAULT_SETTINGS: AppSettings = {
  apiKey: "",
  model: "gpt-4.1-mini",
  systemPrompt: DEFAULT_SYSTEM_PROMPT,
  theme: "light",
  autoScroll: true,
  shortTermCompressionEnabled: true,
  shortTermCompressionTurnLimit: 10,
  orchestrationEnabled: false,
  mcpServers: [
    {
      id: "everything",
      name: "Everything",
      enabled: true,
      transport: "stdio",
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-everything"],
      env: {},
      headers: {}
    }
  ],
  validatorInvariants: DEFAULT_VALIDATOR_INVARIANTS,
  debugManualStateControls: false
};

function createEmptyChat(): ChatSession {
  const now = new Date().toISOString();

  return {
    id: crypto.randomUUID(),
    title: "Новый чат",
    messages: [],
    createdAt: now,
    updatedAt: now
  };
}

function createMemoryItem(
  content: string,
  source?: { chatId?: string; message?: string }
): MemoryItem {
  const now = new Date().toISOString();

  return {
    id: crypto.randomUUID(),
    content: content.trim(),
    createdAt: now,
    updatedAt: now,
    sourceChatId: source?.chatId,
    sourceMessage: source?.message
  };
}

function createDefaultUserProfile(): StoredUserProfile {
  const now = new Date().toISOString();

  return {
    id: crypto.randomUUID(),
    name: FIRST_PROFILE_NAME,
    style: "",
    format: "",
    constraints: "",
    context: "",
    createdAt: now,
    updatedAt: now,
    longTermMemory: []
  };
}

function createBlankUserProfile(index: number): StoredUserProfile {
  const now = new Date().toISOString();

  return {
    id: crypto.randomUUID(),
    name: `Профиль ${index}`,
    style: "",
    format: "",
    constraints: "",
    context: "",
    createdAt: now,
    updatedAt: now,
    longTermMemory: []
  };
}

function loadSettings(): AppSettings {
  try {
    const saved = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!saved) {
      return DEFAULT_SETTINGS;
    }

    const parsed = JSON.parse(saved) as Partial<AppSettings> & {
      mcpEverythingEnabled?: boolean;
    };
    const mcpServers = Array.isArray(parsed.mcpServers)
      ? normalizeMcpServers(parsed.mcpServers)
      : DEFAULT_SETTINGS.mcpServers.map((server) => ({
          ...server,
          enabled: parsed.mcpEverythingEnabled ?? server.enabled,
          args: [...server.args],
          env: { ...server.env },
          headers: { ...server.headers }
        }));
    return {
      ...DEFAULT_SETTINGS,
      ...parsed,
      mcpServers,
      shortTermCompressionTurnLimit: normalizeCompressionTurnLimit(
        parsed.shortTermCompressionTurnLimit
      )
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function normalizeMcpServers(servers: McpServerConfig[]): McpServerConfig[] {
  return servers
    .filter((server) => server && typeof server === "object")
    .map((server, index) => ({
      id:
        typeof server.id === "string" && server.id.trim()
          ? server.id.trim()
          : `mcp-${index + 1}`,
      name:
        typeof server.name === "string" && server.name.trim()
          ? server.name.trim()
          : `MCP ${index + 1}`,
      enabled: server.enabled !== false,
      transport:
        server.transport === "streamableHttp" ? "streamableHttp" : "stdio",
      command: typeof server.command === "string" ? server.command : "",
      args: Array.isArray(server.args)
        ? server.args.filter((arg): arg is string => typeof arg === "string")
        : [],
      env:
        server.env && typeof server.env === "object" && !Array.isArray(server.env)
          ? Object.fromEntries(
              Object.entries(server.env)
                .filter(
                  (entry): entry is [string, string] =>
                    Boolean(entry[0].trim()) && typeof entry[1] === "string"
                )
            )
          : {},
      cwd: typeof server.cwd === "string" ? server.cwd : undefined,
      url: typeof server.url === "string" ? server.url : undefined,
      headers:
        server.headers && typeof server.headers === "object" && !Array.isArray(server.headers)
          ? Object.fromEntries(
              Object.entries(server.headers).filter(
                (entry): entry is [string, string] =>
                  Boolean(entry[0].trim()) && typeof entry[1] === "string"
              )
            )
          : {}
    }));
}

function createMcpServer(
  index: number,
  transport: McpServerConfig["transport"]
): McpServerConfig {
  return {
    id: crypto.randomUUID(),
    name: transport === "streamableHttp" ? `Remote MCP ${index}` : `MCP ${index}`,
    enabled: true,
    transport,
    command: transport === "stdio" ? "npx" : "",
    args: [],
    env: {},
    url: transport === "streamableHttp" ? "https://" : undefined,
    headers: {}
  };
}

function normalizeCompressionTurnLimit(value: unknown): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return DEFAULT_SETTINGS.shortTermCompressionTurnLimit;
  }

  return Math.min(50, Math.max(2, Math.round(parsed)));
}

function loadChats(): ChatSession[] {
  try {
    const saved = localStorage.getItem(CHATS_STORAGE_KEY);
    if (!saved) {
      return [createEmptyChat()];
    }

    const parsed = JSON.parse(saved) as ChatSession[];
    if (!Array.isArray(parsed) || parsed.length === 0) {
      return [createEmptyChat()];
    }

    return parsed.map((chat) => ({
      ...chat,
      messages: Array.isArray(chat.messages)
        ? chat.messages
            .map((message) => normalizeChatMessage(message))
            .filter((message): message is ChatMessage => Boolean(message))
        : [],
      shortTermSummary: normalizeShortTermSummary(chat.shortTermSummary)
    }));
  } catch {
    return [createEmptyChat()];
  }
}

function normalizeShortTermSummary(
  summary: ChatSession["shortTermSummary"]
): ShortTermSummary | undefined {
  if (!summary || typeof summary.content !== "string" || !summary.content.trim()) {
    return undefined;
  }

  return {
    content: summary.content.trim(),
    compressedTurnCount:
      typeof summary.compressedTurnCount === "number"
        ? Math.max(0, Math.floor(summary.compressedTurnCount))
        : 0,
    compressedMessageCount:
      typeof summary.compressedMessageCount === "number"
        ? Math.max(0, Math.floor(summary.compressedMessageCount))
        : 0,
    updatedAt:
      typeof summary.updatedAt === "string" ? summary.updatedAt : new Date().toISOString()
  };
}

function normalizeMemoryItems(items: Partial<MemoryItem>[] | undefined): MemoryItem[] {
  if (!Array.isArray(items)) {
    return [];
  }

  return items
    .filter((item) => typeof item.content === "string" && item.content.trim())
    .map((item) => {
      const now = new Date().toISOString();

      return {
        id: typeof item.id === "string" ? item.id : crypto.randomUUID(),
        content: item.content?.trim() ?? "",
        createdAt: typeof item.createdAt === "string" ? item.createdAt : now,
        updatedAt: typeof item.updatedAt === "string" ? item.updatedAt : now,
        sourceChatId:
          typeof item.sourceChatId === "string" ? item.sourceChatId : undefined,
        sourceMessage:
          typeof item.sourceMessage === "string" ? item.sourceMessage : undefined
      };
    });
}

function loadMemoryItems(storageKey: string): MemoryItem[] {
  try {
    const saved = localStorage.getItem(storageKey);
    if (!saved) {
      return [];
    }

    const parsed = JSON.parse(saved) as Partial<MemoryItem>[];
    if (!Array.isArray(parsed)) {
      return [];
    }

    return normalizeMemoryItems(parsed);
  } catch {
    return [];
  }
}

function normalizeStoredUserProfile(
  profile: Partial<StoredUserProfile>,
  index: number
): StoredUserProfile | null {
  if (!profile || typeof profile !== "object") {
    return null;
  }

  const now = new Date().toISOString();
  const id = typeof profile.id === "string" && profile.id.trim()
    ? profile.id
    : crypto.randomUUID();
  const rawName = typeof profile.name === "string" ? profile.name.trim() : "";
  const name = rawName
    ? rawName === GENERATED_PROFILE_DEFAULTS.name
      ? FIRST_PROFILE_NAME
      : rawName
    : `Профиль ${index + 1}`;
  const style = typeof profile.style === "string" ? profile.style : "";
  const format = typeof profile.format === "string" ? profile.format : "";
  const constraints = typeof profile.constraints === "string" ? profile.constraints : "";
  const context = typeof profile.context === "string" ? profile.context : "";

  return {
    id,
    name,
    style: style === GENERATED_PROFILE_DEFAULTS.style ? "" : style,
    format: format === GENERATED_PROFILE_DEFAULTS.format ? "" : format,
    constraints: constraints === GENERATED_PROFILE_DEFAULTS.constraints ? "" : constraints,
    context: context === GENERATED_PROFILE_DEFAULTS.context ? "" : context,
    createdAt: typeof profile.createdAt === "string" ? profile.createdAt : now,
    updatedAt: typeof profile.updatedAt === "string" ? profile.updatedAt : now,
    longTermMemory: normalizeMemoryItems(profile.longTermMemory)
  };
}

function loadUserProfiles(): StoredUserProfile[] {
  try {
    const saved = localStorage.getItem(USER_PROFILES_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved) as Partial<StoredUserProfile>[];
      if (Array.isArray(parsed)) {
        const profiles = parsed
          .map((profile, index) => normalizeStoredUserProfile(profile, index))
          .filter((profile): profile is StoredUserProfile => Boolean(profile));

        if (profiles.length > 0) {
          return profiles;
        }
      }
    }
  } catch {
    // Fall through to the default profile.
  }

  return [createDefaultUserProfile()];
}

function loadActiveProfileId(profiles: StoredUserProfile[]): string {
  try {
    const saved = localStorage.getItem(ACTIVE_PROFILE_STORAGE_KEY);
    if (saved && profiles.some((profile) => profile.id === saved)) {
      return saved;
    }
  } catch {
    // Use the first available profile.
  }

  return profiles[0]?.id ?? "";
}

function buildChatTitle(messages: ChatMessage[]): string {
  const firstUserMessage = messages.find((message) => message.role === "user");
  if (!firstUserMessage) {
    return "Новый чат";
  }

  const title = firstUserMessage.content.trim().replace(/\s+/g, " ");
  return title.length > 34 ? `${title.slice(0, 34)}...` : title;
}

function formatChatTime(value: string): string {
  return new Intl.DateTimeFormat("ru", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

function previewText(value: string, maxLength = 130): string {
  const compact = value.trim().replace(/\s+/g, " ");
  return compact.length > maxLength ? `${compact.slice(0, maxLength)}...` : compact;
}

function formatJsonPreview(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function getRoleLabel(role: ChatMessage["role"]): string {
  return role === "user" ? "Вы" : "Agent";
}

function hasAnySwarmLog(logs: Record<string, string>): boolean {
  return Object.values(logs).some((value) => value.trim());
}

function getOrderedSwarmActors(
  actors: string[],
  logs: Record<string, string>
): string[] {
  const seen = new Set<string>();
  const ordered: string[] = [];

  for (const actor of actors) {
    if (actor && logs[actor]?.trim() && !seen.has(actor)) {
      seen.add(actor);
      ordered.push(actor);
    }
  }

  for (const actor of Object.keys(logs)) {
    if (actor && logs[actor]?.trim() && !seen.has(actor)) {
      seen.add(actor);
      ordered.push(actor);
    }
  }

  return ordered;
}

function normalizeSwarmLogs(logs: Record<string, unknown>): Record<string, string> {
  return Object.fromEntries(
    Object.entries(logs)
      .filter((entry): entry is [string, string] => typeof entry[1] === "string")
      .map(([actor, log]) => [actor.trim(), log.trim()] as const)
      .filter(([actor, log]) => actor && log)
  );
}

function formatSwarmTranscript(actors: string[], logs: Record<string, string>): string {
  const normalizedLogs = normalizeSwarmLogs(logs);
  const orderedActors = getOrderedSwarmActors(actors, normalizedLogs);

  return [
    "Execution Swarm",
    ...orderedActors.map((actor) => `${actor}\n${normalizedLogs[actor]}`)
  ].join("\n\n");
}

function createSwarmDiscussionMessage(
  actors: string[],
  logs: Record<string, string>,
  status: string
): ChatMessage | undefined {
  const normalizedLogs = normalizeSwarmLogs(logs);
  if (!hasAnySwarmLog(normalizedLogs)) {
    return undefined;
  }

  const orderedActors = getOrderedSwarmActors(actors, normalizedLogs);
  const discussion: SwarmDiscussion = {
    actors: orderedActors,
    logs: normalizedLogs,
    status: status.trim() || "Execution Swarm завершил обсуждение."
  };

  return {
    role: "assistant",
    kind: "swarm",
    content: formatSwarmTranscript(orderedActors, normalizedLogs),
    swarm: discussion
  };
}

function normalizeSwarmDiscussion(value: unknown): SwarmDiscussion | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }

  const raw = value as Partial<SwarmDiscussion>;
  const logs =
    raw.logs && typeof raw.logs === "object"
      ? normalizeSwarmLogs(raw.logs as Record<string, unknown>)
      : {};

  if (!hasAnySwarmLog(logs)) {
    return undefined;
  }

  const actors = Array.isArray(raw.actors)
    ? raw.actors.filter((actor): actor is string => typeof actor === "string")
    : Object.keys(logs);

  return {
    actors: getOrderedSwarmActors(actors, logs),
    logs,
    status: typeof raw.status === "string" && raw.status.trim()
      ? raw.status.trim()
      : "Execution Swarm"
  };
}

function normalizeChatMessage(message: Partial<ChatMessage>): ChatMessage | null {
  if (!message || (message.role !== "user" && message.role !== "assistant")) {
    return null;
  }

  if (message.kind === "swarm") {
    const swarm = normalizeSwarmDiscussion(message.swarm);
    if (!swarm) {
      return null;
    }

    return {
      role: "assistant",
      kind: "swarm",
      content: formatSwarmTranscript(swarm.actors, swarm.logs),
      swarm
    };
  }

  if (typeof message.content !== "string") {
    return null;
  }

  return {
    role: message.role,
    content: message.content,
    mcpSteps: normalizeMcpExecutionSteps(message.mcpSteps)
  };
}

function getMcpToolLabel(toolName: string): string {
  const labels: Record<string, string> = {
    search_tracker_issues: "Ищу задачи в Yandex Tracker",
    summarize_tracker_issues: "Формирую сводку по задачам",
    save_tracker_report: "Сохраняю Markdown-отчёт",
    send_tracker_artifact: "Отправляю отчёт в Telegram",
    get_delivery_status: "Проверяю доставку в Telegram",
    get_telegram_service_status: "Проверяю готовность Telegram-бота",
    create_issue: "Создаю задачу в Yandex Tracker",
    update_issue: "Обновляю задачу в Yandex Tracker",
    search_issues: "Ищу задачи в Yandex Tracker",
    schedule_tracker_report: "Создаю расписание отчёта"
  };
  return labels[toolName] ?? `Вызываю инструмент ${toolName}`;
}

function normalizeMcpExecutionSteps(value: unknown): McpExecutionStep[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const steps = value.flatMap((entry) => {
    if (!entry || typeof entry !== "object") {
      return [];
    }
    const raw = entry as Partial<McpExecutionStep>;
    if (
      typeof raw.toolName !== "string" ||
      !["running", "completed", "failed"].includes(raw.status ?? "")
    ) {
      return [];
    }
    return [{
      serverName: typeof raw.serverName === "string" && raw.serverName.trim()
        ? raw.serverName
        : "MCP",
      toolName: raw.toolName,
      label: typeof raw.label === "string" && raw.label.trim()
        ? raw.label
        : getMcpToolLabel(raw.toolName),
      status: raw.status as McpExecutionStep["status"]
    }];
  });
  return steps.length > 0 ? steps : undefined;
}

function updateMcpExecutionSteps(
  current: McpExecutionStep[],
  serverName: string,
  toolName: string,
  status: McpExecutionStep["status"]
): McpExecutionStep[] {
  if (status === "running") {
    return [
      ...current,
      { serverName, toolName, label: getMcpToolLabel(toolName), status }
    ];
  }
  let index = -1;
  for (let stepIndex = current.length - 1; stepIndex >= 0; stepIndex -= 1) {
    if (current[stepIndex].toolName === toolName && current[stepIndex].status === "running") {
      index = stepIndex;
      break;
    }
  }
  if (index < 0) {
    return [
      ...current,
      { serverName, toolName, label: getMcpToolLabel(toolName), status }
    ];
  }
  return current.map((step, stepIndex) =>
    stepIndex === index ? { ...step, status } : step
  );
}

function SwarmDiscussionPanel({
  discussion,
  activeActor
}: {
  discussion: SwarmDiscussion;
  activeActor?: string | null;
}) {
  const actors = getOrderedSwarmActors(discussion.actors, discussion.logs);

  return (
    <div className="swarm-stream-panel" aria-label="Обсуждение execution swarm">
      <div className="swarm-stream-header">
        <strong>Execution Swarm</strong>
        <span>{discussion.status || "Обсуждение решения"}</span>
      </div>
      <div className="swarm-stream-logs">
        {actors.map((actor) => (
          <div
            className={`swarm-stream-log ${actor === activeActor ? "active" : ""}`}
            key={actor}
          >
            <b>{actor}</b>
            <p>{discussion.logs[actor]}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function McpExecutionPanel({ steps }: { steps: McpExecutionStep[] }) {
  if (steps.length === 0) {
    return null;
  }
  return (
    <div className="mcp-execution-panel" role="status" aria-live="polite">
      <div className="mcp-execution-title">MCP pipeline</div>
      {steps.map((step, index) => (
        <div className={`mcp-execution-step ${step.status}`} key={`${step.toolName}-${index}`}>
          <span className="mcp-execution-indicator" aria-hidden="true" />
          <span>
            <b>{step.serverName}</b> · {step.label}
          </span>
          <small>
            {step.status === "running"
              ? "выполняется"
              : step.status === "completed"
                ? "готово"
                : "ошибка"}
          </small>
        </div>
      ))}
    </div>
  );
}

function getMemoryActionLabel(action: MemoryAction): string {
  if (action === "saved") {
    return "сохранено";
  }

  if (action === "deleted") {
    return "удалено";
  }

  return "пропущено";
}

function createMemoryWrite(
  action: MemoryAction,
  layer: MemoryLayerId,
  content: string,
  reason: string,
  id: string = crypto.randomUUID()
): MemoryWrite {
  return {
    id,
    action,
    layer,
    content: previewText(content, 180),
    createdAt: new Date().toISOString(),
    reason
  };
}

function normalizeMemoryText(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

function hasSimilarMemory(items: MemoryItem[], content: string): boolean {
  const normalizedContent = normalizeMemoryText(content);
  return items.some((item) => normalizeMemoryText(item.content) === normalizedContent);
}

function shouldCompressShortTerm(
  messages: ChatMessage[],
  settings: AppSettings
): boolean {
  if (!settings.shortTermCompressionEnabled) {
    return false;
  }

  const turnCount = messages.filter((message) => message.role === "user").length;
  return turnCount > settings.shortTermCompressionTurnLimit;
}

function createTaskState(phase: TaskPhase = "planning"): TaskState {
  const defaults = TASK_PHASE_DEFAULTS[phase];

  return {
    phase,
    task: "",
    step: Math.max(0, getTaskPhaseIndex(phase)),
    totalSteps: TASK_PHASES.length,
    draftPlan: "",
    approvedPlan: "",
    solution: "",
    validationReport: "",
    violations: [],
    done: [],
    currentStep: defaults.currentStep,
    expectedAction: defaults.expectedAction,
    isPaused: false,
    isCancelled: false,
    updatedAt: new Date().toISOString()
  };
}

function isTaskPhase(value: unknown): value is TaskPhase {
  return typeof value === "string" && TASK_PHASE_ORDER.includes(value as TaskPhase);
}

function normalizeTaskState(value: Partial<TaskState> | undefined): TaskState {
  if (!value || typeof value !== "object") {
    return createTaskState();
  }

  const phase = isTaskPhase(value.phase) ? value.phase : "planning";
  const defaults = TASK_PHASE_DEFAULTS[phase];
  const isCancelled = phase === "done" && Boolean(value.isCancelled);
  const terminalDefaults =
    phase === "done" && isCancelled ? TASK_CANCELLED_DEFAULTS : defaults;
  const currentStep =
    phase === "done"
      ? terminalDefaults.currentStep
      : typeof value.currentStep === "string" && value.currentStep.trim()
        ? value.currentStep.trim()
        : defaults.currentStep;
  const expectedAction =
    phase === "done"
      ? terminalDefaults.expectedAction
      : typeof value.expectedAction === "string" && value.expectedAction.trim()
        ? value.expectedAction.trim()
        : defaults.expectedAction;
  const updatedAt =
    typeof value.updatedAt === "string" &&
    value.updatedAt.trim() &&
    !Number.isNaN(Date.parse(value.updatedAt))
      ? value.updatedAt
      : new Date().toISOString();

  return {
    phase,
    task: typeof value.task === "string" ? value.task : "",
    step:
      typeof value.step === "number" && Number.isFinite(value.step)
        ? Math.max(0, Math.floor(value.step))
        : Math.max(0, getTaskPhaseIndex(phase)),
    totalSteps:
      typeof value.totalSteps === "number" && Number.isFinite(value.totalSteps)
        ? Math.max(1, Math.floor(value.totalSteps))
        : TASK_PHASES.length,
    draftPlan: typeof value.draftPlan === "string" ? value.draftPlan : "",
    approvedPlan: typeof value.approvedPlan === "string" ? value.approvedPlan : "",
    solution: typeof value.solution === "string" ? value.solution : "",
    validationReport:
      typeof value.validationReport === "string" ? value.validationReport : "",
    violations: Array.isArray(value.violations)
      ? value.violations.filter((item): item is string => typeof item === "string")
      : [],
    done: Array.isArray(value.done)
      ? value.done.filter((item): item is string => typeof item === "string")
      : [],
    currentStep,
    expectedAction,
    isPaused: Boolean(value.isPaused),
    isCancelled,
    updatedAt
  };
}

function loadTaskState(): TaskState {
  try {
    const saved = localStorage.getItem(TASK_STATE_STORAGE_KEY);
    if (!saved) {
      return createTaskState();
    }

    return normalizeTaskState(JSON.parse(saved) as Partial<TaskState>);
  } catch {
    return createTaskState();
  }
}

function canTransitionTaskPhase(from: TaskPhase, to: TaskPhase): boolean {
  return TASK_ALLOWED_TRANSITIONS[from].includes(to);
}

function getTaskPhaseIndex(phase: TaskPhase): number {
  return TASK_PHASE_ORDER.indexOf(phase);
}

function moveTaskStateToPhase(
  state: TaskState,
  phase: TaskPhase,
  fields: Partial<TaskState> = {}
): TaskState {
  const defaults = TASK_PHASE_DEFAULTS[phase];

  return {
    ...state,
    ...fields,
    phase,
    step: Math.max(0, getTaskPhaseIndex(phase)),
    totalSteps: TASK_PHASES.length,
    isCancelled: fields.isCancelled ?? (phase === "done" ? state.isCancelled : false),
    currentStep: fields.currentStep ?? defaults.currentStep,
    expectedAction: fields.expectedAction ?? defaults.expectedAction,
    updatedAt: new Date().toISOString()
  };
}

function createCancelledTaskState(state: TaskState): TaskState {
  return moveTaskStateToPhase(state, "done", {
    isPaused: false,
    isCancelled: true,
    violations: [],
    done: ["Задача отменена пользователем"],
    currentStep: TASK_CANCELLED_DEFAULTS.currentStep,
    expectedAction: TASK_CANCELLED_DEFAULTS.expectedAction
  });
}

function getOptimisticTaskStateForAction(
  state: TaskState,
  action: OrchestratorAction,
  text: string
): TaskState | undefined {
  if (action === "cancelTask" && state.phase !== "done") {
    return createCancelledTaskState(state);
  }

  if (state.phase === "done" && action === "userMessage") {
    return {
      ...createTaskState("planning"),
      task: text
    };
  }

  if (state.isPaused || state.isCancelled) {
    return undefined;
  }

  if (state.phase === "planning" && action === "approvePlan") {
    return moveTaskStateToPhase(state, "execution", {
      approvedPlan: state.draftPlan.trim() ? state.draftPlan : state.approvedPlan
    });
  }

  if (state.phase === "execution" && action === "approveSolution") {
    return moveTaskStateToPhase(state, "validation");
  }

  if (action === "disputeSolution" && state.phase === "execution") {
    return moveTaskStateToPhase(state, "planning", {
      currentStep: "Пересобрать план с учетом замечаний пользователя.",
      expectedAction: "Пользователь вносит правки, Planning Agent обновляет план."
    });
  }

  if (action === "disputeSolution" && state.phase === "validation") {
    return moveTaskStateToPhase(state, "execution", {
      currentStep: "Доработать решение после замечания пользователя.",
      expectedAction: "Execution Agent исправляет решение и затем его можно снова валидировать."
    });
  }

  return undefined;
}

function App() {
  const initialChats = useMemo(() => loadChats(), []);
  const initialUserProfiles = useMemo(() => loadUserProfiles(), []);
  const [chats, setChats] = useState<ChatSession[]>(initialChats);
  const [activeChatId, setActiveChatId] = useState(initialChats[0]?.id ?? "");
  const [input, setInput] = useState("");
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());
  const [userProfiles, setUserProfiles] =
    useState<StoredUserProfile[]>(initialUserProfiles);
  const [activeProfileId, setActiveProfileId] = useState(() =>
    loadActiveProfileId(initialUserProfiles)
  );
  const [workingMemory, setWorkingMemory] = useState<MemoryItem[]>(() =>
    loadMemoryItems(WORKING_MEMORY_STORAGE_KEY)
  );
  const [taskState, setTaskState] = useState<TaskState>(() => loadTaskState());
  const [lastMemoryWrites, setLastMemoryWrites] = useState<MemoryWrite[]>([]);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isModelPickerOpen, setIsModelPickerOpen] = useState(false);
  const [isProfilePickerOpen, setIsProfilePickerOpen] = useState(false);
  const [isChatSidebarCollapsed, setIsChatSidebarCollapsed] = useState(false);
  const [isMemoryPanelCollapsed, setIsMemoryPanelCollapsed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [agentPhase, setAgentPhase] = useState<AgentPhase>("idle");
  const [streamingContent, setStreamingContent] = useState("");
  const [mcpExecutionSteps, setMcpExecutionSteps] = useState<McpExecutionStep[]>([]);
  const [swarmActors, setSwarmActors] = useState<string[]>([]);
  const [activeSwarmActor, setActiveSwarmActor] = useState<string | null>(null);
  const [swarmStatus, setSwarmStatus] = useState("");
  const [swarmActorLogs, setSwarmActorLogs] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [lastReply, setLastReply] = useState<AgentReply | null>(null);
  const [testingMcpServerId, setTestingMcpServerId] = useState<string | null>(null);
  const [mcpConnectionTests, setMcpConnectionTests] = useState<
    Record<string, McpConnectionTestResult>
  >({});
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const activeRequestIdRef = useRef<string | null>(null);
  const cancelledRequestIdsRef = useRef<Set<string>>(new Set());
  const profileSwitcherRef = useRef<HTMLDivElement | null>(null);

  const activeChat = chats.find((chat) => chat.id === activeChatId) ?? chats[0];
  const activeProfile =
    userProfiles.find((profile) => profile.id === activeProfileId) ?? userProfiles[0];
  const activeProfileForRequest = useMemo<UserProfile | undefined>(
    () =>
      activeProfile
        ? {
            id: activeProfile.id,
            name: activeProfile.name,
            style: activeProfile.style,
            format: activeProfile.format,
            constraints: activeProfile.constraints,
            context: activeProfile.context,
            createdAt: activeProfile.createdAt,
            updatedAt: activeProfile.updatedAt
          }
        : undefined,
    [activeProfile]
  );
  const activeProfileName = activeProfile?.name.trim() || "Без профиля";
  const longTermMemory = activeProfile?.longTermMemory ?? [];
  const activeModelOption = MODEL_OPTIONS.find((model) => model.id === settings.model);
  const activeModelLabel = activeModelOption?.label ?? settings.model;
  const messages = activeChat?.messages ?? [];
  const hasMessages = messages.length > 0;
  const shortTermSummary =
    settings.shortTermCompressionEnabled ? activeChat?.shortTermSummary : undefined;
  const shortTermCompression: ShortTermCompressionSettings = useMemo(
    () => ({
      enabled: settings.shortTermCompressionEnabled,
      maxUncompressedTurns: settings.shortTermCompressionTurnLimit
    }),
    [settings.shortTermCompressionEnabled, settings.shortTermCompressionTurnLimit]
  );

  const memoryContext = useMemo<MemoryContext>(
    () => ({
      activeProfile: activeProfileForRequest,
      shortTerm: messages,
      shortTermSummary,
      working: workingMemory,
      longTerm: longTermMemory,
      taskState
    }),
    [
      activeProfileForRequest,
      messages,
      shortTermSummary,
      workingMemory,
      longTermMemory,
      taskState
    ]
  );

  const shortTermTurnCount = useMemo(
    () => messages.filter((message) => message.role === "user").length,
    [messages]
  );

  const estimatedChars = useMemo(
    () => messages.reduce((total, message) => total + message.content.length, 0),
    [messages]
  );
  const taskPhaseIndex = getTaskPhaseIndex(taskState.phase);
  const taskPhaseLabel =
    TASK_PHASES.find((phase) => phase.id === taskState.phase)?.label ?? taskState.phase;
  const nextTaskPhase = TASK_PHASE_ORDER[taskPhaseIndex + 1];
  const previousTaskPhase = TASK_PHASE_ORDER[taskPhaseIndex - 1];
  const canAdvanceTask =
    Boolean(nextTaskPhase) &&
    !taskState.isPaused &&
    !taskState.isCancelled &&
    canTransitionTaskPhase(taskState.phase, nextTaskPhase);
  const canRewindTask =
    Boolean(previousTaskPhase) &&
    !taskState.isPaused &&
    !taskState.isCancelled &&
    canTransitionTaskPhase(taskState.phase, previousTaskPhase);
  const canApprovePlan =
    settings.orchestrationEnabled &&
    taskState.phase === "planning" &&
    Boolean(taskState.draftPlan.trim()) &&
    !taskState.isCancelled &&
    !isLoading;
  const canApproveSolution =
    settings.orchestrationEnabled &&
    taskState.phase === "execution" &&
    Boolean(taskState.solution.trim()) &&
    !taskState.isCancelled &&
    !isLoading;
  const canDisputeSolution =
    settings.orchestrationEnabled &&
    taskState.phase === "execution" &&
    !taskState.isCancelled &&
    !isLoading;
  const canCancelTask =
    settings.orchestrationEnabled &&
    taskState.phase !== "done" &&
    !taskState.isCancelled;
  const hasSwarmActivity =
    settings.orchestrationEnabled &&
    swarmActors.length > 0 &&
    (isLoading || taskState.phase === "execution" || taskState.phase === "validation");
  const hasSwarmLogs = hasAnySwarmLog(swarmActorLogs);
  const enabledMcpServers = settings.mcpServers.filter((server) => server.enabled);

  useEffect(() => {
    localStorage.setItem(CHATS_STORAGE_KEY, JSON.stringify(chats));
  }, [chats]);

  useEffect(() => {
    localStorage.setItem(WORKING_MEMORY_STORAGE_KEY, JSON.stringify(workingMemory));
  }, [workingMemory]);

  useEffect(() => {
    localStorage.setItem(TASK_STATE_STORAGE_KEY, JSON.stringify(taskState));
  }, [taskState]);

  useEffect(() => {
    localStorage.setItem(USER_PROFILES_STORAGE_KEY, JSON.stringify(userProfiles));
  }, [userProfiles]);

  useEffect(() => {
    localStorage.setItem(ACTIVE_PROFILE_STORAGE_KEY, activeProfileId);
  }, [activeProfileId]);

  useEffect(() => {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
    document.documentElement.dataset.theme = settings.theme;
  }, [settings]);

  useEffect(() => {
    if (!settings.autoScroll) {
      return;
    }

    const messagesElement = messagesRef.current;
    if (!messagesElement) {
      return;
    }

    requestAnimationFrame(() => {
      messagesElement.scrollTo({
        top: messagesElement.scrollHeight,
        behavior: isLoading ? "auto" : "smooth"
      });
    });
  }, [
    activeChatId,
    messages.length,
    streamingContent,
    mcpExecutionSteps,
    swarmActorLogs,
    agentPhase,
    isLoading,
    settings.autoScroll
  ]);

  useEffect(() => {
    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsSettingsOpen(false);
        setIsModelPickerOpen(false);
        setIsProfilePickerOpen(false);
      }
    }

    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, []);

  useEffect(() => {
    if (!isProfilePickerOpen) {
      return;
    }

    function handlePointerDown(event: PointerEvent) {
      const target = event.target;
      if (
        target instanceof Node &&
        profileSwitcherRef.current?.contains(target)
      ) {
        return;
      }

      setIsProfilePickerOpen(false);
    }

    window.addEventListener("pointerdown", handlePointerDown);
    return () => window.removeEventListener("pointerdown", handlePointerDown);
  }, [isProfilePickerOpen]);

  function updateSettings(nextSettings: Partial<AppSettings>) {
    setSettings((current) => ({
      ...current,
      ...nextSettings,
      shortTermCompressionTurnLimit:
        nextSettings.shortTermCompressionTurnLimit === undefined
          ? current.shortTermCompressionTurnLimit
          : normalizeCompressionTurnLimit(nextSettings.shortTermCompressionTurnLimit)
    }));
  }

  function updateMcpServer(serverId: string, patch: Partial<McpServerConfig>) {
    setSettings((current) => ({
      ...current,
      mcpServers: current.mcpServers.map((server) =>
        server.id === serverId ? { ...server, ...patch } : server
      )
    }));
    setMcpConnectionTests((current) => {
      const next = { ...current };
      delete next[serverId];
      return next;
    });
  }

  function addMcpServer(transport: McpServerConfig["transport"]) {
    setSettings((current) => ({
      ...current,
      mcpServers: [
        ...current.mcpServers,
        createMcpServer(current.mcpServers.length + 1, transport)
      ]
    }));
  }

  function addFigmaMcpServer() {
    setSettings((current) => ({
      ...current,
      mcpServers: [
        ...current.mcpServers,
        {
          ...createMcpServer(current.mcpServers.length + 1, "streamableHttp"),
          name: "Figma Remote",
          url: "https://mcp.figma.com/mcp"
        }
      ]
    }));
  }

  function addYandexTrackerMcpServer() {
    setSettings((current) => {
      let suffix = 1;
      let id = "yandex-tracker";
      while (current.mcpServers.some((server) => server.id === id)) {
        suffix += 1;
        id = `yandex-tracker-${suffix}`;
      }

      return {
        ...current,
        mcpServers: [
          ...current.mcpServers,
          {
            ...createMcpServer(current.mcpServers.length + 1, "streamableHttp"),
            id,
            name: "Yandex Tracker",
            enabled: false,
            url: "http://127.0.0.1:8788/mcp",
            headers: { Authorization: "Bearer change-me" }
          }
        ]
      };
    });
  }

  function addTelegramDeliveryMcpServer() {
    setSettings((current) => {
      let suffix = 1;
      let id = "telegram-delivery";
      while (current.mcpServers.some((server) => server.id === id)) {
        suffix += 1;
        id = `telegram-delivery-${suffix}`;
      }

      return {
        ...current,
        mcpServers: [
          ...current.mcpServers,
          {
            ...createMcpServer(current.mcpServers.length + 1, "streamableHttp"),
            id,
            name: "Telegram Delivery",
            enabled: false,
            url: "http://127.0.0.1:8792/mcp",
            headers: { Authorization: "Bearer change-me" }
          }
        ]
      };
    });
  }

  function removeMcpServer(serverId: string) {
    setSettings((current) => ({
      ...current,
      mcpServers: current.mcpServers.filter((server) => server.id !== serverId)
    }));
    setMcpConnectionTests((current) => {
      const next = { ...current };
      delete next[serverId];
      return next;
    });
  }

  function updateMcpEnvironmentVariable(
    server: McpServerConfig,
    previousKey: string,
    key: string,
    value: string
  ) {
    const env = { ...server.env };
    if (previousKey !== key) {
      delete env[previousKey];
    }
    if (key.trim()) {
      env[key] = value;
    }
    updateMcpServer(server.id, { env });
  }

  function addMcpEnvironmentVariable(server: McpServerConfig) {
    let index = Object.keys(server.env).length + 1;
    let key = `VARIABLE_${index}`;
    while (key in server.env) {
      index += 1;
      key = `VARIABLE_${index}`;
    }
    updateMcpServer(server.id, { env: { ...server.env, [key]: "" } });
  }

  function updateMcpHeader(
    server: McpServerConfig,
    previousKey: string,
    key: string,
    value: string
  ) {
    const headers = { ...server.headers };
    if (previousKey !== key) {
      delete headers[previousKey];
    }
    if (key.trim()) {
      headers[key] = value;
    }
    updateMcpServer(server.id, { headers });
  }

  function addMcpHeader(server: McpServerConfig) {
    let index = Object.keys(server.headers).length + 1;
    let key = index === 1 ? "Authorization" : `X-Header-${index}`;
    while (key in server.headers) {
      index += 1;
      key = `X-Header-${index}`;
    }
    updateMcpServer(server.id, { headers: { ...server.headers, [key]: "" } });
  }

  async function testMcpConnection(server: McpServerConfig) {
    setTestingMcpServerId(server.id);
    setMcpConnectionTests((current) => {
      const next = { ...current };
      delete next[server.id];
      return next;
    });

    try {
      const result = await invoke<McpConnectionTestResult>("test_mcp_connection", {
        server
      });
      setMcpConnectionTests((current) => ({ ...current, [server.id]: result }));
    } catch (caughtError) {
      setMcpConnectionTests((current) => ({
        ...current,
        [server.id]: {
          serverId: server.id,
          serverName: server.name,
          connected: false,
          status:
            caughtError instanceof Error ? caughtError.message : String(caughtError),
          tools: []
        }
      }));
    } finally {
      setTestingMcpServerId(null);
    }
  }

  function selectUserProfile(profileId: string) {
    if (isLoading || !userProfiles.some((profile) => profile.id === profileId)) {
      return;
    }

    setActiveProfileId(profileId);
    setIsProfilePickerOpen(false);
    setLastReply(null);
    setLastMemoryWrites([]);
    inputRef.current?.focus();
  }

  function createUserProfile() {
    if (isLoading) {
      return;
    }

    const profile = createBlankUserProfile(userProfiles.length + 1);
    setUserProfiles((current) => [profile, ...current]);
    setActiveProfileId(profile.id);
    setIsProfilePickerOpen(false);
    setLastReply(null);
    setLastMemoryWrites([
      createMemoryWrite(
        "saved",
        "userProfile",
        profile.name,
        "Создан новый профиль пользователя"
      )
    ]);
    inputRef.current?.focus();
  }

  function deleteUserProfile(profileId: string) {
    if (isLoading || userProfiles.length <= 1) {
      return;
    }

    const profileToDelete = userProfiles.find((profile) => profile.id === profileId);
    const nextProfiles = userProfiles.filter((profile) => profile.id !== profileId);
    const safeProfiles = nextProfiles.length > 0 ? nextProfiles : [createDefaultUserProfile()];
    const nextActiveProfileId =
      profileId === activeProfileId ? safeProfiles[0].id : activeProfileId;

    setUserProfiles(safeProfiles);
    setActiveProfileId(nextActiveProfileId);
    setIsProfilePickerOpen(false);
    setLastReply(null);
    setLastMemoryWrites([
      createMemoryWrite(
        "deleted",
        "userProfile",
        profileToDelete?.name ?? "Удаленный профиль",
        "Профиль пользователя удален вместе с его долгосрочной памятью"
      )
    ]);
    inputRef.current?.focus();
  }

  function updateActiveUserProfile(
    nextProfile: Partial<Pick<UserProfile, "name" | "style" | "format" | "constraints" | "context">>
  ) {
    if (!activeProfile) {
      return;
    }

    const now = new Date().toISOString();
    setUserProfiles((current) =>
      current.map((profile) =>
        profile.id === activeProfile.id
          ? {
              ...profile,
              ...nextProfile,
              name:
                nextProfile.name === undefined
                  ? profile.name
                  : nextProfile.name || "Без имени",
              updatedAt: now
            }
          : profile
      )
    );
  }

  function updateActiveChat(
    messagesForChat: ChatMessage[],
    nextShortTermSummary?: ShortTermSummary | null
  ) {
    const now = new Date().toISOString();

    setChats((currentChats) =>
      currentChats.map((chat) =>
        chat.id === activeChatId
          ? {
              ...chat,
              title: buildChatTitle(messagesForChat),
              messages: messagesForChat,
              shortTermSummary:
                nextShortTermSummary === undefined
                  ? chat.shortTermSummary
                  : nextShortTermSummary === null
                    ? undefined
                    : nextShortTermSummary,
              updatedAt: now
            }
          : chat
      )
    );
  }

  function createNewChat() {
    const chat = createEmptyChat();
    setChats((currentChats) => [chat, ...currentChats]);
    setActiveChatId(chat.id);
    setInput("");
    setError(null);
    setLastReply(null);
    setAgentPhase("idle");
    setStreamingContent("");
    setLastMemoryWrites([]);
    inputRef.current?.focus();
  }

  function clearChat() {
    updateActiveChat([], null);
    setLastReply(null);
    setAgentPhase("idle");
    setStreamingContent("");
    setLastMemoryWrites([
      createMemoryWrite(
        "deleted",
        "shortTerm",
        "История текущего чата очищена",
        "Краткосрочная память очищена только для выбранного чата"
      )
    ]);
    setError(null);
    inputRef.current?.focus();
  }

  function selectChat(chatId: string) {
    if (isLoading) {
      return;
    }

    setActiveChatId(chatId);
    setInput("");
    setError(null);
    setLastReply(null);
    setAgentPhase("idle");
    setStreamingContent("");
    setLastMemoryWrites([]);
    inputRef.current?.focus();
  }

  function deleteChat(chatId: string) {
    if (isLoading) {
      return;
    }

    const chatToDelete = chats.find((chat) => chat.id === chatId);
    const nextChats = chats.filter((chat) => chat.id !== chatId);
    const safeChats = nextChats.length > 0 ? nextChats : [createEmptyChat()];
    const isDeletingActiveChat = activeChatId === chatId;
    const nextActiveChatId =
      isDeletingActiveChat || !safeChats.some((chat) => chat.id === activeChatId)
        ? safeChats[0].id
        : activeChatId;

    setChats(safeChats);
    setActiveChatId(nextActiveChatId);

    if (isDeletingActiveChat) {
      setInput("");
      setError(null);
      setLastReply(null);
      setAgentPhase("idle");
      setStreamingContent("");
    }

    setLastMemoryWrites([
      createMemoryWrite(
        "deleted",
        "shortTerm",
        chatToDelete?.title ?? "Удаленный чат",
        "Чат удален из истории; его краткосрочная память удалена вместе с ним"
      )
    ]);
    inputRef.current?.focus();
  }

  function removePersistentMemory(layer: PersistentMemoryLayer, item: MemoryItem) {
    if (layer === "working") {
      setWorkingMemory((current) =>
        current.filter((memoryItem) => memoryItem.id !== item.id)
      );
    } else if (activeProfile) {
      const now = new Date().toISOString();
      setUserProfiles((current) =>
        current.map((profile) =>
          profile.id === activeProfile.id
            ? {
                ...profile,
                longTermMemory: profile.longTermMemory.filter(
                  (memoryItem) => memoryItem.id !== item.id
                ),
                updatedAt: now
              }
            : profile
        )
      );
    }

    setLastMemoryWrites([
      createMemoryWrite("deleted", layer, item.content, "Удалено вручную", item.id)
    ]);
  }

  function applyMemoryDecisions(
    decisions: MemoryDecision[] | undefined,
    sourceMessage: string,
    sourceChatId: string
  ): MemoryWrite[] {
    if (!decisions || decisions.length === 0) {
      return [
        createMemoryWrite(
          "skipped",
          "working",
          sourceMessage,
          "OpenAI memory-router не вернул решений"
        ),
        createMemoryWrite(
          "skipped",
          "longTerm",
          sourceMessage,
          "OpenAI memory-router не вернул решений"
        )
      ];
    }

    let nextWorkingMemory = workingMemory;
    let nextLongTermMemory = longTermMemory;
    const writes: MemoryWrite[] = [];

    for (const decision of decisions) {
      const layer = decision.layer;
      const memoryText = decision.memoryText.trim();
      const reason = `OpenAI LLM: ${decision.reason || "без причины"}`;

      if (decision.action !== "save" || !memoryText) {
        writes.push(createMemoryWrite("skipped", layer, sourceMessage, reason));
        continue;
      }

      const currentLayerMemory =
        layer === "working" ? nextWorkingMemory : nextLongTermMemory;

      if (hasSimilarMemory(currentLayerMemory, memoryText)) {
        writes.push(
          createMemoryWrite(
            "skipped",
            layer,
            memoryText,
            `${reason}; похожая запись уже есть`
          )
        );
        continue;
      }

      const item = createMemoryItem(memoryText, {
        chatId: sourceChatId,
        message: sourceMessage
      });

      if (layer === "working") {
        nextWorkingMemory = [item, ...nextWorkingMemory];
      } else {
        nextLongTermMemory = [item, ...nextLongTermMemory];
      }

      writes.push(createMemoryWrite("saved", layer, item.content, reason, item.id));
    }

    setWorkingMemory(nextWorkingMemory);
    if (activeProfile) {
      const now = new Date().toISOString();
      setUserProfiles((current) =>
        current.map((profile) =>
          profile.id === activeProfile.id
            ? {
                ...profile,
                longTermMemory: nextLongTermMemory,
                updatedAt: now
              }
            : profile
        )
      );
    }

    return writes;
  }

  async function submitCurrentInput(
    orchestratorAction: OrchestratorAction = "userMessage",
    actionText?: string
  ) {
    const text = (actionText ?? input).trim();
    if (!text || isLoading || !activeChat) {
      return;
    }

    const requestId = crypto.randomUUID();
    activeRequestIdRef.current = requestId;
    cancelledRequestIdsRef.current.delete(requestId);
    let requestSwarmActors: string[] = [];
    let requestSwarmStatus = "";
    const requestSwarmLogs: Record<string, string> = {};
    let requestMcpSteps: McpExecutionStep[] = [];
    const previousMessages = activeChat.messages;
    const requestTaskState = taskState;
    const optimisticTaskState = settings.orchestrationEnabled
      ? getOptimisticTaskStateForAction(requestTaskState, orchestratorAction, text)
      : undefined;
    const nextMessages: ChatMessage[] = [
      ...previousMessages,
      { role: "user", content: text }
    ];

    const memoryWrites: MemoryWrite[] = [
      createMemoryWrite(
        "saved",
        "shortTerm",
        text,
        "Сообщение добавлено в краткосрочную память текущего чата"
      )
    ];

    updateActiveChat(nextMessages);
    setInput("");
    setError(null);
    setLastReply(null);
    setLastMemoryWrites(memoryWrites);
    setIsLoading(true);
    if (optimisticTaskState) {
      setTaskState(optimisticTaskState);
    }
    setAgentPhase(shouldCompressShortTerm(nextMessages, settings) ? "compressing" : "streaming");
    setStreamingContent("");
    setMcpExecutionSteps([]);
    setSwarmActors([]);
    setActiveSwarmActor(null);
    setSwarmStatus("");
    setSwarmActorLogs({});

    let unlistenStream: (() => void) | undefined;
    let unlistenMemory: (() => void) | undefined;
    let unlistenSwarm: (() => void) | undefined;

    try {
      unlistenStream = await listen<AgentStreamDelta>(
        "agent_stream_delta",
        (event) => {
          if (
            event.payload.requestId !== requestId ||
            cancelledRequestIdsRef.current.has(requestId)
          ) {
            return;
          }

          if (event.payload.channel === "mcp") {
            const toolName = event.payload.actor ?? "mcp_tool";
            const serverName = event.payload.serverName ?? "MCP";
            const status = ["running", "completed", "failed"].includes(event.payload.delta)
              ? event.payload.delta as McpExecutionStep["status"]
              : "running";
            requestMcpSteps = updateMcpExecutionSteps(
              requestMcpSteps,
              serverName,
              toolName,
              status
            );
            setMcpExecutionSteps(requestMcpSteps);
            setAgentPhase("streaming");
            return;
          }

          if ((event.payload.channel ?? "final") === "swarm") {
            const actor = event.payload.actor ?? "SWARM";
            requestSwarmLogs[actor] = `${requestSwarmLogs[actor] ?? ""}${event.payload.delta}`;
            setSwarmActorLogs((current) => ({
              ...current,
              [actor]: `${current[actor] ?? ""}${event.payload.delta}`
            }));
          } else {
            setStreamingContent((current) => current + event.payload.delta);
          }
          setAgentPhase((current) => (current === "compressing" ? "streaming" : current));
        }
      );

      unlistenMemory = await listen<AgentMemoryStarted>(
        "agent_memory_started",
        (event) => {
          if (
            event.payload.requestId === requestId &&
            !cancelledRequestIdsRef.current.has(requestId)
          ) {
            setAgentPhase("memory");
          }
        }
      );

      unlistenSwarm = await listen<AgentSwarmStatus>(
        "agent_swarm_status",
        (event) => {
          if (
            event.payload.requestId !== requestId ||
            cancelledRequestIdsRef.current.has(requestId)
          ) {
            return;
          }

          requestSwarmActors = event.payload.actors;
          requestSwarmStatus = event.payload.status;
          setSwarmActors(event.payload.actors);
          setActiveSwarmActor(event.payload.activeActor ?? null);
          setSwarmStatus(event.payload.status);
        }
      );

      const requestMemoryContext: MemoryContext = {
        activeProfile: activeProfileForRequest,
        shortTerm: nextMessages,
        shortTermSummary,
        working: workingMemory,
        longTerm: longTermMemory,
        taskState: requestTaskState
      };

      const reply = await invoke<AgentReply>("send_agent_message", {
        request: {
          requestId,
          apiKey: settings.apiKey.trim() || undefined,
          model: settings.model.trim(),
          systemPrompt: settings.systemPrompt,
          messages: nextMessages,
          memoryContext: requestMemoryContext,
          shortTermCompression,
          orchestration: {
            enabled: settings.orchestrationEnabled,
            action: orchestratorAction,
            validatorInvariants: settings.validatorInvariants
          },
          mcp: {
            servers: settings.mcpServers
          }
        }
      });

      if (cancelledRequestIdsRef.current.has(requestId)) {
        return;
      }

      const swarmMessage = createSwarmDiscussionMessage(
        requestSwarmActors,
        requestSwarmLogs,
        requestSwarmStatus
      );
      const completedMessages: ChatMessage[] = [
        ...nextMessages,
        ...(swarmMessage ? [swarmMessage] : []),
        {
          role: "assistant",
          content: reply.content,
          mcpSteps: requestMcpSteps.length > 0
              ? requestMcpSteps
            : reply.debug?.mcpToolCalls?.map((call) => ({
                serverName: call.serverName,
                toolName: call.toolName,
                label: getMcpToolLabel(call.toolName),
                status: call.isError ? "failed" as const : "completed" as const
              }))
        }
      ];
      const nextShortTermSummary = reply.shortTermSummary
        ? {
            ...reply.shortTermSummary,
            updatedAt: new Date().toISOString()
          }
        : undefined;

      setLastReply(reply);
      if (reply.taskState) {
        setTaskState(normalizeTaskState(reply.taskState));
      }
      updateActiveChat(completedMessages, nextShortTermSummary);
      setLastMemoryWrites([
        ...memoryWrites,
        ...applyMemoryDecisions(reply.memoryDecisions, text, activeChat.id)
      ]);
      setStreamingContent("");
      setMcpExecutionSteps([]);
    } catch (caughtError) {
      if (
        cancelledRequestIdsRef.current.has(requestId) ||
        activeRequestIdRef.current !== requestId
      ) {
        return;
      }

      const message =
        caughtError instanceof Error ? caughtError.message : String(caughtError);
      setError(message);
      updateActiveChat(previousMessages);
      if (optimisticTaskState) {
        setTaskState(requestTaskState);
      }
      setStreamingContent("");
      setMcpExecutionSteps([]);
    } finally {
      unlistenStream?.();
      unlistenMemory?.();
      unlistenSwarm?.();
      cancelledRequestIdsRef.current.delete(requestId);
      if (activeRequestIdRef.current === requestId) {
        activeRequestIdRef.current = null;
        setIsLoading(false);
        setAgentPhase("idle");
        setMcpExecutionSteps([]);
        inputRef.current?.focus();
      }
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    void submitCurrentInput();
  }

  function submitOrchestratorAction(
    action: OrchestratorAction,
    actionText: string
  ) {
    void submitCurrentInput(action, actionText);
  }

  function cancelTask() {
    if (!canCancelTask || !activeChat) {
      return;
    }

    if (!isLoading) {
      submitOrchestratorAction("cancelTask", "🛑 Отменить текущую задачу");
      return;
    }

    const requestId = activeRequestIdRef.current;
    if (requestId) {
      cancelledRequestIdsRef.current.add(requestId);
      void invoke("cancel_agent_message", { requestId }).catch(() => undefined);
      activeRequestIdRef.current = null;
    }

    setTaskState((current) => createCancelledTaskState(current));
    setStreamingContent("");
    setSwarmActors([]);
    setActiveSwarmActor(null);
    setSwarmStatus("");
    setSwarmActorLogs({});
    setLastReply(null);
    setError(null);
    setIsLoading(false);
    setAgentPhase("idle");

    const alreadyCancelled =
      activeChat.messages[activeChat.messages.length - 1]?.role === "assistant" &&
      activeChat.messages[activeChat.messages.length - 1]?.content === TASK_CANCEL_MESSAGE;

    if (!alreadyCancelled) {
      updateActiveChat([
        ...activeChat.messages,
        { role: "assistant", content: TASK_CANCEL_MESSAGE }
      ]);
    }

    inputRef.current?.focus();
  }

  function selectModel(modelId: string) {
    updateSettings({ model: modelId });
    setIsModelPickerOpen(false);
    inputRef.current?.focus();
  }

  function transitionTaskPhase(nextPhase: TaskPhase) {
    setTaskState((current) => {
      if (current.phase === nextPhase || !canTransitionTaskPhase(current.phase, nextPhase)) {
        return current;
      }

      return createTaskState(nextPhase);
    });
  }

  function advanceTaskPhase() {
    if (nextTaskPhase && canTransitionTaskPhase(taskState.phase, nextTaskPhase)) {
      transitionTaskPhase(nextTaskPhase);
    }
  }

  function rewindTaskPhase() {
    if (previousTaskPhase && canTransitionTaskPhase(taskState.phase, previousTaskPhase)) {
      transitionTaskPhase(previousTaskPhase);
    }
  }

  function pauseTask() {
    setTaskState((current) => ({
      ...current,
      isPaused: true,
      updatedAt: new Date().toISOString()
    }));
  }

  function resumeTask() {
    setTaskState((current) => ({
      ...current,
      isPaused: false,
      updatedAt: new Date().toISOString()
    }));
  }

  function resetTaskState() {
    setTaskState(createTaskState());
  }

  function updateTaskStateText(
    nextFields: Partial<Pick<TaskState, "currentStep" | "expectedAction">>
  ) {
    setTaskState((current) => ({
      ...current,
      ...nextFields,
      updatedAt: new Date().toISOString()
    }));
  }

  return (
    <main
      className={`app-shell ${isChatSidebarCollapsed ? "history-collapsed" : ""} ${
        isMemoryPanelCollapsed ? "memory-collapsed" : ""
      }`}
    >
      <aside className={`chat-sidebar ${isChatSidebarCollapsed ? "collapsed" : ""}`}>
        {isChatSidebarCollapsed ? (
          <button
            className="collapsed-panel-button"
            type="button"
            onClick={() => setIsChatSidebarCollapsed(false)}
            aria-label="Развернуть историю чатов"
            title="Развернуть историю чатов"
          >
            <ButtonIcon src={ICONS.arrowRight} />
            <b>Чаты</b>
          </button>
        ) : (
          <>
            <div className="sidebar-header">
              <div>
                <p className="eyebrow">История</p>
                <h2>Чаты</h2>
              </div>
              <div className="sidebar-actions">
                <button
                  className="collapse-panel-button"
                  type="button"
                  onClick={() => setIsChatSidebarCollapsed(true)}
                  aria-label="Свернуть историю чатов"
                  title="Свернуть историю чатов"
                >
                  <ButtonIcon src={ICONS.arrowLeft} />
                </button>
                <button className="new-chat-button" type="button" onClick={createNewChat}>
                  <ButtonIcon src={ICONS.plus} />
                  Новый
                </button>
              </div>
            </div>

            <div className="chat-list" aria-label="История чатов">
              {chats.map((chat) => (
                <div
                  className={`chat-list-item ${chat.id === activeChatId ? "active" : ""}`}
                  key={chat.id}
                >
                  <button
                    className="chat-select-button"
                    type="button"
                    onClick={() => selectChat(chat.id)}
                  >
                    <span>{chat.title}</span>
                    <small>
                      {chat.messages.length} сообщ. · {formatChatTime(chat.updatedAt)}
                    </small>
                  </button>
                  <button
                    className="delete-chat-button"
                    type="button"
                    onClick={() => deleteChat(chat.id)}
                    aria-label={`Удалить чат ${chat.title}`}
                    title="Удалить чат"
                  >
                    <ButtonIcon src={ICONS.cross} />
                  </button>
                </div>
              ))}
            </div>
          </>
        )}
      </aside>

      <section className="chat-panel">
        <header className="top-bar">
          <div>
            <p className="eyebrow">Desktop AI Agent</p>
            <h1>{activeChat?.title ?? "Chatbot AI"}</h1>
          </div>
          <div className="top-actions">
            <div
              className="profile-switcher"
              ref={profileSwitcherRef}
              aria-label="Активный профиль пользователя"
            >
              <button
                className="profile-current-button"
                type="button"
                onClick={() => setIsProfilePickerOpen((current) => !current)}
                disabled={isLoading}
                aria-haspopup="listbox"
                aria-expanded={isProfilePickerOpen}
                title="Активный профиль"
              >
                <span>{activeProfileName}</span>
                <ButtonIcon className="button-icon-small" src={ICONS.arrowDown} />
              </button>
              <button
                className="profile-action-button"
                type="button"
                onClick={createUserProfile}
                disabled={isLoading}
                aria-label="Создать профиль пользователя"
                title="Создать профиль"
              >
                <ButtonIcon src={ICONS.plus} />
              </button>
              <button
                className="profile-action-button danger"
                type="button"
                onClick={() => activeProfile && deleteUserProfile(activeProfile.id)}
                disabled={isLoading || userProfiles.length <= 1}
                aria-label="Удалить активный профиль пользователя"
                title="Удалить активный профиль"
              >
                <ButtonIcon src={ICONS.cross} />
              </button>
              {isProfilePickerOpen && (
                <div className="profile-dropdown" role="listbox">
                  {userProfiles.map((profile) => (
                    <button
                      className={profile.id === activeProfile?.id ? "active" : ""}
                      key={profile.id}
                      type="button"
                      onClick={() => selectUserProfile(profile.id)}
                      role="option"
                      aria-selected={profile.id === activeProfile?.id}
                    >
                      <strong>{profile.name || "Без имени"}</strong>
                      <small>
                        {profile.longTermMemory.length} долгоср. · обновлено{" "}
                        {formatChatTime(profile.updatedAt)}
                      </small>
                    </button>
                  ))}
                </div>
              )}
            </div>
            <button className="ghost-button" type="button" onClick={clearChat}>
              Очистить чат
            </button>
            <button
              className="settings-button"
              type="button"
              onClick={() => setIsSettingsOpen(true)}
              aria-label="Открыть настройки"
            >
              <ButtonIcon src={ICONS.settings} />
              Настройки
            </button>
          </div>
        </header>

        <div className="status-strip">
          <span>profile: {activeProfileName}</span>
          {settings.orchestrationEnabled && (
            <span>
              task: {taskPhaseLabel}
              {taskState.isCancelled ? " · cancelled" : taskState.isPaused ? " · paused" : ""}
            </span>
          )}
          <span>{messages.length} сообщений</span>
          <span>{estimatedChars} символов</span>
          <span>{lastReply?.model ?? settings.model}</span>
          <span>tokens: {lastReply?.usage?.totalTokens ?? "-"}</span>
          <span>turns: {shortTermTurnCount}</span>
          <span>short-term: {memoryContext.shortTerm.length}</span>
          <span>
            summary: {shortTermSummary?.compressedTurnCount ?? 0}/
            {settings.shortTermCompressionTurnLimit}
          </span>
          <span>working: {memoryContext.working.length}</span>
          <span>long-term: {memoryContext.longTerm.length}</span>
          <span>
            MCP:{" "}
            {enabledMcpServers.length > 0
              ? lastReply?.debug?.mcpStatus ?? `${enabledMcpServers.length} on`
              : "off"}
          </span>
        </div>

        <div className="messages" aria-live="polite" ref={messagesRef}>
          {settings.orchestrationEnabled && (
            <div
              className={`task-stage-badge ${taskState.isPaused ? "paused" : ""} ${
                taskState.isCancelled ? "cancelled" : ""
              }`}
              aria-label="Этапы текущей задачи"
            >
              {TASK_PHASES.map((phase, index) => {
                const isActive = phase.id === taskState.phase;
                const isCompleted = index < taskPhaseIndex;

                return (
                  <span className="task-stage-group" key={phase.id}>
                    <span
                      className={`task-stage-chip ${isActive ? "active" : ""} ${
                        isCompleted ? "completed" : ""
                      } ${index > taskPhaseIndex ? "upcoming" : ""}`}
                      title={`${phase.label}: ${phase.description}`}
                    >
                      {phase.id}
                    </span>
                    {index < TASK_PHASES.length - 1 && (
                      <span
                        className={`task-stage-arrow ${
                          index < taskPhaseIndex ? "completed" : ""
                        }`}
                        aria-hidden="true"
                      >
                        →
                      </span>
                    )}
                  </span>
                );
              })}
              {taskState.isPaused && <b className="task-paused-pill">Пауза</b>}
              {taskState.isCancelled && <b className="task-cancelled-pill">Отменено</b>}
            </div>
          )}

          {hasSwarmActivity && (
            <div className="task-swarm-strip" aria-label="Акторы execution swarm">
              <span className="task-swarm-status">{swarmStatus || "Execution Swarm"}</span>
              <div className="task-swarm-actors">
                {swarmActors.map((actor) => (
                  <span
                    className={`task-swarm-actor ${
                      actor === activeSwarmActor ? "active" : ""
                    }`}
                    key={actor}
                  >
                    {actor}
                  </span>
                ))}
              </div>
            </div>
          )}

          {!hasMessages && (
            <div className="empty-state">
              <span>Agent готов</span>
              <p>
                Напишите сообщение. Оно попадет в краткосрочную память текущего
                чата, а OpenAI memory-router решит, нужно ли сохранить его в
                рабочую память или долгосрочную память активного профиля.
              </p>
            </div>
          )}

          {messages.map((message, index) => (
            <article
              className={`message ${message.role} ${
                message.kind === "swarm" ? "swarm-message" : ""
              }`}
              key={`${message.role}-${message.kind ?? "text"}-${index}`}
            >
              <div className="message-header">
                <div className="message-meta">
                  {message.kind === "swarm" ? "Execution Swarm" : getRoleLabel(message.role)}
                </div>
              </div>
              {message.kind === "swarm" && message.swarm ? (
                <SwarmDiscussionPanel discussion={message.swarm} />
              ) : (
                <>
                  <p>{message.content}</p>
                  {message.mcpSteps && message.mcpSteps.length > 0 && (
                    <McpExecutionPanel steps={message.mcpSteps} />
                  )}
                </>
              )}
            </article>
          ))}

          {isLoading && (
            <article className="message assistant">
              <div className="message-meta">Agent</div>
              {hasSwarmLogs && (
                <SwarmDiscussionPanel
                  activeActor={activeSwarmActor}
                  discussion={{
                    actors: swarmActors,
                    logs: swarmActorLogs,
                    status: swarmStatus || "Обсуждение решения"
                  }}
                />
              )}
              {streamingContent ? (
                <>
                  {hasSwarmLogs && <div className="final-stream-label">Final answer</div>}
                  <p>{streamingContent}</p>
                </>
              ) : (
                <p className="typing">
                  {agentPhase === "compressing"
                    ? "Сжимаю краткосрочную память..."
                    : hasSwarmLogs
                      ? "Integrator собирает финальный ответ..."
                      : "Подключаю stream и вызываю LLM..."}
                </p>
              )}
              {mcpExecutionSteps.length > 0 && (
                <McpExecutionPanel steps={mcpExecutionSteps} />
              )}
              {agentPhase === "memory" && (
                <div className="memory-loader" role="status" aria-live="polite">
                  <span aria-hidden="true" />
                  <span>Создаю память агента</span>
                </div>
              )}
            </article>
          )}
        </div>

        {error && <div className="error-box">{error}</div>}

        {settings.orchestrationEnabled && (
          <div className="orchestrator-actions" aria-label="Действия оркестратора">
            <div>
              <strong>Оркестратор управляет этапами</strong>
              <span>{taskState.expectedAction}</span>
            </div>
            <div className="orchestrator-action-buttons">
              <button
                type="button"
                onClick={() =>
                  submitOrchestratorAction("approvePlan", "✅ План одобрен")
                }
                disabled={!canApprovePlan}
              >
                Одобрить план
              </button>
              <button
                type="button"
                onClick={() =>
                  submitOrchestratorAction(
                    "approveSolution",
                    "✅ Решение готово к валидации"
                  )
                }
                disabled={!canApproveSolution}
              >
                Отправить на валидацию
              </button>
              <button
                type="button"
                onClick={() =>
                  submitOrchestratorAction(
                    "disputeSolution",
                    "❌ Решение нужно пересмотреть. Возвращаемся к плану."
                  )
                }
                disabled={!canDisputeSolution}
              >
                Оспорить
              </button>
              <button
                className="danger"
                type="button"
                onClick={cancelTask}
                disabled={!canCancelTask}
              >
                Отменить задачу
              </button>
            </div>
          </div>
        )}

        <form className="composer" onSubmit={handleSubmit}>
          <div className="composer-main">
            <div className="composer-input-shell">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void submitCurrentInput();
                  }
                }}
                placeholder="Введите запрос агенту..."
                rows={3}
              />
              <div className="composer-controls">
                <button
                  className="composer-control-button composer-model-button"
                  type="button"
                  onClick={() => setIsModelPickerOpen(true)}
                  aria-label={`Выбрать модель. Сейчас ${activeModelLabel}`}
                  title="Выбрать модель"
                >
                  <ButtonIcon src={ICONS.ai} />
                </button>
                <button
                  className="composer-control-button composer-send-button"
                  type="submit"
                  disabled={!input.trim() || isLoading}
                  aria-label="Отправить prompt"
                  title="Отправить"
                >
                  <ButtonIcon src={ICONS.sendLetter} />
                </button>
              </div>
            </div>
            <p className="auto-memory-note">
              OpenAI memory-router классифицирует запрос коротким TOON-like
              ответом: рабочий контекст, долгосрочные предпочтения или пропуск.
            </p>
            {enabledMcpServers.length > 0 && (
              <p className="auto-memory-note">
                MCP включен. Агент автоматически выбирает инструменты; прямой вызов остаётся запасным вариантом:{" "}
                <code>/mcp &lt;serverId&gt; &lt;tool&gt; {"{\"arg\":\"value\"}"}</code>.
              </p>
            )}
          </div>
        </form>
      </section>

      <aside
        className={`memory-panel ${isMemoryPanelCollapsed ? "collapsed" : ""}`}
        aria-label="Память и отладка агента"
      >
        {isMemoryPanelCollapsed ? (
          <button
            className="collapsed-panel-button"
            type="button"
            onClick={() => setIsMemoryPanelCollapsed(false)}
            aria-label="Развернуть память агента"
            title="Развернуть память агента"
          >
            <ButtonIcon src={ICONS.arrowLeft} />
            <b>Память</b>
          </button>
        ) : (
          <>
            <section className="memory-section">
              <div className="memory-panel-title-row">
                <div>
                  <p className="eyebrow">Memory layers</p>
                  <h2>Память агента</h2>
                </div>
                <button
                  className="collapse-panel-button"
                  type="button"
                  onClick={() => setIsMemoryPanelCollapsed(true)}
                  aria-label="Свернуть память агента"
                  title="Свернуть память агента"
                >
                  <ButtonIcon src={ICONS.arrowRight} />
                </button>
              </div>

              <div className="memory-layer-list">
                <div className="memory-layer">
                  <strong>{MEMORY_LAYER_LABELS.shortTerm.title}</strong>
                  <span>{MEMORY_LAYER_LABELS.shortTerm.scope}</span>
                  <p>{MEMORY_LAYER_LABELS.shortTerm.description}</p>
                  <b>
                    {messages.length} messages · {shortTermSummary?.compressedTurnCount ?? 0} compressed turns
                  </b>
                </div>
                <div className="memory-layer">
                  <strong>{MEMORY_LAYER_LABELS.userProfile.title}</strong>
                  <span>{MEMORY_LAYER_LABELS.userProfile.scope}</span>
                  <p>{MEMORY_LAYER_LABELS.userProfile.description}</p>
                  <b>{activeProfileName}</b>
                </div>
                <div className="memory-layer">
                  <strong>{MEMORY_LAYER_LABELS.working.title}</strong>
                  <span>{MEMORY_LAYER_LABELS.working.scope}</span>
                  <p>{MEMORY_LAYER_LABELS.working.description}</p>
                  <b>{workingMemory.length}</b>
                </div>
                <div className="memory-layer">
                  <strong>{MEMORY_LAYER_LABELS.longTerm.title}</strong>
                  <span>{MEMORY_LAYER_LABELS.longTerm.scope}</span>
                  <p>{MEMORY_LAYER_LABELS.longTerm.description}</p>
                  <b>{longTermMemory.length}</b>
                </div>
              </div>
            </section>

            <section className="memory-section task-state-section">
              <div>
                <p className="eyebrow">Task State Machine</p>
                <h2>Состояние задачи</h2>
              </div>

              <div className="task-state-card">
                <div className="task-state-row">
                  <span>Этап</span>
                  <strong>{taskPhaseLabel}</strong>
                </div>
                <div className="task-state-row">
                  <span>Статус</span>
                  <strong>
                    {taskState.isCancelled
                      ? "Отменено"
                      : taskState.isPaused
                        ? "Пауза"
                        : "Активно"}
                  </strong>
                </div>
                <div className="task-state-row">
                  <span>Обновлено</span>
                  <strong>{formatChatTime(taskState.updatedAt)}</strong>
                </div>
              </div>

              {settings.debugManualStateControls && (
                <div className="task-state-controls" role="group" aria-label="Управление состоянием задачи">
                  <button
                    type="button"
                    onClick={rewindTaskPhase}
                    disabled={isLoading || !canRewindTask}
                  >
                    Назад
                  </button>
                  <button
                    type="button"
                    onClick={taskState.isPaused ? resumeTask : pauseTask}
                    disabled={isLoading || taskState.isCancelled}
                  >
                    {taskState.isPaused ? "Продолжить" : "Пауза"}
                  </button>
                  <button
                    type="button"
                    onClick={advanceTaskPhase}
                    disabled={isLoading || !canAdvanceTask}
                  >
                    Далее
                  </button>
                  <button type="button" onClick={resetTaskState} disabled={isLoading}>
                    Reset
                  </button>
                </div>
              )}

              <label className="field">
                <span>Текущий шаг</span>
                <textarea
                  value={taskState.currentStep}
                  onChange={(event) =>
                    updateTaskStateText({ currentStep: event.target.value })
                  }
                  readOnly={!settings.debugManualStateControls}
                  rows={3}
                />
              </label>

              <label className="field">
                <span>Ожидаемое действие</span>
                <textarea
                  value={taskState.expectedAction}
                  onChange={(event) =>
                    updateTaskStateText({ expectedAction: event.target.value })
                  }
                  readOnly={!settings.debugManualStateControls}
                  rows={3}
                />
              </label>

              <p className="muted-text">
                Основной сценарий: state machine переключает агент-оркестратор.
                Ручные переходы и редактирование включаются только в debug-настройке.
              </p>
            </section>

            <section className="memory-section">
              <div className="profile-section-header">
                <div>
                  <p className="eyebrow">User profiles</p>
                  <h2>Персонализация</h2>
                </div>
                <button
                  className="new-chat-button compact"
                  type="button"
                  onClick={createUserProfile}
                  disabled={isLoading}
                >
                  <ButtonIcon src={ICONS.plus} />
                  Новый
                </button>
              </div>

              <div className="profile-list">
                {userProfiles.map((profile) => (
                  <div
                    className={`profile-list-item ${
                      profile.id === activeProfile?.id ? "active" : ""
                    }`}
                    key={profile.id}
                  >
                    <button
                      type="button"
                      onClick={() => selectUserProfile(profile.id)}
                      disabled={isLoading}
                    >
                      <strong>{profile.name || "Без имени"}</strong>
                      <small>
                        {profile.longTermMemory.length} долгоср. · обновлено{" "}
                        {formatChatTime(profile.updatedAt)}
                      </small>
                    </button>
                    <button
                      className="delete-chat-button"
                      type="button"
                      onClick={() => deleteUserProfile(profile.id)}
                      disabled={isLoading || userProfiles.length <= 1}
                      aria-label={`Удалить профиль ${profile.name || "Без имени"}`}
                      title="Удалить профиль"
                    >
                      <ButtonIcon src={ICONS.cross} />
                    </button>
                  </div>
                ))}
              </div>

              {activeProfile && (
                <div className="profile-editor">
                  <label className="field">
                    <span>Имя профиля</span>
                    <input
                      value={activeProfile.name}
                      onChange={(event) =>
                        updateActiveUserProfile({ name: event.target.value })
                      }
                    />
                  </label>
                  <label className="field">
                    <span>Стиль</span>
                    <textarea
                      value={activeProfile.style}
                      onChange={(event) =>
                        updateActiveUserProfile({ style: event.target.value })
                      }
                      placeholder="Например: кратко, дружелюбно, без лишней теории"
                      rows={3}
                    />
                  </label>
                  <label className="field">
                    <span>Формат ответа</span>
                    <textarea
                      value={activeProfile.format}
                      onChange={(event) =>
                        updateActiveUserProfile({ format: event.target.value })
                      }
                      placeholder="Например: сначала вывод, потом шаги и пример кода"
                      rows={3}
                    />
                  </label>
                  <label className="field">
                    <span>Ограничения</span>
                    <textarea
                      value={activeProfile.constraints}
                      onChange={(event) =>
                        updateActiveUserProfile({ constraints: event.target.value })
                      }
                      placeholder="Стек, запреты, правила проекта, лимиты"
                      rows={3}
                    />
                  </label>
                  <label className="field">
                    <span>Контекст</span>
                    <textarea
                      value={activeProfile.context}
                      onChange={(event) =>
                        updateActiveUserProfile({ context: event.target.value })
                      }
                      placeholder="Кто пользователь и какой результат ему обычно нужен"
                      rows={3}
                    />
                  </label>
                </div>
              )}
            </section>
            <section className="memory-section">
              <div>
                <p className="eyebrow">LLM routing</p>
                <h2>Как сохраняется</h2>
              </div>
              <div className="router-rules">
                <p>
                  <strong>Всегда:</strong> сообщение сохраняется в краткосрочную память
                  текущего чата.
                </p>
                <p>
                  <strong>Рабочая:</strong> OpenAI LLM сохраняет краткую выжимку,
                  только если это временный контекст активной задачи: фича, баг,
                  файлы, ограничения или ближайшие решения.
                </p>
                <p>
                  <strong>Долгосрочная:</strong> OpenAI LLM сохраняет краткую выжимку,
                  если сообщение содержит профиль: профессию, навыки, языки, цели,
                  проекты или предпочтения активного пользователя.
                </p>
                <p>
                  <strong>Экономия:</strong> роутер получает только последнее сообщение
                  пользователя и отвечает двумя компактными строками.
                </p>
              </div>
            </section>
            <section className="memory-section">
              <div>
                <p className="eyebrow">Current chat</p>
                <h2>Краткосрочная</h2>
              </div>
              <div className="memory-items">
                {shortTermSummary && (
                  <div className="memory-item readonly">
                    <small>
                      Summary · {shortTermSummary.compressedTurnCount} compressed turns
                    </small>
                    <p>{previewText(shortTermSummary.content, 260)}</p>
                  </div>
                )}
                {messages.length === 0 ? (
                  <p className="muted-text">В текущем чате пока нет сообщений.</p>
                ) : (
                  messages.slice(-6).map((message, index) => (
                    <div className="memory-item readonly" key={`${message.role}-${index}`}>
                      <small>{getRoleLabel(message.role)}</small>
                      <p>{previewText(message.content)}</p>
                    </div>
                  ))
                )}
              </div>
            </section>
            <section className="memory-section">
          <div>
            <p className="eyebrow">All chats</p>
            <h2>Рабочая</h2>
          </div>
          <div className="memory-items">
            {workingMemory.length === 0 ? (
              <p className="muted-text">Рабочий контекст пока не сохранен.</p>
            ) : (
              workingMemory.map((item) => (
                <div className="memory-item" key={item.id}>
                  <p>{item.content}</p>
                  <small>обновлено {formatChatTime(item.updatedAt)}</small>
                  <button
                    type="button"
                    onClick={() => removePersistentMemory("working", item)}
                  >
                    Удалить
                  </button>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="memory-section">
          <div>
            <p className="eyebrow">User profile</p>
            <h2>Долгосрочная</h2>
          </div>
          <div className="memory-items">
            {longTermMemory.length === 0 ? (
              <p className="muted-text">Факты и предпочтения пока не сохранены.</p>
            ) : (
              longTermMemory.map((item) => (
                <div className="memory-item" key={item.id}>
                  <p>{item.content}</p>
                  <small>обновлено {formatChatTime(item.updatedAt)}</small>
                  <button
                    type="button"
                    onClick={() => removePersistentMemory("longTerm", item)}
                  >
                    Удалить
                  </button>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="memory-section debug-section">
          <div>
            <p className="eyebrow">Debug</p>
            <h2>Отладка памяти</h2>
          </div>

          <div className="debug-block">
            <strong>Последние решения памяти</strong>
            {lastMemoryWrites.length === 0 ? (
              <p className="muted-text">Пока ничего не сохранялось.</p>
            ) : (
              lastMemoryWrites.map((write) => (
                <p key={`${write.id}-${write.layer}-${write.action}`}>
                  <span>{MEMORY_LAYER_LABELS[write.layer].title}</span>:{" "}
                  {getMemoryActionLabel(write.action)} · {write.reason} ·{" "}
                  {write.content}
                </p>
              ))
            )}
          </div>

          <div className="debug-block">
            <strong>Последний запрос к агенту</strong>
            {lastReply?.debug ? (
              <>
                <dl className="debug-grid">
                  <div>
                    <dt>layers</dt>
                    <dd>{lastReply.debug.includedLayers.join(", ")}</dd>
                  </div>
                  <div>
                    <dt>profile</dt>
                    <dd>{lastReply.debug.activeProfileName ?? "none"}</dd>
                  </div>
                  <div>
                    <dt>task</dt>
                    <dd>
                      {lastReply.debug.taskPhase ?? "none"}
                      {lastReply.debug.taskPaused ? " · paused" : ""}
                    </dd>
                  </div>
                  <div>
                    <dt>task step</dt>
                    <dd>{lastReply.debug.taskCurrentStep || "none"}</dd>
                  </div>
                  <div>
                    <dt>expected</dt>
                    <dd>{lastReply.debug.taskExpectedAction || "none"}</dd>
                  </div>
                  <div>
                    <dt>profile chars</dt>
                    <dd>{lastReply.debug.activeProfileChars}</dd>
                  </div>
                  <div>
                    <dt>input</dt>
                    <dd>{lastReply.debug.inputMessageCount}</dd>
                  </div>
                  <div>
                    <dt>short</dt>
                    <dd>{lastReply.debug.shortTermMessageCount}</dd>
                  </div>
                  <div>
                    <dt>short visible</dt>
                    <dd>{lastReply.debug.shortTermVisibleMessageCount}</dd>
                  </div>
                  <div>
                    <dt>short input</dt>
                    <dd>{lastReply.debug.shortTermInputMessageCount}</dd>
                  </div>
                  <div>
                    <dt>summary turns</dt>
                    <dd>{lastReply.debug.shortTermCompressedTurnCount}</dd>
                  </div>
                  <div>
                    <dt>summary chars</dt>
                    <dd>{lastReply.debug.shortTermSummaryChars}</dd>
                  </div>
                  <div>
                    <dt>compression</dt>
                    <dd>
                      {lastReply.debug.shortTermCompressionEnabled ? "on" : "off"} · limit{" "}
                      {lastReply.debug.shortTermCompressionLimit} ·{" "}
                      {lastReply.debug.shortTermCompressionTriggered ? "updated" : "idle"}
                    </dd>
                  </div>
                  <div>
                    <dt>working</dt>
                    <dd>{lastReply.debug.workingItemCount}</dd>
                  </div>
                  <div>
                    <dt>long</dt>
                    <dd>{lastReply.debug.longTermItemCount}</dd>
                  </div>
                  <div>
                    <dt>chars</dt>
                    <dd>{lastReply.debug.memoryInstructionChars}</dd>
                  </div>
                  <div>
                    <dt>mcp</dt>
                    <dd>{lastReply.debug.mcpEnabled ? lastReply.debug.mcpStatus : "off"}</dd>
                  </div>
                  <div>
                    <dt>mcp tools</dt>
                    <dd>{lastReply.debug.mcpToolCount}</dd>
                  </div>
                  <div>
                    <dt>mcp call</dt>
                    <dd>{lastReply.debug.mcpToolCall?.toolName ?? "none"}</dd>
                  </div>
                </dl>
                <pre>{lastReply.debug.promptPreview}</pre>
                <strong>MCP tools</strong>
                {lastReply.debug.mcpTools.length === 0 ? (
                  <p className="muted-text">
                    {lastReply.debug.mcpEnabled
                      ? "No tools returned from enabled MCP servers."
                      : "MCP is disabled."}
                  </p>
                ) : (
                  <div className="mcp-tool-list">
                    {lastReply.debug.mcpTools.map((tool) => (
                      <div
                        className="mcp-tool-item"
                        key={`${tool.serverId}:${tool.name}`}
                      >
                        <b>[{tool.serverName}] {tool.name}</b>
                        <p>{tool.description ?? "No description."}</p>
                        <code>{formatJsonPreview(tool.inputSchema)}</code>
                      </div>
                    ))}
                  </div>
                )}
                <strong>MCP tool call</strong>
                {lastReply.debug.mcpToolCall ? (
                  <pre>
                    {[
                      `server: ${lastReply.debug.mcpToolCall.serverName} (${lastReply.debug.mcpToolCall.serverId})`,
                      `tool: ${lastReply.debug.mcpToolCall.toolName}`,
                      `arguments: ${lastReply.debug.mcpToolCall.arguments}`,
                      `isError: ${lastReply.debug.mcpToolCall.isError}`,
                      "result:",
                      lastReply.debug.mcpToolCall.result
                    ].join("\n")}
                  </pre>
                ) : (
                  <p className="muted-text">No MCP tool was called for the last message.</p>
                )}
                <strong>Short-term compression input</strong>
                <pre>
                  {lastReply.debug.shortTermCompressionInput ||
                    "Компрессия краткосрочной памяти не запускалась."}
                </pre>
                <strong>Raw short-term compression</strong>
                <pre>
                  {lastReply.debug.shortTermCompressionRaw ||
                    "Компрессия краткосрочной памяти еще не возвращала результат."}
                </pre>
                <strong>Memory-router input</strong>
                <pre>
                  {lastReply.debug.memoryRouterInput ||
                    "Memory-router input еще не сформирован."}
                </pre>
                <strong>Raw memory-router</strong>
                <pre>
                  {lastReply.debug.memoryRouterRaw ||
                    "Memory-router еще не вернул сырой ответ."}
                </pre>
              </>
            ) : (
              <p className="muted-text">
                После ответа здесь появится фрагмент memory-инструкции, которую
                получил backend.
              </p>
            )}
          </div>
        </section>
          </>
        )}
      </aside>

      {isModelPickerOpen && (
        <div
          className="modal-backdrop model-picker-backdrop"
          role="presentation"
          onMouseDown={() => setIsModelPickerOpen(false)}
        >
          <section
            className="model-picker-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="model-picker-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header className="modal-header">
              <div>
                <p className="eyebrow">Модель ответа</p>
                <h2 id="model-picker-title">Выбор модели</h2>
              </div>
              <button
                className="icon-button"
                type="button"
                aria-label="Закрыть выбор модели"
                onClick={() => setIsModelPickerOpen(false)}
              >
                <ButtonIcon src={ICONS.cross} />
              </button>
            </header>

            <div className="model-option-list">
              {MODEL_OPTIONS.map((model) => (
                <button
                  className={`model-option-button ${
                    model.id === settings.model ? "active" : ""
                  }`}
                  key={model.id}
                  type="button"
                  onClick={() => selectModel(model.id)}
                >
                  <ButtonIcon src={ICONS.ai} />
                  <span>
                    <strong>{model.label}</strong>
                    <small>{model.id}</small>
                  </span>
                  {model.id === settings.model && <b>Выбрана</b>}
                </button>
              ))}
            </div>
          </section>
        </div>
      )}

      {isSettingsOpen && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={() => setIsSettingsOpen(false)}
        >
          <section
            className="settings-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header className="modal-header">
              <div>
                <p className="eyebrow">Параметры агента</p>
                <h2 id="settings-title">Настройки</h2>
              </div>
              <button
                className="icon-button"
                type="button"
                aria-label="Закрыть настройки"
                onClick={() => setIsSettingsOpen(false)}
              >
                <ButtonIcon src={ICONS.cross} />
              </button>
            </header>

            <div className="settings-list">
              <label className="field">
                <span>OpenAI API key</span>
                <input
                  value={settings.apiKey}
                  onChange={(event) => updateSettings({ apiKey: event.target.value })}
                  placeholder="sk-..."
                  type="password"
                />
                <small>
                  Ключ сохраняется локально на этом устройстве. Для учебного проекта
                  этого достаточно; позже можно перенести его в системное хранилище.
                </small>
              </label>

              <label className="field">
                <span>Модель по умолчанию</span>
                <div className="select-with-icon">
                  <ButtonIcon src={ICONS.ai} />
                  <select
                    value={settings.model}
                    onChange={(event) => updateSettings({ model: event.target.value })}
                  >
                    {MODEL_OPTIONS.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                  <ButtonIcon className="button-icon-small" src={ICONS.arrowDown} />
                </div>
              </label>

              <div className="field">
                <span>Тема</span>
                <div className="segmented-control" role="group" aria-label="Переключение темы">
                  <button
                    className={settings.theme === "light" ? "active" : ""}
                    type="button"
                    onClick={() => updateSettings({ theme: "light" })}
                  >
                    Светлая
                  </button>
                  <button
                    className={settings.theme === "dark" ? "active" : ""}
                    type="button"
                    onClick={() => updateSettings({ theme: "dark" })}
                  >
                    Темная
                  </button>
                </div>
              </div>

              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={settings.autoScroll}
                  onChange={(event) => updateSettings({ autoScroll: event.target.checked })}
                />
                <span>
                  <b>Автоскролл чата</b>
                  <small>Во время stream-ответа автоматически держать чат внизу.</small>
                </span>
              </label>

              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={settings.shortTermCompressionEnabled}
                  onChange={(event) =>
                    updateSettings({ shortTermCompressionEnabled: event.target.checked })
                  }
                />
                <span>
                  <b>Компрессия краткосрочной памяти</b>
                  <small>
                    Старые ходы текущего чата сворачиваются в summary, но полная история остается
                    видимой.
                  </small>
                </span>
              </label>

              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={settings.orchestrationEnabled}
                  onChange={(event) =>
                    updateSettings({ orchestrationEnabled: event.target.checked })
                  }
                />
                <span>
                  <b>Агент-оркестратор и stage agents</b>
                  <small>
                    Если включено, этапами управляет backend-оркестратор: Planning Agent,
                    Execution Agent, Validation Agent и Done state. Если выключено,
                    работает старый обычный чат.
                  </small>
                </span>
              </label>

              <section className="mcp-settings-section" aria-labelledby="mcp-settings-title">
                <div className="mcp-settings-header">
                  <div>
                    <b id="mcp-settings-title">MCP-серверы</b>
                    <small>
                      Поддерживаются локальные stdio и удалённые Streamable HTTP MCP.
                      Агент объединяет их tools и сохраняет раздельные идентификаторы.
                    </small>
                  </div>
                  <div className="mcp-add-actions">
                    <button
                      className="ghost-button"
                      type="button"
                      onClick={addYandexTrackerMcpServer}
                    >
                      + Tracker
                    </button>
                    <button
                      className="ghost-button"
                      type="button"
                      onClick={addTelegramDeliveryMcpServer}
                    >
                      + Telegram MCP
                    </button>
                    <button className="ghost-button" type="button" onClick={addFigmaMcpServer}>
                      + Figma
                    </button>
                    <button
                      className="ghost-button"
                      type="button"
                      onClick={() => addMcpServer("streamableHttp")}
                    >
                      + Remote
                    </button>
                    <button
                      className="ghost-button"
                      type="button"
                      onClick={() => addMcpServer("stdio")}
                    >
                      + stdio
                    </button>
                  </div>
                </div>

                {settings.mcpServers.length === 0 ? (
                  <p className="muted-text">
                    Серверов пока нет. Добавьте MCP и укажите команду запуска.
                  </p>
                ) : (
                  <div className="mcp-server-list">
                    {settings.mcpServers.map((server) => {
                      const connectionTest = mcpConnectionTests[server.id];
                      const isTesting = testingMcpServerId === server.id;

                      return (
                        <article className="mcp-server-card" key={server.id}>
                          <div className="mcp-server-card-header">
                            <label className="mcp-server-toggle">
                              <input
                                type="checkbox"
                                checked={server.enabled}
                                onChange={(event) =>
                                  updateMcpServer(server.id, {
                                    enabled: event.target.checked
                                  })
                                }
                              />
                              <span>Включен</span>
                            </label>
                            <code title="ID для команды /mcp">{server.id}</code>
                            <button
                              className="mcp-remove-button"
                              type="button"
                              onClick={() => removeMcpServer(server.id)}
                              aria-label={`Удалить MCP ${server.name}`}
                            >
                              <ButtonIcon src={ICONS.cross} />
                            </button>
                          </div>

                          <div className="mcp-server-grid">
                            <label className="field">
                              <span>Название</span>
                              <input
                                value={server.name}
                                onChange={(event) =>
                                  updateMcpServer(server.id, { name: event.target.value })
                                }
                                placeholder="Filesystem"
                              />
                            </label>
                            <label className="field">
                              <span>Транспорт</span>
                              <select
                                value={server.transport}
                                onChange={(event) =>
                                  updateMcpServer(server.id, {
                                    transport: event.target.value as McpServerConfig["transport"]
                                  })
                                }
                              >
                                <option value="streamableHttp">Streamable HTTP</option>
                                <option value="stdio">stdio</option>
                              </select>
                            </label>
                          </div>

                          {server.transport === "stdio" ? (
                            <>
                              <label className="field">
                                <span>Команда</span>
                                <input
                                  value={server.command}
                                  onChange={(event) =>
                                    updateMcpServer(server.id, { command: event.target.value })
                                  }
                                  placeholder="npx, uvx, node или полный путь"
                                  spellCheck={false}
                                />
                              </label>

                              <label className="field">
                                <span>Аргументы — по одному на строку</span>
                                <textarea
                                  value={server.args.join("\n")}
                                  onChange={(event) =>
                                    updateMcpServer(server.id, {
                                      args: event.target.value.split(/\r?\n/)
                                    })
                                  }
                                  rows={Math.max(2, Math.min(5, server.args.length + 1))}
                                  placeholder={"-y\n@modelcontextprotocol/server-everything"}
                                  spellCheck={false}
                                />
                              </label>

                              <label className="field">
                                <span>Рабочая папка (необязательно)</span>
                                <input
                                  value={server.cwd ?? ""}
                                  onChange={(event) =>
                                    updateMcpServer(server.id, { cwd: event.target.value })
                                  }
                                  placeholder="C:\\path\\to\\project"
                                  spellCheck={false}
                                />
                              </label>

                              <div className="mcp-env-block">
                                <div className="mcp-env-header">
                                  <span>Переменные окружения</span>
                                  <button
                                    type="button"
                                    onClick={() => addMcpEnvironmentVariable(server)}
                                  >
                                    + Переменная
                                  </button>
                                </div>
                                {Object.entries(server.env).map(([key, value]) => (
                                  <div className="mcp-env-row" key={key}>
                                    <input
                                      aria-label="Имя переменной окружения"
                                      value={key}
                                      onChange={(event) =>
                                        updateMcpEnvironmentVariable(
                                          server,
                                          key,
                                          event.target.value,
                                          value
                                        )
                                      }
                                      placeholder="API_KEY"
                                      spellCheck={false}
                                    />
                                    <input
                                      aria-label={`Значение ${key}`}
                                      value={value}
                                      onChange={(event) =>
                                        updateMcpEnvironmentVariable(
                                          server,
                                          key,
                                          key,
                                          event.target.value
                                        )
                                      }
                                      placeholder="value"
                                      type={/key|token|secret|password/i.test(key) ? "password" : "text"}
                                      spellCheck={false}
                                    />
                                    <button
                                      type="button"
                                      onClick={() =>
                                        updateMcpEnvironmentVariable(server, key, "", "")
                                      }
                                      aria-label={`Удалить переменную ${key}`}
                                    >
                                      ×
                                    </button>
                                  </div>
                                ))}
                              </div>
                            </>
                          ) : (
                            <>
                              <label className="field">
                                <span>MCP endpoint URL</span>
                                <input
                                  value={server.url ?? ""}
                                  onChange={(event) =>
                                    updateMcpServer(server.id, { url: event.target.value })
                                  }
                                  placeholder="https://example.com/mcp"
                                  type="url"
                                  spellCheck={false}
                                />
                              </label>

                              <div className="mcp-env-block">
                                <div className="mcp-env-header">
                                  <span>HTTP-заголовки</span>
                                  <button type="button" onClick={() => addMcpHeader(server)}>
                                    + Заголовок
                                  </button>
                                </div>
                                {Object.entries(server.headers).map(([key, value]) => (
                                  <div className="mcp-env-row" key={key}>
                                    <input
                                      aria-label="Имя HTTP-заголовка"
                                      value={key}
                                      onChange={(event) =>
                                        updateMcpHeader(server, key, event.target.value, value)
                                      }
                                      placeholder="Authorization"
                                      spellCheck={false}
                                    />
                                    <input
                                      aria-label={`Значение заголовка ${key}`}
                                      value={value}
                                      onChange={(event) =>
                                        updateMcpHeader(server, key, key, event.target.value)
                                      }
                                      placeholder="Bearer ..."
                                      type={/authorization|key|token|secret/i.test(key) ? "password" : "text"}
                                      spellCheck={false}
                                    />
                                    <button
                                      type="button"
                                      onClick={() => updateMcpHeader(server, key, "", "")}
                                      aria-label={`Удалить заголовок ${key}`}
                                    >
                                      ×
                                    </button>
                                  </div>
                                ))}
                              </div>

                              {server.url === "https://mcp.figma.com/mcp" && (
                                <p className="mcp-oauth-note">
                                  Figma Remote требует интерактивный OAuth и сейчас принимает
                                  только клиенты из каталога Figma. Этот клиент проверит HTTP
                                  endpoint, но для входа потребуется регистрация приложения у Figma.
                                </p>
                              )}
                              {server.name === "Yandex Tracker" && (
                                <p className="mcp-oauth-note">
                                  Укажите URL развёрнутого YandexTrackerMCP и замените
                                  <code> Bearer change-me</code> на ваш MCP_API_KEY. Затем
                                  включите сервер и нажмите «Проверить».
                                </p>
                              )}
                              {server.name === "Telegram Delivery" && (
                                <p className="mcp-oauth-note">
                                  Второй MCP-сервер для Day 20. Укажите URL
                                  YandexTrackerTelegramMCP и его отдельный
                                  <code> MCP_API_KEY</code>. Для длинного флоу включите
                                  одновременно Tracker и Telegram Delivery.
                                </p>
                              )}
                            </>
                          )}

                          <div className="mcp-test-actions">
                            <button
                              className="ghost-button"
                              type="button"
                              disabled={
                                isTesting ||
                                (server.transport === "stdio"
                                  ? !server.command.trim()
                                  : !server.url?.trim())
                              }
                              onClick={() => void testMcpConnection(server)}
                            >
                              {isTesting ? "Подключение…" : "Проверить и получить tools"}
                            </button>
                            <small>
                              Вызов: <code>/mcp {server.id} &lt;tool&gt; {"{...}"}</code>
                            </small>
                          </div>

                          {connectionTest && (
                            <div
                              className={`mcp-test-result ${
                                connectionTest.connected ? "success" : "error"
                              }`}
                            >
                              <b>
                                {connectionTest.connected ? "Соединение установлено" : "Ошибка"}
                              </b>
                              <p>{connectionTest.status}</p>
                              {connectionTest.tools.length > 0 && (
                                <div className="mcp-test-tools">
                                  {connectionTest.tools.map((tool) => (
                                    <code key={`${tool.serverId}:${tool.name}`}>
                                      {tool.name}
                                    </code>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                        </article>
                      );
                    })}
                  </div>
                )}
              </section>

              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={settings.debugManualStateControls}
                  onChange={(event) =>
                    updateSettings({ debugManualStateControls: event.target.checked })
                  }
                />
                <span>
                  <b>Debug: ручные переходы state machine</b>
                  <small>
                    Только для отладки. В основном сценарии пользователь не переключает
                    state вручную.
                  </small>
                </span>
              </label>

              <label className="field">
                <span>Инварианты валидатора</span>
                <textarea
                  value={settings.validatorInvariants}
                  onChange={(event) =>
                    updateSettings({ validatorInvariants: event.target.value })
                  }
                  rows={6}
                />
                <small>
                  Кодовая проверка поддерживает строки `must: текст` и `forbid: текст`.
                  Эти же правила уходят в prompt Validation Agent.
                </small>
              </label>

              <label className="field">
                <span>Ходов без сжатия</span>
                <input
                  min={2}
                  max={50}
                  type="number"
                  value={settings.shortTermCompressionTurnLimit}
                  onChange={(event) =>
                    updateSettings({
                      shortTermCompressionTurnLimit: Number(event.target.value)
                    })
                  }
                />
                <small>
                  Один ход = запрос пользователя и ответ ИИ. По умолчанию храним 10 свежих ходов
                  без сжатия.
                </small>
              </label>

              <label className="field">
                <span>System prompt</span>
                <textarea
                  value={settings.systemPrompt}
                  onChange={(event) => updateSettings({ systemPrompt: event.target.value })}
                  rows={6}
                />
              </label>
            </div>

            <footer className="modal-footer">
              <span>Сохраняется автоматически</span>
              <button
                className="primary-button"
                type="button"
                onClick={() => setIsSettingsOpen(false)}
              >
                Готово
              </button>
            </footer>
          </section>
        </div>
      )}
    </main>
  );
}

export default App;
