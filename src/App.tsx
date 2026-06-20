import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  AgentReply,
  AgentMemoryStarted,
  ChatMessage,
  MemoryContext,
  MemoryDecision,
  MemoryItem,
  MemoryLayerId,
  AgentStreamDelta,
  ShortTermCompressionSettings,
  ShortTermSummary,
  UserProfile
} from "./types";

const DEFAULT_SYSTEM_PROMPT =
  "You are a helpful AI agent inside a desktop application. Answer clearly and keep context from the conversation.";

const SETTINGS_STORAGE_KEY = "chatbot-ai.settings.v1";
const CHATS_STORAGE_KEY = "chatbot-ai.chats.v1";
const WORKING_MEMORY_STORAGE_KEY = "chatbot-ai.memory.working.v1";
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
  shortTermCompressionTurnLimit: 10
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

    const parsed = JSON.parse(saved) as Partial<AppSettings>;
    return {
      ...DEFAULT_SETTINGS,
      ...parsed,
      shortTermCompressionTurnLimit: normalizeCompressionTurnLimit(
        parsed.shortTermCompressionTurnLimit
      )
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
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

function getRoleLabel(role: ChatMessage["role"]): string {
  return role === "user" ? "Вы" : "Agent";
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
  const [lastMemoryWrites, setLastMemoryWrites] = useState<MemoryWrite[]>([]);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isModelPickerOpen, setIsModelPickerOpen] = useState(false);
  const [isProfilePickerOpen, setIsProfilePickerOpen] = useState(false);
  const [isChatSidebarCollapsed, setIsChatSidebarCollapsed] = useState(false);
  const [isMemoryPanelCollapsed, setIsMemoryPanelCollapsed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [agentPhase, setAgentPhase] = useState<AgentPhase>("idle");
  const [streamingContent, setStreamingContent] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [lastReply, setLastReply] = useState<AgentReply | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
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
      longTerm: longTermMemory
    }),
    [activeProfileForRequest, messages, shortTermSummary, workingMemory, longTermMemory]
  );

  const shortTermTurnCount = useMemo(
    () => messages.filter((message) => message.role === "user").length,
    [messages]
  );

  const estimatedChars = useMemo(
    () => messages.reduce((total, message) => total + message.content.length, 0),
    [messages]
  );

  useEffect(() => {
    localStorage.setItem(CHATS_STORAGE_KEY, JSON.stringify(chats));
  }, [chats]);

  useEffect(() => {
    localStorage.setItem(WORKING_MEMORY_STORAGE_KEY, JSON.stringify(workingMemory));
  }, [workingMemory]);

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

  async function submitCurrentInput() {
    const text = input.trim();
    if (!text || isLoading || !activeChat) {
      return;
    }

    const requestId = crypto.randomUUID();
    const previousMessages = activeChat.messages;
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
    setAgentPhase(shouldCompressShortTerm(nextMessages, settings) ? "compressing" : "streaming");
    setStreamingContent("");

    let unlistenStream: (() => void) | undefined;
    let unlistenMemory: (() => void) | undefined;

    try {
      unlistenStream = await listen<AgentStreamDelta>(
        "agent_stream_delta",
        (event) => {
          if (event.payload.requestId !== requestId) {
            return;
          }

          setStreamingContent((current) => current + event.payload.delta);
          setAgentPhase((current) => (current === "compressing" ? "streaming" : current));
        }
      );

      unlistenMemory = await listen<AgentMemoryStarted>(
        "agent_memory_started",
        (event) => {
          if (event.payload.requestId === requestId) {
            setAgentPhase("memory");
          }
        }
      );

      const requestMemoryContext: MemoryContext = {
        activeProfile: activeProfileForRequest,
        shortTerm: nextMessages,
        shortTermSummary,
        working: workingMemory,
        longTerm: longTermMemory
      };

      const reply = await invoke<AgentReply>("send_agent_message", {
        request: {
          requestId,
          apiKey: settings.apiKey.trim() || undefined,
          model: settings.model.trim(),
          systemPrompt: settings.systemPrompt,
          messages: nextMessages,
          memoryContext: requestMemoryContext,
          shortTermCompression
        }
      });

      const completedMessages: ChatMessage[] = [
        ...nextMessages,
        { role: "assistant", content: reply.content }
      ];
      const nextShortTermSummary = reply.shortTermSummary
        ? {
            ...reply.shortTermSummary,
            updatedAt: new Date().toISOString()
          }
        : undefined;

      setLastReply(reply);
      updateActiveChat(completedMessages, nextShortTermSummary);
      setLastMemoryWrites([
        ...memoryWrites,
        ...applyMemoryDecisions(reply.memoryDecisions, text, activeChat.id)
      ]);
      setStreamingContent("");
    } catch (caughtError) {
      const message =
        caughtError instanceof Error ? caughtError.message : String(caughtError);
      setError(message);
      updateActiveChat(previousMessages);
      setStreamingContent("");
    } finally {
      unlistenStream?.();
      unlistenMemory?.();
      setIsLoading(false);
      setAgentPhase("idle");
      inputRef.current?.focus();
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    void submitCurrentInput();
  }

  function selectModel(modelId: string) {
    updateSettings({ model: modelId });
    setIsModelPickerOpen(false);
    inputRef.current?.focus();
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
        </div>

        <div className="messages" aria-live="polite" ref={messagesRef}>
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
            <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <div className="message-header">
                <div className="message-meta">{getRoleLabel(message.role)}</div>
              </div>
              <p>{message.content}</p>
            </article>
          ))}

          {isLoading && (
            <article className="message assistant">
              <div className="message-meta">Agent</div>
              {streamingContent ? (
                <p>{streamingContent}</p>
              ) : (
                <p className="typing">
                  {agentPhase === "compressing"
                    ? "Сжимаю краткосрочную память..."
                    : "Подключаю stream и вызываю LLM..."}
                </p>
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
                </dl>
                <pre>{lastReply.debug.promptPreview}</pre>
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
