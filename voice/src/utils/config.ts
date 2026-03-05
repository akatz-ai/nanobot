import { homedir } from "node:os";
import { readFileSync } from "node:fs";
import path from "node:path";

export interface EnvConfig {
  openAiApiKey: string;
  discordBotToken: string;
  discordGuildId: string;
  discordVoiceChannelId: string;
  discordOwnerUserId: string;
  openAiRealtimeModel: string;
  openAiRealtimeVoice: string;
  idleTimeoutMs: number;
  logLevel: string;
  anthropicApiKey?: string;
  nanobotConfigPath: string;
}

export interface NanobotAgentProfile {
  discordChannels?: string[];
  displayName?: string | null;
  discordWebhookUrl?: string | null;
}

export interface NanobotConfig {
  agents?: {
    profiles?: Record<string, NanobotAgentProfile>;
  };
  channels?: {
    discord?: {
      guildId?: string;
      token?: string;
    };
  };
}

export interface ResolvedAgent {
  requested: string;
  resolvedName: string;
  channelId: string;
  displayName: string;
  webhookUrl?: string;
}

function required(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

export function expandHome(inputPath: string): string {
  if (inputPath.startsWith("~/")) {
    return path.join(homedir(), inputPath.slice(2));
  }
  return inputPath;
}

export function loadEnvConfig(): EnvConfig {
  const idleRaw = process.env.IDLE_TIMEOUT_MS?.trim() ?? "300000";
  const idleTimeoutMs = Number.parseInt(idleRaw, 10);
  if (!Number.isFinite(idleTimeoutMs) || idleTimeoutMs < 0) {
    throw new Error(`Invalid IDLE_TIMEOUT_MS: ${idleRaw}`);
  }

  return {
    openAiApiKey: required("OPENAI_API_KEY"),
    discordBotToken: required("DISCORD_BOT_TOKEN"),
    discordGuildId: required("DISCORD_GUILD_ID"),
    discordVoiceChannelId: required("DISCORD_VOICE_CHANNEL_ID"),
    discordOwnerUserId: required("DISCORD_OWNER_USER_ID"),
    openAiRealtimeModel: process.env.OPENAI_REALTIME_MODEL?.trim() || "gpt-realtime-mini",
    openAiRealtimeVoice: process.env.OPENAI_REALTIME_VOICE?.trim() || "sage",
    idleTimeoutMs,
    logLevel: process.env.LOG_LEVEL?.trim() || "info",
    anthropicApiKey: process.env.ANTHROPIC_API_KEY?.trim() || undefined,
    nanobotConfigPath: expandHome(process.env.NANOBOT_CONFIG_PATH?.trim() || "~/.nanobot/config.json"),
  };
}

export function loadNanobotConfig(configPath: string): NanobotConfig {
  const raw = readFileSync(configPath, "utf-8");
  return JSON.parse(raw) as NanobotConfig;
}

function normalizeAgentName(name: string): string {
  return name.trim().toLowerCase().replace(/_/g, "-");
}

const AGENT_ALIASES: Record<string, string[]> = {
  general: ["general", "devius"],
  recon: ["recon", "researcher"],
  "comfygit-dev": ["comfygit-dev", "comfygit"],
  forge: ["forge", "nanobot-dev", "sysadmin"],
  atlas: ["atlas", "assistant"],
};

export function listAgentChannelMappings(config: NanobotConfig): ResolvedAgent[] {
  const profiles = config.agents?.profiles ?? {};
  const results: ResolvedAgent[] = [];

  for (const [name, profile] of Object.entries(profiles)) {
    const channelId = profile.discordChannels?.[0];
    if (!channelId) {
      continue;
    }
    results.push({
      requested: name,
      resolvedName: name,
      channelId,
      displayName: profile.displayName || name,
      webhookUrl: profile.discordWebhookUrl || undefined,
    });
  }

  return results.sort((a, b) => a.resolvedName.localeCompare(b.resolvedName));
}

export function resolveAgent(config: NanobotConfig, requestedAgent: string): ResolvedAgent | null {
  const profiles = config.agents?.profiles ?? {};
  const byName = new Map<string, [string, NanobotAgentProfile]>();

  for (const [name, profile] of Object.entries(profiles)) {
    byName.set(normalizeAgentName(name), [name, profile]);
  }

  const requested = normalizeAgentName(requestedAgent);
  const aliasList = AGENT_ALIASES[requested] ?? [requested];

  for (const candidate of aliasList) {
    const hit = byName.get(candidate);
    if (!hit) {
      continue;
    }
    const [resolvedName, profile] = hit;
    const channelId = profile.discordChannels?.[0];
    if (!channelId) {
      return null;
    }
    return {
      requested: requestedAgent,
      resolvedName,
      channelId,
      displayName: profile.displayName || resolvedName,
      webhookUrl: profile.discordWebhookUrl || undefined,
    };
  }

  return null;
}
