import { loadNanobotConfig, resolveAgent } from "../utils/config.js";

const DISCORD_API_BASE = "https://discord.com/api/v10";

export interface ReadAgentChannelInput {
  discordBotToken: string;
  nanobotConfigPath: string;
  agent: string;
  count?: number;
}

function clampCount(value: number | undefined): number {
  if (!value || !Number.isFinite(value)) {
    return 5;
  }
  return Math.min(20, Math.max(1, Math.floor(value)));
}

export async function readAgentChannel(input: ReadAgentChannelInput): Promise<Record<string, unknown>> {
  const config = loadNanobotConfig(input.nanobotConfigPath);
  const resolved = resolveAgent(config, input.agent);
  if (!resolved) {
    return {
      status: "error",
      error: `Unknown agent or missing channel mapping: ${input.agent}`,
    };
  }

  const count = clampCount(input.count);
  const url = `${DISCORD_API_BASE}/channels/${resolved.channelId}/messages?limit=${count}`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      Authorization: `Bot ${input.discordBotToken}`,
    },
  });

  if (!response.ok) {
    const body = await response.text();
    return {
      status: "error",
      error: `Discord API returned ${response.status}`,
      details: body.slice(0, 300),
    };
  }

  const data = (await response.json()) as Array<Record<string, unknown>>;
  const messages = data
    .slice()
    .reverse()
    .map((message) => {
      const author = (message.author as Record<string, unknown> | undefined) ?? {};
      const username = String(author.global_name ?? author.username ?? "unknown");
      const content = String(message.content ?? "").trim();
      return {
        id: String(message.id ?? ""),
        author: username,
        timestamp: String(message.timestamp ?? ""),
        content: content || "[non-text message]",
      };
    });

  return {
    status: "ok",
    agent: input.agent,
    resolved_agent: resolved.resolvedName,
    channel_id: resolved.channelId,
    message_count: messages.length,
    messages,
  };
}
