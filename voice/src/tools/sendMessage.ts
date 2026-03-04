import { ConfirmationGate, PendingSendAction } from "./confirmationGate.js";
import { loadNanobotConfig, resolveAgent } from "../utils/config.js";

const DISCORD_API_BASE = "https://discord.com/api/v10";

export interface QueueAgentMessageInput {
  gate: ConfirmationGate;
  nanobotConfigPath: string;
  agent: string;
  message: string;
}

export interface SendDiscordMessageInput {
  discordBotToken: string;
  channelId: string;
  content: string;
}

export function queueMessageToAgent(input: QueueAgentMessageInput): Record<string, unknown> {
  const message = input.message.trim();
  if (!message) {
    return {
      status: "error",
      error: "Message cannot be empty.",
    };
  }

  const config = loadNanobotConfig(input.nanobotConfigPath);
  const resolved = resolveAgent(config, input.agent);
  if (!resolved) {
    return {
      status: "error",
      error: `Unknown agent or missing channel mapping: ${input.agent}`,
    };
  }

  const preview = `Send to ${resolved.resolvedName}: "${message}"`;
  const action = input.gate.queueSendAction({
    agent: input.agent,
    resolvedAgent: resolved.resolvedName,
    channelId: resolved.channelId,
    message,
    preview,
  });

  return pendingActionToResult(action);
}

export function pendingActionToResult(action: PendingSendAction): Record<string, unknown> {
  return {
    status: "pending_confirmation",
    action_id: action.id,
    expires_at: new Date(action.expiresAt).toISOString(),
    preview: action.preview,
    agent: action.resolvedAgent,
    channel_id: action.channelId,
  };
}

export async function sendDiscordMessage(input: SendDiscordMessageInput & { webhookUrl?: string }): Promise<Record<string, unknown>> {
  // Prefer webhook so the message doesn't come from the bot account
  // (nanobot ignores bot messages, but allows webhook messages)
  if (input.webhookUrl) {
    const response = await fetch(`${input.webhookUrl}?wait=true`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: input.content,
        username: "Alex (Voice)",
      }),
    });

    if (!response.ok) {
      const body = await response.text();
      return {
        status: "error",
        error: `Discord webhook send failed: ${response.status}`,
        details: body.slice(0, 300),
      };
    }

    const payload = (await response.json()) as Record<string, unknown>;
    return {
      status: "ok",
      message_id: String(payload.id ?? ""),
      channel_id: input.channelId,
      content: input.content,
      sent_via: "webhook",
    };
  }

  // Fallback to bot token (won't trigger nanobot responses)
  const response = await fetch(`${DISCORD_API_BASE}/channels/${input.channelId}/messages`, {
    method: "POST",
    headers: {
      Authorization: `Bot ${input.discordBotToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ content: input.content }),
  });

  if (!response.ok) {
    const body = await response.text();
    return {
      status: "error",
      error: `Discord send failed: ${response.status}`,
      details: body.slice(0, 300),
    };
  }

  const payload = (await response.json()) as Record<string, unknown>;
  return {
    status: "ok",
    message_id: String(payload.id ?? ""),
    channel_id: input.channelId,
    content: input.content,
    sent_via: "bot_token",
  };
}
