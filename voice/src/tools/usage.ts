import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

const ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01";
const ANTHROPIC_BETA = "oauth-2025-04-20";
const PROBE_MODEL = process.env.ANTHROPIC_PROBE_MODEL || "claude-sonnet-4-20250514";

function toFloat(headers: Headers, key: string): number {
  const raw = headers.get(key);
  if (!raw) {
    return 0;
  }
  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : 0;
}

function toInt(headers: Headers, key: string): number {
  const raw = headers.get(key);
  if (!raw) {
    return 0;
  }
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : 0;
}

function toStr(headers: Headers, key: string): string {
  return headers.get(key) || "unknown";
}

function extractUsageHeaders(headers: Headers): Record<string, unknown> {
  return {
    utilization_5h: toFloat(headers, "anthropic-ratelimit-unified-5h-utilization"),
    utilization_7d: toFloat(headers, "anthropic-ratelimit-unified-7d-utilization"),
    utilization_7d_sonnet: toFloat(headers, "anthropic-ratelimit-unified-7d_sonnet-utilization"),
    utilization_overage: toFloat(headers, "anthropic-ratelimit-unified-overage-utilization"),
    reset_5h: toInt(headers, "anthropic-ratelimit-unified-5h-reset"),
    reset_7d: toInt(headers, "anthropic-ratelimit-unified-7d-reset"),
    reset_7d_sonnet: toInt(headers, "anthropic-ratelimit-unified-7d_sonnet-reset"),
    reset_overage: toInt(headers, "anthropic-ratelimit-unified-overage-reset"),
    status_5h: toStr(headers, "anthropic-ratelimit-unified-5h-status"),
    status_7d: toStr(headers, "anthropic-ratelimit-unified-7d-status"),
    status_7d_sonnet: toStr(headers, "anthropic-ratelimit-unified-7d_sonnet-status"),
    status_overage: toStr(headers, "anthropic-ratelimit-unified-overage-status"),
    representative_claim: toStr(headers, "anthropic-ratelimit-unified-representative-claim"),
    fallback_percentage: toFloat(headers, "anthropic-ratelimit-unified-fallback-percentage"),
  };
}

function loadClaudeOauthTokenFromFile(): string | null {
  const credentialsPath = path.join(homedir(), ".claude", ".credentials.json");
  if (!existsSync(credentialsPath)) {
    return null;
  }

  try {
    const raw = readFileSync(credentialsPath, "utf-8");
    const data = JSON.parse(raw) as {
      claudeAiOauth?: {
        accessToken?: string;
      };
    };
    const token = data.claudeAiOauth?.accessToken?.trim();
    return token || null;
  } catch {
    return null;
  }
}

function resolveAnthropicToken(explicitToken?: string): { token: string | null; source: string } {
  if (explicitToken?.trim()) {
    return { token: explicitToken.trim(), source: "env.ANTHROPIC_API_KEY" };
  }

  const envApiKey = process.env.ANTHROPIC_API_KEY?.trim();
  if (envApiKey) {
    return { token: envApiKey, source: "env.ANTHROPIC_API_KEY" };
  }

  const envOauth = process.env.CLAUDE_CODE_OAUTH_TOKEN?.trim();
  if (envOauth) {
    return { token: envOauth, source: "env.CLAUDE_CODE_OAUTH_TOKEN" };
  }

  const fileOauth = loadClaudeOauthTokenFromFile();
  if (fileOauth) {
    return { token: fileOauth, source: "~/.claude/.credentials.json" };
  }

  return { token: null, source: "none" };
}

export async function checkUsage(explicitToken?: string): Promise<Record<string, unknown>> {
  const { token, source } = resolveAnthropicToken(explicitToken);
  if (!token) {
    return {
      status: "error",
      error: "No Anthropic API/OAuth token available for usage check.",
    };
  }

  const response = await fetch(ANTHROPIC_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "anthropic-version": ANTHROPIC_VERSION,
      "anthropic-beta": ANTHROPIC_BETA,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: PROBE_MODEL,
      max_tokens: 1,
      messages: [{ role: "user", content: "." }],
    }),
  });

  const headerData = extractUsageHeaders(response.headers);
  if (!response.ok && response.status !== 429) {
    const body = await response.text();
    return {
      status: "error",
      token_source: source,
      http_status: response.status,
      error: body.slice(0, 300),
      usage: headerData,
    };
  }

  return {
    status: "ok",
    token_source: source,
    http_status: response.status,
    usage: headerData,
  };
}
