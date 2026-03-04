import { ConfirmationGate } from "./confirmationGate.js";
import { listAgents } from "./listAgents.js";
import { readAgentChannel } from "./readChannel.js";
import { queueMessageToAgent, sendDiscordMessage } from "./sendMessage.js";
import { checkSystemStatus } from "./systemStatus.js";
import { loadNanobotConfig, resolveAgent } from "../utils/config.js";
import { checkUsage } from "./usage.js";
import { Logger } from "../utils/logger.js";

interface ToolExecutorOptions {
  discordBotToken: string;
  nanobotConfigPath: string;
  anthropicApiKey?: string;
  logger: Logger;
}

function asObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value as Record<string, unknown>;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function asNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
}

export class ToolExecutor {
  private readonly logger: Logger;
  private readonly gate = new ConfirmationGate(60_000);

  constructor(private readonly options: ToolExecutorOptions) {
    this.logger = options.logger.child("tools");
  }

  stop(): void {
    this.gate.stop();
  }

  async execute(toolName: string, rawArgs: unknown): Promise<Record<string, unknown>> {
    const args = asObject(rawArgs);
    this.logger.info("Executing tool", { toolName });

    try {
      switch (toolName) {
        case "read_agent_channel": {
          const agent = asString(args.agent);
          if (!agent) {
            return { status: "error", error: "Missing required argument: agent" };
          }
          return await readAgentChannel({
            discordBotToken: this.options.discordBotToken,
            nanobotConfigPath: this.options.nanobotConfigPath,
            agent,
            count: asNumber(args.count),
          });
        }

        case "send_message_to_agent": {
          const agent = asString(args.agent);
          const message = asString(args.message);
          if (!agent || !message) {
            return { status: "error", error: "Missing required arguments: agent and message" };
          }
          return queueMessageToAgent({
            gate: this.gate,
            nanobotConfigPath: this.options.nanobotConfigPath,
            agent,
            message,
          });
        }

        case "confirm_action": {
          const actionId = asString(args.action_id);
          if (!actionId) {
            return { status: "error", error: "Missing required argument: action_id" };
          }

          const action = this.gate.confirm(actionId);
          if (!action) {
            return { status: "error", error: "Action not found or expired." };
          }

          // Look up webhook URL for the target agent channel
          const confirmConfig = loadNanobotConfig(this.options.nanobotConfigPath);
          const confirmResolved = resolveAgent(confirmConfig, action.agent);

          const sendResult = await sendDiscordMessage({
            discordBotToken: this.options.discordBotToken,
            channelId: action.channelId,
            content: action.message,
            webhookUrl: confirmResolved?.webhookUrl,
          });

          return {
            status: sendResult.status === "ok" ? "ok" : "error",
            action_id: actionId,
            preview: action.preview,
            result: sendResult,
          };
        }

        case "cancel_action": {
          const actionId = asString(args.action_id);
          if (!actionId) {
            return { status: "error", error: "Missing required argument: action_id" };
          }
          const canceled = this.gate.cancel(actionId);
          if (!canceled) {
            return { status: "error", error: "Action not found." };
          }
          return {
            status: "ok",
            action_id: actionId,
            canceled: true,
            preview: canceled.preview,
          };
        }

        case "check_system_status":
          return await checkSystemStatus();

        case "check_usage":
          return await checkUsage(this.options.anthropicApiKey);

        case "list_agents":
          return listAgents(this.options.nanobotConfigPath);

        case "search_knowledge_base":
        case "search_agent_memory":
          return {
            status: "not_implemented",
            message: `${toolName} is not implemented yet (Phase 3).`,
          };

        default:
          return {
            status: "error",
            error: `Unknown tool: ${toolName}`,
          };
      }
    } catch (error) {
      this.logger.error("Tool execution failed", {
        toolName,
        error: String(error),
      });
      return {
        status: "error",
        error: String(error),
      };
    }
  }
}
