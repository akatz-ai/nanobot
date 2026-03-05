export interface RealtimeToolDefinition {
  type: "function";
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: Record<string, unknown>;
    required?: string[];
  };
}

const AGENT_ENUM = ["general", "devius", "recon", "researcher", "comfygit-dev", "forge", "nanobot-dev", "atlas", "assistant"];

export const VOICE_AGENT_INSTRUCTIONS = `
You are the voice interface for Alex's AI agent system in Discord.
Always respond in English. Keep your responses short and conversational: no more than 2-3 sentences unless asked.

Your jobs:
1) Check what agents are doing by reading their channels.
2) Queue messages to agents and confirm before sending.
3) Report system/usage status.
4) Relay concise updates between Alex and the agent fleet.

Rules:
- Always speak in English, regardless of what language you think you hear.
- Never fabricate tool results.
- Always use send_message_to_agent first. It queues but does not send.
- Read the preview aloud, then call confirm_action only after explicit confirmation.
- If Alex says cancel/never mind, call cancel_action.
- If search_knowledge_base or search_agent_memory is requested, call the tool and state it is not implemented yet.
`.trim();

export const TOOL_DEFINITIONS: RealtimeToolDefinition[] = [
  {
    type: "function",
    name: "read_agent_channel",
    description: "Read recent messages from an agent's Discord text channel.",
    parameters: {
      type: "object",
      properties: {
        agent: { type: "string", enum: AGENT_ENUM, description: "Agent name" },
        count: { type: "number", description: "Number of messages (default 5, max 20)" },
      },
      required: ["agent"],
    },
  },
  {
    type: "function",
    name: "send_message_to_agent",
    description:
      "Queue a message to an agent channel. This does NOT send immediately and must be followed by confirm_action.",
    parameters: {
      type: "object",
      properties: {
        agent: { type: "string", enum: AGENT_ENUM, description: "Agent name" },
        message: { type: "string", description: "Message content" },
      },
      required: ["agent", "message"],
    },
  },
  {
    type: "function",
    name: "confirm_action",
    description: "Confirm and execute a pending action by action_id.",
    parameters: {
      type: "object",
      properties: {
        action_id: { type: "string", description: "Pending action ID" },
      },
      required: ["action_id"],
    },
  },
  {
    type: "function",
    name: "cancel_action",
    description: "Cancel a pending action by action_id.",
    parameters: {
      type: "object",
      properties: {
        action_id: { type: "string", description: "Pending action ID" },
      },
      required: ["action_id"],
    },
  },
  {
    type: "function",
    name: "check_system_status",
    description: "Check basic system health: uptime, disk, memory, load.",
    parameters: { type: "object", properties: {} },
  },
  {
    type: "function",
    name: "check_usage",
    description: "Check Anthropic API unified rate limit utilization from response headers.",
    parameters: { type: "object", properties: {} },
  },
  {
    type: "function",
    name: "list_agents",
    description: "List configured nanobot agents and mapped Discord channels.",
    parameters: { type: "object", properties: {} },
  },
  {
    type: "function",
    name: "search_knowledge_base",
    description: "Search the shared knowledge base for a query (not implemented in this phase).",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
      },
      required: ["query"],
    },
  },
  {
    type: "function",
    name: "search_agent_memory",
    description: "Search a specific agent memory for a query (not implemented in this phase).",
    parameters: {
      type: "object",
      properties: {
        agent: { type: "string", enum: AGENT_ENUM, description: "Agent name" },
        query: { type: "string", description: "Search query" },
      },
      required: ["agent", "query"],
    },
  },
];

export function buildSessionUpdate(model: string, voice: string): Record<string, unknown> {
  return {
    type: "session.update",
    session: {
      type: "realtime",
      model,
      output_modalities: ["audio"],
      instructions: VOICE_AGENT_INSTRUCTIONS,
      audio: {
        input: {
          turn_detection: {
            type: "server_vad",
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500,
          },
        },
        output: {
          voice,
        },
      },
      tools: TOOL_DEFINITIONS,
      tool_choice: "auto",
    },
  };
}
