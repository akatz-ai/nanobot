import { decodeBase64Pcm } from "../utils/audio.js";
import { Logger } from "../utils/logger.js";
import { RealtimeSession } from "./session.js";
import { DiscordVoicePlayer } from "../voice/player.js";
import { ToolExecutor } from "../tools/index.js";

interface RealtimeEventRouterOptions {
  session: RealtimeSession;
  player: DiscordVoicePlayer;
  tools: ToolExecutor;
  logger: Logger;
  onSpeechStarted?: () => void;
  onSpeechStopped?: () => void;
  onResponseCreated?: () => void;
  onResponseDone?: () => void;
  onResponseAudioDelta?: (pcmBytes: number) => void;
  onToolCall?: (toolName: string, callId: string) => void;
  onRealtimeError?: () => void;
}

function asObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value as Record<string, unknown>;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

interface FunctionCallPayload {
  callId: string;
  name: string;
  rawArguments: string;
}

function extractFunctionCall(event: Record<string, unknown>): FunctionCallPayload | null {
  const type = asString(event.type);

  if (type === "response.function_call_arguments.done") {
    return {
      callId: asString(event.call_id),
      name: asString(event.name),
      rawArguments: asString(event.arguments),
    };
  }

  if (type === "response.output_item.done") {
    const item = asObject(event.item);
    if (asString(item.type) !== "function_call") {
      return null;
    }
    return {
      callId: asString(item.call_id),
      name: asString(item.name),
      rawArguments: asString(item.arguments),
    };
  }

  return null;
}

function parseArguments(raw: string): unknown {
  if (!raw.trim()) {
    return {};
  }
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return {};
  }
}

export function attachRealtimeEventRouter(options: RealtimeEventRouterOptions): () => void {
  const logger = options.logger.child("events");
  const handledCallIds = new Set<string>();

  const onEvent = (event: Record<string, unknown>): void => {
    void handleEvent(event);
  };

  const handleEvent = async (event: Record<string, unknown>): Promise<void> => {
    const type = asString(event.type);

    if (type === "session.created") {
      logger.info("Realtime session created");
      return;
    }

    if (type === "session.updated") {
      const session = asObject(event.session);
      const tools = session.tools as unknown[];
      logger.info("Realtime session updated successfully", {
        toolCount: Array.isArray(tools) ? tools.length : 0,
        voice: asString(asObject(asObject(session.audio).output).voice),
      });
      return;
    }

    if (type === "input_audio_buffer.speech_started") {
      options.onSpeechStarted?.();
      logger.info("Speech detected — user is speaking");
      return;
    }

    if (type === "input_audio_buffer.speech_stopped") {
      options.onSpeechStopped?.();
      logger.info("Speech ended — processing");
      return;
    }

    if (type === "input_audio_buffer.committed") {
      logger.info("Audio buffer committed");
      return;
    }

    if (type === "response.created") {
      options.onResponseCreated?.();
      logger.info("Response generation started");
      return;
    }

    if (type === "error") {
      options.onRealtimeError?.();
      logger.error("Realtime API error event", { event });
      return;
    }

    if (type === "response.audio.delta" || type === "response.output_audio.delta") {
      const delta = asString(event.delta);
      if (!delta) {
        logger.warn("Got audio delta with empty data");
        return;
      }
      const pcm = decodeBase64Pcm(delta);
      options.onResponseAudioDelta?.(pcm.length);
      logger.info("Audio delta received", { pcmBytes: pcm.length, b64Len: delta.length });
      options.player.pushPcm24kMono(pcm);
      return;
    }

    if (type === "response.audio.done" || type === "response.output_audio.done" || type === "response.done") {
      if (type === "response.done") {
        options.onResponseDone?.();
      }
      options.player.finalizeResponse();
      if (type === "response.done") {
        const response = asObject(event.response);
        const output = response.output as unknown[];
        if (Array.isArray(output) && output.length > 0) {
          for (const item of output) {
            const o = asObject(item);
            logger.info("Response output item", { type: asString(o.type), role: asString(o.role), contentTypes: Array.isArray(o.content) ? (o.content as Array<Record<string,unknown>>).map(c => asString(c.type)) : [] });
          }
        }
        handledCallIds.clear();
      }
      return;
    }

    const call = extractFunctionCall(event);
    if (!call) {
      // Log unhandled event types (skip noisy delta/transcript events)
      if (!type.includes("delta") && !type.includes("transcript")) {
        logger.info("Unhandled realtime event", { type });
      }
      return;
    }

    if (!call.callId || !call.name) {
      logger.warn("Ignoring malformed function call payload", { call });
      return;
    }

    if (handledCallIds.has(call.callId)) {
      return;
    }
    handledCallIds.add(call.callId);

    logger.info("Handling realtime tool call", {
      tool: call.name,
      callId: call.callId,
    });
    options.onToolCall?.(call.name, call.callId);

    const result = await options.tools.execute(call.name, parseArguments(call.rawArguments));
    options.session.sendToolResult(call.callId, result);
  };

  options.session.on("event", onEvent);

  return () => {
    options.session.off("event", onEvent);
  };
}
