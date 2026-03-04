import { EventEmitter } from "node:events";
import WebSocket from "ws";

import { encodeBase64Pcm } from "../utils/audio.js";
import { Logger } from "../utils/logger.js";
import { buildSessionUpdate } from "./config.js";

interface RealtimeSessionOptions {
  apiKey: string;
  model: string;
  voice: string;
  logger: Logger;
}

export interface RealtimeCloseInfo {
  code: number;
  reason: string;
  expected: boolean;
}

export class RealtimeSession extends EventEmitter {
  private ws: WebSocket | null = null;
  private manualClose = false;
  private connectPromise: Promise<void> | null = null;
  private readonly logger: Logger;

  constructor(private readonly options: RealtimeSessionOptions) {
    super();
    this.logger = options.logger.child("realtime");
  }

  async connect(): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }
    if (this.connectPromise) {
      return this.connectPromise;
    }

    const url = `wss://api.openai.com/v1/realtime?model=${encodeURIComponent(this.options.model)}`;
    this.manualClose = false;

    this.connectPromise = new Promise<void>((resolve, reject) => {
      const ws = new WebSocket(url, {
        headers: {
          Authorization: `Bearer ${this.options.apiKey}`,
        },
      });

      this.ws = ws;
      let settled = false;

      ws.once("open", () => {
        settled = true;
        this.logger.info("Realtime websocket connected");
        const sessionUpdate = buildSessionUpdate(this.options.model, this.options.voice);
        this.logger.info("Sending session.update", { payload: JSON.stringify(sessionUpdate) });
        this.send(sessionUpdate);
        this.emit("open");
        resolve();
      });

      ws.once("error", (error) => {
        if (!settled) {
          settled = true;
          reject(error);
        }
      });

      ws.on("message", (data) => {
        const parsed = this.parseMessage(data);
        if (!parsed) {
          return;
        }
        this.emit("event", parsed);
      });

      ws.on("close", (code, reasonBuffer) => {
        const reason = reasonBuffer.toString("utf-8");
        const info: RealtimeCloseInfo = {
          code,
          reason,
          expected: this.manualClose,
        };
        this.logger.info("Realtime websocket closed", {
          code: info.code,
          reason: info.reason,
          expected: info.expected,
        });
        this.ws = null;
        this.connectPromise = null;
        this.emit("close", info);
      });

      ws.on("error", (error) => {
        this.logger.error("Realtime websocket error", { error: String(error) });
        this.emit("error", error);
      });
    }).finally(() => {
      this.connectPromise = null;
    });

    const pending = this.connectPromise;
    if (!pending) {
      return;
    }
    return pending;
  }

  isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  close(): void {
    this.manualClose = true;
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) {
      this.ws.close(1000, "shutdown");
    }
    this.ws = null;
  }

  appendInputAudio(pcm24Mono: Buffer): void {
    if (!this.isOpen() || pcm24Mono.length === 0) {
      return;
    }

    this.send({
      type: "input_audio_buffer.append",
      audio: encodeBase64Pcm(pcm24Mono),
    });
  }

  sendToolResult(callId: string, output: unknown): void {
    const outputString = JSON.stringify(output);
    this.send({
      type: "conversation.item.create",
      item: {
        type: "function_call_output",
        call_id: callId,
        output: outputString,
      },
    });
    this.send({ type: "response.create" });
  }

  send(event: Record<string, unknown>): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    this.ws.send(JSON.stringify(event));
  }

  private parseMessage(data: WebSocket.RawData): Record<string, unknown> | null {
    try {
      let text = "";
      if (typeof data === "string") {
        text = data;
      } else if (Buffer.isBuffer(data)) {
        text = data.toString("utf-8");
      } else if (data instanceof ArrayBuffer) {
        text = Buffer.from(data).toString("utf-8");
      } else if (Array.isArray(data)) {
        text = Buffer.concat(data).toString("utf-8");
      }
      return JSON.parse(text) as Record<string, unknown>;
    } catch (error) {
      this.logger.warn("Failed to parse realtime message", { error: String(error) });
      return null;
    }
  }
}
