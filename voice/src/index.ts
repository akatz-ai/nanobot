import { Client, GatewayIntentBits } from "discord.js";
import type { VoiceConnection } from "@discordjs/voice";

import { attachRealtimeEventRouter } from "./realtime/events.js";
import { RealtimeCloseInfo, RealtimeSession } from "./realtime/session.js";
import { ToolExecutor } from "./tools/index.js";
import { loadEnvConfig } from "./utils/config.js";
import { Logger, parseLogLevel } from "./utils/logger.js";
import { VoiceRuntimeTelemetry } from "./utils/telemetry.js";
import { OwnerVoiceConnectionController } from "./voice/connection.js";
import { DiscordVoicePlayer } from "./voice/player.js";
import { DiscordVoiceReceiver } from "./voice/receiver.js";

interface ActiveRuntime {
  session: RealtimeSession;
  player: DiscordVoicePlayer;
  receiver: DiscordVoiceReceiver;
  detachRouter: () => void;
  onSessionClose: (info: RealtimeCloseInfo) => void;
  idleTimer: NodeJS.Timeout | null;
  reconnectPromise: Promise<void> | null;
  pendingAudio: Buffer[];
  telemetry: VoiceRuntimeTelemetry;
}

const MAX_PENDING_AUDIO_CHUNKS = 600;

async function main(): Promise<void> {
  const env = loadEnvConfig();
  const logger = new Logger("voice-agent", parseLogLevel(env.logLevel));

  const client = new Client({
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
  });

  const tools = new ToolExecutor({
    discordBotToken: env.discordBotToken,
    nanobotConfigPath: env.nanobotConfigPath,
    anthropicApiKey: env.anthropicApiKey,
    logger,
  });

  let active: ActiveRuntime | null = null;
  let controller: OwnerVoiceConnectionController | null = null;
  let shuttingDown = false;

  const stopRuntime = async (reason: string): Promise<void> => {
    const runtime = active;
    if (!runtime) {
      return;
    }
    active = null;

    logger.info("Stopping active runtime", { reason });

    if (runtime.idleTimer) {
      clearTimeout(runtime.idleTimer);
      runtime.idleTimer = null;
    }

    runtime.receiver.stop();
    runtime.detachRouter();
    runtime.session.off("close", runtime.onSessionClose);
    runtime.session.close();
    runtime.player.stop();
    runtime.pendingAudio.length = 0;
    runtime.telemetry.stop();
  };

  const startRuntime = async (connection: VoiceConnection): Promise<void> => {
    await stopRuntime("replace_existing_runtime");
    logger.info("Starting active runtime");
    const telemetry = new VoiceRuntimeTelemetry(logger);

    const session = new RealtimeSession({
      apiKey: env.openAiApiKey,
      model: env.openAiRealtimeModel,
      voice: env.openAiRealtimeVoice,
      logger,
    });
    session.on("open", () => {
      telemetry.recordConnectOpen();
    });
    session.on("connect_error", (error) => {
      telemetry.recordConnectFailure(String(error));
    });
    session.on("error", () => {
      telemetry.recordRealtimeError();
    });
    session.on("input_audio_skipped", (payload) => {
      const info = payload as { reason?: string };
      if (info.reason === "not_open") {
        telemetry.recordInputDropped("not_open");
      }
    });

    const player = new DiscordVoicePlayer(connection, logger, {
      onPlaybackStart: () => {
        telemetry.recordPlaybackStart();
      },
      onPlaybackChunk: (bytes) => {
        telemetry.recordPlaybackChunk(bytes);
      },
    });
    const pendingAudio: Buffer[] = [];

    const runtime: ActiveRuntime = {
      session,
      player,
      receiver: null as unknown as DiscordVoiceReceiver,
      detachRouter: () => undefined,
      onSessionClose: () => undefined,
      idleTimer: null,
      reconnectPromise: null,
      pendingAudio,
      telemetry,
    };

    const resetIdleTimer = (): void => {
      if (env.idleTimeoutMs <= 0) {
        return;
      }
      if (runtime.idleTimer) {
        clearTimeout(runtime.idleTimer);
      }
      runtime.idleTimer = setTimeout(() => {
        logger.info("Idle timeout reached, closing realtime session");
        runtime.telemetry.recordIdleTimeoutClose();
        session.close();
      }, env.idleTimeoutMs);
      runtime.idleTimer.unref();
    };

    const flushPendingAudio = (): void => {
      while (runtime.pendingAudio.length > 0 && session.isOpen()) {
        const chunk = runtime.pendingAudio.shift();
        if (chunk) {
          const sent = session.appendInputAudio(chunk);
          if (sent) {
            runtime.telemetry.recordInputSent(chunk.length, "flush");
          } else {
            runtime.telemetry.recordInputDropped("not_open");
          }
          runtime.telemetry.recordQueueDepth(runtime.pendingAudio.length);
        }
      }
    };

    const ensureSessionConnected = async (): Promise<void> => {
      if (session.isOpen()) {
        return;
      }
      if (runtime.reconnectPromise) {
        return runtime.reconnectPromise;
      }
      runtime.telemetry.recordConnectStart("ensure_session_connected");
      runtime.reconnectPromise = session
        .connect()
        .then(() => {
          flushPendingAudio();
        })
        .catch((error) => {
          logger.error("Realtime reconnect failed", { error: String(error) });
        })
        .finally(() => {
          runtime.reconnectPromise = null;
        });
      return runtime.reconnectPromise;
    };

    const receiver = new DiscordVoiceReceiver({
      connection,
      ownerUserId: env.discordOwnerUserId,
      logger,
      onPcm24kAudio: (pcm24Mono) => {
        runtime.telemetry.recordInputReceived(pcm24Mono.length);
        resetIdleTimer();
        if (session.isOpen()) {
          const sent = session.appendInputAudio(pcm24Mono);
          if (sent) {
            runtime.telemetry.recordInputSent(pcm24Mono.length, "live");
          } else {
            runtime.telemetry.recordInputDropped("not_open");
          }
          return;
        }

        if (runtime.pendingAudio.length >= MAX_PENDING_AUDIO_CHUNKS) {
          runtime.pendingAudio.shift();
          runtime.telemetry.recordInputDropped("queue_full");
        }
        runtime.pendingAudio.push(pcm24Mono);
        runtime.telemetry.recordInputQueued(runtime.pendingAudio.length);
        void ensureSessionConnected();
      },
    });

    runtime.receiver = receiver;
    runtime.detachRouter = attachRealtimeEventRouter({
      session,
      player,
      tools,
      logger,
      onSpeechStarted: () => {
        runtime.telemetry.recordSpeechStarted();
      },
      onSpeechStopped: () => {
        runtime.telemetry.recordSpeechStopped();
      },
      onResponseCreated: () => {
        runtime.telemetry.recordResponseCreated();
      },
      onResponseDone: () => {
        runtime.telemetry.recordResponseDone();
      },
      onResponseAudioDelta: (pcmBytes) => {
        runtime.telemetry.recordResponseAudioDelta(pcmBytes);
      },
      onToolCall: () => {
        runtime.telemetry.recordToolCall();
      },
      onRealtimeError: () => {
        runtime.telemetry.recordRealtimeError();
      },
    });
    runtime.onSessionClose = (info) => {
      runtime.telemetry.recordSessionClose(info.expected, info.code, info.reason);
      if (!info.expected && !shuttingDown) {
        logger.warn("Realtime session closed unexpectedly, reconnecting", {
          code: info.code,
          reason: info.reason,
          expected: info.expected,
        });
        void ensureSessionConnected();
      }
    };

    session.on("close", runtime.onSessionClose);

    active = runtime;

    receiver.start();
    await ensureSessionConnected();
    resetIdleTimer();
  };

  const shutdown = async (signal: string): Promise<void> => {
    if (shuttingDown) {
      return;
    }
    shuttingDown = true;
    logger.info("Shutting down voice agent", { signal });

    try {
      if (controller) {
        await controller.stop();
      }
      await stopRuntime("process_shutdown");
      tools.stop();
      client.destroy();
    } finally {
      process.exit(0);
    }
  };

  client.once("ready", async () => {
    logger.info("Discord client ready", { user: client.user?.tag });

    controller = new OwnerVoiceConnectionController({
      client,
      guildId: env.discordGuildId,
      voiceChannelId: env.discordVoiceChannelId,
      ownerUserId: env.discordOwnerUserId,
      logger,
      onConnected: startRuntime,
      onDisconnected: stopRuntime,
    });

    await controller.start();
  });

  client.on("error", (error) => {
    logger.error("Discord client error", { error: String(error) });
  });

  client.on("warn", (message) => {
    logger.warn("Discord client warning", { message });
  });

  process.on("SIGINT", () => {
    void shutdown("SIGINT");
  });
  process.on("SIGTERM", () => {
    void shutdown("SIGTERM");
  });

  await client.login(env.discordBotToken);
}

main().catch((error) => {
  console.error(
    JSON.stringify({
      ts: new Date().toISOString(),
      level: "error",
      scope: "voice-agent",
      message: "Fatal startup error",
      meta: { error: String(error) },
    }),
  );
  process.exit(1);
});
