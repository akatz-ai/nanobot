import { Logger } from "./logger.js";

type SendSource = "live" | "flush";
type DropReason = "queue_full" | "not_open";

interface TelemetryCounters {
  connectAttempts: number;
  connectFailures: number;
  connectSuccesses: number;
  wsClosesExpected: number;
  wsClosesUnexpected: number;
  realtimeErrors: number;
  idleTimeoutCloses: number;

  inputChunksReceived: number;
  inputBytesReceived: number;
  inputChunksQueued: number;
  inputChunksDropped: number;
  inputChunksSent: number;
  inputChunksSentLive: number;
  inputChunksSentFlush: number;
  inputBytesSent: number;

  responseCreated: number;
  responseDone: number;
  responseAudioDeltaChunks: number;
  responseAudioDeltaBytes: number;
  speechStarted: number;
  speechStopped: number;
  toolCalls: number;

  playbackStarts: number;
  playbackChunksWritten: number;
  playbackBytesWritten: number;
}

interface TelemetryMilestones {
  runtimeStartedAt?: number;
  firstConnectStartAt?: number;
  firstConnectOpenAt?: number;
  firstInputChunkReceivedAt?: number;
  firstInputChunkSentAt?: number;
  firstSpeechStartedAt?: number;
  firstResponseCreatedAt?: number;
  firstResponseAudioDeltaAt?: number;
  firstPlaybackStartAt?: number;
}

function durationMs(start?: number, end?: number): number | null {
  if (!start || !end || end < start) {
    return null;
  }
  return end - start;
}

export class VoiceRuntimeTelemetry {
  private readonly logger: Logger;
  private readonly startedAt = Date.now();
  private readonly counters: TelemetryCounters = {
    connectAttempts: 0,
    connectFailures: 0,
    connectSuccesses: 0,
    wsClosesExpected: 0,
    wsClosesUnexpected: 0,
    realtimeErrors: 0,
    idleTimeoutCloses: 0,

    inputChunksReceived: 0,
    inputBytesReceived: 0,
    inputChunksQueued: 0,
    inputChunksDropped: 0,
    inputChunksSent: 0,
    inputChunksSentLive: 0,
    inputChunksSentFlush: 0,
    inputBytesSent: 0,

    responseCreated: 0,
    responseDone: 0,
    responseAudioDeltaChunks: 0,
    responseAudioDeltaBytes: 0,
    speechStarted: 0,
    speechStopped: 0,
    toolCalls: 0,

    playbackStarts: 0,
    playbackChunksWritten: 0,
    playbackBytesWritten: 0,
  };
  private readonly milestones: TelemetryMilestones = {};
  private maxPendingQueueDepth = 0;
  private intervalTimer: NodeJS.Timeout | null = null;

  constructor(logger: Logger, intervalMs = 30_000) {
    this.logger = logger.child("telemetry");
    this.markRuntimeStarted();
    if (intervalMs > 0) {
      this.intervalTimer = setInterval(() => {
        this.logSummary("interval");
      }, intervalMs);
      this.intervalTimer.unref();
    }
  }

  stop(): void {
    if (this.intervalTimer) {
      clearInterval(this.intervalTimer);
      this.intervalTimer = null;
    }
    this.logSummary("shutdown");
  }

  markRuntimeStarted(): void {
    this.markOnce("runtimeStartedAt");
    this.logger.info("Voice telemetry initialized");
  }

  recordConnectStart(reason: string): void {
    this.counters.connectAttempts += 1;
    this.markOnce("firstConnectStartAt");
    this.logger.info("Realtime connect attempt", {
      attempt: this.counters.connectAttempts,
      reason,
    });
  }

  recordConnectOpen(): void {
    this.counters.connectSuccesses += 1;
    this.markOnce("firstConnectOpenAt");
    if (this.counters.connectSuccesses === 1) {
      this.logger.info("Realtime connect established", {
        startup_ms: durationMs(this.milestones.runtimeStartedAt, this.milestones.firstConnectOpenAt),
      });
    }
  }

  recordConnectFailure(error: string): void {
    this.counters.connectFailures += 1;
    this.logger.warn("Realtime connect failed", {
      failures: this.counters.connectFailures,
      error,
    });
  }

  recordSessionClose(expected: boolean, code: number, reason: string): void {
    if (expected) {
      this.counters.wsClosesExpected += 1;
    } else {
      this.counters.wsClosesUnexpected += 1;
    }
    this.logger.info("Realtime session close observed", {
      expected,
      code,
      reason,
      unexpected_close_count: this.counters.wsClosesUnexpected,
    });
  }

  recordIdleTimeoutClose(): void {
    this.counters.idleTimeoutCloses += 1;
  }

  recordRealtimeError(): void {
    this.counters.realtimeErrors += 1;
  }

  recordInputReceived(bytes: number): void {
    this.counters.inputChunksReceived += 1;
    this.counters.inputBytesReceived += bytes;
    this.markOnce("firstInputChunkReceivedAt");
  }

  recordInputQueued(queueDepth: number): void {
    this.counters.inputChunksQueued += 1;
    if (queueDepth > this.maxPendingQueueDepth) {
      this.maxPendingQueueDepth = queueDepth;
    }
  }

  recordQueueDepth(queueDepth: number): void {
    if (queueDepth > this.maxPendingQueueDepth) {
      this.maxPendingQueueDepth = queueDepth;
    }
  }

  recordInputDropped(reason: DropReason): void {
    this.counters.inputChunksDropped += 1;
    if (this.counters.inputChunksDropped === 1 || this.counters.inputChunksDropped % 25 === 0) {
      this.logger.warn("Input chunk dropped", {
        reason,
        dropped: this.counters.inputChunksDropped,
      });
    }
  }

  recordInputSent(bytes: number, source: SendSource): void {
    this.counters.inputChunksSent += 1;
    this.counters.inputBytesSent += bytes;
    if (source === "live") {
      this.counters.inputChunksSentLive += 1;
    } else {
      this.counters.inputChunksSentFlush += 1;
    }
    this.markOnce("firstInputChunkSentAt");
  }

  recordSpeechStarted(): void {
    this.counters.speechStarted += 1;
    this.markOnce("firstSpeechStartedAt");
  }

  recordSpeechStopped(): void {
    this.counters.speechStopped += 1;
  }

  recordResponseCreated(): void {
    this.counters.responseCreated += 1;
    this.markOnce("firstResponseCreatedAt");
  }

  recordResponseDone(): void {
    this.counters.responseDone += 1;
  }

  recordResponseAudioDelta(bytes: number): void {
    this.counters.responseAudioDeltaChunks += 1;
    this.counters.responseAudioDeltaBytes += bytes;
    this.markOnce("firstResponseAudioDeltaAt");
  }

  recordToolCall(): void {
    this.counters.toolCalls += 1;
  }

  recordPlaybackStart(): void {
    this.counters.playbackStarts += 1;
    this.markOnce("firstPlaybackStartAt");
  }

  recordPlaybackChunk(bytes: number): void {
    this.counters.playbackChunksWritten += 1;
    this.counters.playbackBytesWritten += bytes;
  }

  private markOnce(key: keyof TelemetryMilestones): void {
    if (this.milestones[key]) {
      return;
    }
    this.milestones[key] = Date.now();
  }

  private logSummary(trigger: "interval" | "shutdown"): void {
    const now = Date.now();
    this.logger.info("Voice telemetry summary", {
      trigger,
      uptime_ms: now - this.startedAt,
      counters: this.counters,
      max_pending_audio_queue_depth: this.maxPendingQueueDepth,
      timings_ms: {
        runtime_to_ws_open: durationMs(this.milestones.runtimeStartedAt, this.milestones.firstConnectOpenAt),
        ws_open_to_first_input_sent: durationMs(this.milestones.firstConnectOpenAt, this.milestones.firstInputChunkSentAt),
        first_input_to_response_created: durationMs(this.milestones.firstInputChunkSentAt, this.milestones.firstResponseCreatedAt),
        response_created_to_first_audio_delta: durationMs(this.milestones.firstResponseCreatedAt, this.milestones.firstResponseAudioDeltaAt),
        first_audio_delta_to_playback_start: durationMs(this.milestones.firstResponseAudioDeltaAt, this.milestones.firstPlaybackStartAt),
      },
    });
  }
}
