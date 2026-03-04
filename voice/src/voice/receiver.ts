import opus from "@discordjs/opus";
const { OpusEncoder } = opus;
import { EndBehaviorType, VoiceConnection } from "@discordjs/voice";
import type { AudioReceiveStream } from "@discordjs/voice";

import { downsample48kStereoTo24kMono } from "../utils/audio.js";
import { Logger } from "../utils/logger.js";

interface ReceiverOptions {
  connection: VoiceConnection;
  ownerUserId: string;
  logger: Logger;
  onPcm24kAudio: (pcm24Mono: Buffer) => void;
}

export class DiscordVoiceReceiver {
  private readonly decoder = new OpusEncoder(48000, 2);
  private readonly logger: Logger;
  private stream: AudioReceiveStream | null = null;
  private stopped = true;
  private packetCount = 0;
  private subscribing = false;

  constructor(private readonly options: ReceiverOptions) {
    this.logger = options.logger.child("receiver");
  }

  start(): void {
    if (!this.stopped) {
      return;
    }
    this.stopped = false;

    // Listen for speaking events to trigger subscription
    const receiver = this.options.connection.receiver;
    receiver.speaking.on("start", (userId) => {
      if (this.stopped || userId !== this.options.ownerUserId) {
        return;
      }
      // Only subscribe if we don't already have an active stream
      if (!this.stream && !this.subscribing) {
        this.subscribe();
      }
    });

    this.logger.info("Receiver started, listening for owner speaking events", {
      ownerUserId: this.options.ownerUserId,
    });
  }

  stop(): void {
    this.stopped = true;
    if (this.stream) {
      this.stream.removeAllListeners();
      this.stream.destroy();
      this.stream = null;
    }
  }

  private subscribe(): void {
    if (this.stopped || this.subscribing) {
      return;
    }
    this.subscribing = true;

    this.stream = this.options.connection.receiver.subscribe(this.options.ownerUserId, {
      end: {
        behavior: EndBehaviorType.AfterSilence,
        duration: 5000, // Keep stream alive for 5s of silence
      },
    });

    this.subscribing = false;
    this.logger.info("Subscribed to owner voice stream");

    this.stream.on("data", (opusPacket: Buffer) => {
      try {
        const pcm48Stereo = this.decoder.decode(opusPacket);
        const pcm24Mono = downsample48kStereoTo24kMono(pcm48Stereo);
        if (pcm24Mono.length > 0) {
          this.packetCount++;
          if (this.packetCount === 1 || this.packetCount % 500 === 0) {
            this.logger.info("Audio packets forwarded to OpenAI", { count: this.packetCount });
          }
          this.options.onPcm24kAudio(pcm24Mono);
        }
      } catch (error) {
        this.logger.warn("Failed decoding voice packet", { error: String(error) });
      }
    });

    this.stream.on("error", (error) => {
      this.logger.warn("Voice receiver stream error", { error: String(error) });
    });

    this.stream.on("close", () => {
      this.logger.info("Owner voice stream ended (silence timeout), will re-subscribe on next speech");
      this.stream = null;
      // Don't re-subscribe here — wait for next speaking.start event
    });
  }
}
