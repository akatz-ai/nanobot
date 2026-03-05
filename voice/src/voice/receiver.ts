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
  private readonly onSpeakingStartBound: (userId: string) => void;

  constructor(private readonly options: ReceiverOptions) {
    this.logger = options.logger.child("receiver");
    this.onSpeakingStartBound = (userId) => {
      if (this.stopped || userId !== this.options.ownerUserId) {
        return;
      }
      if (!this.stream && !this.subscribing) {
        this.subscribe();
      }
    };
  }

  start(): void {
    if (!this.stopped) {
      return;
    }
    this.stopped = false;

    // Listen for speaking events to trigger subscription
    const receiver = this.options.connection.receiver;
    receiver.speaking.on("start", this.onSpeakingStartBound);

    this.logger.info("Receiver started, listening for owner speaking events", {
      ownerUserId: this.options.ownerUserId,
    });

    // Subscribe immediately so speech that starts during realtime connect is captured.
    this.subscribe();
  }

  stop(): void {
    this.stopped = true;
    this.options.connection.receiver.speaking.off("start", this.onSpeakingStartBound);
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
    if (this.stream) {
      return;
    }
    this.subscribing = true;

    this.stream = this.options.connection.receiver.subscribe(this.options.ownerUserId, {
      end: {
        behavior: EndBehaviorType.Manual,
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
      this.logger.info("Owner voice stream closed");
      this.stream = null;
      if (!this.stopped) {
        setTimeout(() => {
          this.subscribe();
        }, 100).unref();
      }
    });
  }
}
