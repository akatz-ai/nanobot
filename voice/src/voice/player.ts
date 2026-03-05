import {
  AudioPlayer,
  NoSubscriberBehavior,
  StreamType,
  VoiceConnection,
  createAudioPlayer,
  createAudioResource,
} from "@discordjs/voice";
import { PassThrough } from "node:stream";

import { upsample24kMonoTo48kMono, monoToStereo } from "../utils/audio.js";
import { Logger } from "../utils/logger.js";

interface DiscordVoicePlayerHooks {
  onPlaybackStart?: () => void;
  onPlaybackChunk?: (bytes: number) => void;
  onPlaybackFinalize?: (totalChunks: number) => void;
}

export class DiscordVoicePlayer {
  private readonly audioPlayer: AudioPlayer;
  private pcmStream: PassThrough;
  private readonly connection: VoiceConnection;
  private destroyed = false;
  private readonly logger: Logger;
  private chunkCount = 0;
  private resourceActive = false;
  private inPlayingState = false;

  constructor(
    connection: VoiceConnection,
    logger: Logger,
    private readonly hooks?: DiscordVoicePlayerHooks,
  ) {
    this.logger = logger.child("player");
    this.connection = connection;
    this.audioPlayer = createAudioPlayer({
      behaviors: { noSubscriber: NoSubscriberBehavior.Play },
    });
    this.pcmStream = new PassThrough();
    connection.subscribe(this.audioPlayer);

    this.audioPlayer.on("stateChange", (oldState, newState) => {
      this.logger.info("Audio player state change", { from: oldState.status, to: newState.status });
      const nowPlaying = newState.status === "playing";
      if (!this.inPlayingState && nowPlaying) {
        this.hooks?.onPlaybackStart?.();
      }
      this.inPlayingState = nowPlaying;
    });

    this.audioPlayer.on("error", (error) => {
      this.logger.error("Audio player error", { error: String(error) });
    });
  }

  pushPcm24kMono(pcm24Mono: Buffer): void {
    if (this.destroyed || pcm24Mono.length === 0) {
      return;
    }

    // Start a new resource if not active
    if (!this.resourceActive) {
      this.pcmStream = new PassThrough();
      const resource = createAudioResource(this.pcmStream, {
        inputType: StreamType.Raw, // s16le, 48kHz, 2ch
      });
      this.audioPlayer.play(resource);
      this.resourceActive = true;
      this.logger.info("Started new audio resource for playback");
    }

    // Upsample 24kHz mono → 48kHz stereo s16le (what discord expects)
    const pcm48Mono = upsample24kMonoTo48kMono(pcm24Mono);
    const pcm48Stereo = monoToStereo(pcm48Mono);
    this.pcmStream.write(pcm48Stereo);
    this.hooks?.onPlaybackChunk?.(pcm48Stereo.length);

    this.chunkCount++;
    if (this.chunkCount === 1 || this.chunkCount % 50 === 0) {
      this.logger.info("PCM chunks written to player", { count: this.chunkCount, bytes: pcm48Stereo.length });
    }
  }

  finalizeResponse(): void {
    if (this.destroyed || !this.resourceActive) {
      return;
    }
    this.logger.info("Finalizing response audio", { totalChunks: this.chunkCount });
    this.hooks?.onPlaybackFinalize?.(this.chunkCount);
    this.pcmStream.end();
    this.resourceActive = false;
    this.chunkCount = 0;
  }

  stop(): void {
    if (this.destroyed) {
      return;
    }
    this.destroyed = true;
    this.audioPlayer.stop(true);
    if (this.resourceActive) {
      this.pcmStream.end();
      this.resourceActive = false;
    }
  }
}
