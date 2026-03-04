import type { Client, VoiceState } from "discord.js";
import {
  VoiceConnection,
  VoiceConnectionStatus,
  entersState,
  getVoiceConnection,
  joinVoiceChannel,
} from "@discordjs/voice";

import { Logger } from "../utils/logger.js";

interface ConnectionOptions {
  client: Client;
  guildId: string;
  voiceChannelId: string;
  ownerUserId: string;
  logger: Logger;
  onConnected: (connection: VoiceConnection) => Promise<void> | void;
  onDisconnected: (reason: string) => Promise<void> | void;
}

export class OwnerVoiceConnectionController {
  private readonly logger: Logger;
  private connection: VoiceConnection | null = null;
  private active = false;
  private readonly onVoiceStateUpdateBound: (oldState: VoiceState, newState: VoiceState) => void;

  constructor(private readonly options: ConnectionOptions) {
    this.logger = options.logger.child("connection");
    this.onVoiceStateUpdateBound = (oldState, newState) => {
      void this.onVoiceStateUpdate(oldState, newState);
    };
  }

  async start(): Promise<void> {
    if (this.active) {
      return;
    }
    this.active = true;
    this.options.client.on("voiceStateUpdate", this.onVoiceStateUpdateBound);
    await this.syncToCurrentState();
  }

  async stop(): Promise<void> {
    if (!this.active) {
      return;
    }

    this.active = false;
    this.options.client.off("voiceStateUpdate", this.onVoiceStateUpdateBound);
    await this.leave("controller_stop");
  }

  private async onVoiceStateUpdate(oldState: VoiceState, newState: VoiceState): Promise<void> {
    if (!this.active) {
      return;
    }

    if (oldState.id !== this.options.ownerUserId && newState.id !== this.options.ownerUserId) {
      return;
    }

    const previousInTarget = oldState.channelId === this.options.voiceChannelId;
    const nowInTarget = newState.channelId === this.options.voiceChannelId;

    if (!previousInTarget && nowInTarget) {
      await this.join();
      return;
    }

    if (previousInTarget && !nowInTarget) {
      await this.leave("owner_left_channel");
    }
  }

  private async syncToCurrentState(): Promise<void> {
    const guild = await this.options.client.guilds.fetch(this.options.guildId);
    const member = await guild.members.fetch(this.options.ownerUserId);
    if (member.voice.channelId === this.options.voiceChannelId) {
      await this.join();
      return;
    }
    await this.leave("owner_not_in_channel");
  }

  private async join(): Promise<void> {
    if (this.connection && this.connection.state.status !== VoiceConnectionStatus.Destroyed) {
      return;
    }

    const guild = await this.options.client.guilds.fetch(this.options.guildId);
    const existing = getVoiceConnection(this.options.guildId);
    if (existing && existing.state.status !== VoiceConnectionStatus.Destroyed) {
      this.connection = existing;
      await entersState(existing, VoiceConnectionStatus.Ready, 30_000);
      await this.options.onConnected(existing);
      return;
    }

    this.logger.info("Joining target voice channel", {
      guildId: this.options.guildId,
      channelId: this.options.voiceChannelId,
    });

    const connection = joinVoiceChannel({
      channelId: this.options.voiceChannelId,
      guildId: this.options.guildId,
      adapterCreator: guild.voiceAdapterCreator,
      selfDeaf: false,
      selfMute: false,
      daveEncryption: true,
      debug: true,
    });

    connection.on("stateChange", (oldState, newState) => {
      this.logger.info("Voice connection state change", {
        from: oldState.status,
        to: newState.status,
      });
    });

    connection.on("debug", (message) => {
      this.logger.info("Voice connection debug", { message });
    });

    await entersState(connection, VoiceConnectionStatus.Ready, 30_000);
    this.connection = connection;

    connection.on("error", (error) => {
      this.logger.error("Voice connection error", { error: String(error) });
    });

    connection.on("stateChange", (_oldState, newState) => {
      if (newState.status === VoiceConnectionStatus.Disconnected) {
        this.logger.warn("Voice connection disconnected");
      }
    });

    await this.options.onConnected(connection);
  }

  private async leave(reason: string): Promise<void> {
    if (!this.connection) {
      return;
    }

    this.logger.info("Leaving target voice channel", { reason });
    try {
      await this.options.onDisconnected(reason);
    } catch (error) {
      this.logger.error("Error during disconnect callback", { error: String(error) });
    }

    this.connection.destroy();
    this.connection = null;
  }
}
