import { randomUUID } from "node:crypto";

export interface PendingSendAction {
  id: string;
  type: "send_message_to_agent";
  agent: string;
  resolvedAgent: string;
  channelId: string;
  message: string;
  preview: string;
  createdAt: number;
  expiresAt: number;
}

export class ConfirmationGate {
  private readonly pending = new Map<string, PendingSendAction>();
  private readonly cleanupTimer: NodeJS.Timeout;

  constructor(private readonly ttlMs: number = 60_000) {
    this.cleanupTimer = setInterval(() => this.cleanupExpired(), 5_000);
    this.cleanupTimer.unref();
  }

  stop(): void {
    clearInterval(this.cleanupTimer);
    this.pending.clear();
  }

  queueSendAction(input: Omit<PendingSendAction, "id" | "createdAt" | "expiresAt" | "type">): PendingSendAction {
    const now = Date.now();
    const action: PendingSendAction = {
      id: randomUUID(),
      type: "send_message_to_agent",
      createdAt: now,
      expiresAt: now + this.ttlMs,
      ...input,
    };
    this.pending.set(action.id, action);
    return action;
  }

  confirm(actionId: string): PendingSendAction | null {
    const action = this.pending.get(actionId);
    if (!action) {
      return null;
    }
    if (action.expiresAt <= Date.now()) {
      this.pending.delete(actionId);
      return null;
    }
    this.pending.delete(actionId);
    return action;
  }

  cancel(actionId: string): PendingSendAction | null {
    const action = this.pending.get(actionId);
    if (!action) {
      return null;
    }
    this.pending.delete(actionId);
    return action;
  }

  cleanupExpired(): void {
    const now = Date.now();
    for (const [id, action] of this.pending.entries()) {
      if (action.expiresAt <= now) {
        this.pending.delete(id);
      }
    }
  }
}
