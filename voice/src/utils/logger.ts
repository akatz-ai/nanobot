export type LogLevel = "debug" | "info" | "warn" | "error";

const LOG_LEVEL_ORDER: Record<LogLevel, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
};

export function parseLogLevel(value?: string): LogLevel {
  if (!value) {
    return "info";
  }

  const normalized = value.toLowerCase();
  if (normalized === "debug" || normalized === "info" || normalized === "warn" || normalized === "error") {
    return normalized;
  }
  return "info";
}

export class Logger {
  constructor(
    private readonly scope: string,
    private readonly level: LogLevel = "info",
  ) {}

  child(scope: string): Logger {
    return new Logger(`${this.scope}:${scope}`, this.level);
  }

  debug(message: string, meta?: Record<string, unknown>): void {
    this.log("debug", message, meta);
  }

  info(message: string, meta?: Record<string, unknown>): void {
    this.log("info", message, meta);
  }

  warn(message: string, meta?: Record<string, unknown>): void {
    this.log("warn", message, meta);
  }

  error(message: string, meta?: Record<string, unknown>): void {
    this.log("error", message, meta);
  }

  private log(level: LogLevel, message: string, meta?: Record<string, unknown>): void {
    if (LOG_LEVEL_ORDER[level] < LOG_LEVEL_ORDER[this.level]) {
      return;
    }

    const payload: Record<string, unknown> = {
      ts: new Date().toISOString(),
      level,
      scope: this.scope,
      message,
    };
    if (meta && Object.keys(meta).length > 0) {
      payload.meta = meta;
    }

    const line = JSON.stringify(payload);
    if (level === "error") {
      console.error(line);
      return;
    }
    console.log(line);
  }
}
