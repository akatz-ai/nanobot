import os from "node:os";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

function formatUptime(seconds: number): string {
  if (seconds < 60) {
    return `${Math.floor(seconds)}s`;
  }
  if (seconds < 3600) {
    return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${minutes}m`;
}

async function getRootDiskUsage(): Promise<Record<string, unknown> | null> {
  try {
    const { stdout } = await execFileAsync("df", ["-kP", "/"]);
    const lines = stdout.trim().split("\n");
    if (lines.length < 2) {
      return null;
    }
    const parts = lines[1].trim().split(/\s+/);
    if (parts.length < 6) {
      return null;
    }
    const sizeKb = Number.parseInt(parts[1], 10);
    const usedKb = Number.parseInt(parts[2], 10);
    const availKb = Number.parseInt(parts[3], 10);
    const usePercent = parts[4];
    const mountpoint = parts[5];
    return {
      filesystem: parts[0],
      size_kb: Number.isFinite(sizeKb) ? sizeKb : null,
      used_kb: Number.isFinite(usedKb) ? usedKb : null,
      available_kb: Number.isFinite(availKb) ? availKb : null,
      use_percent: usePercent,
      mountpoint,
    };
  } catch {
    return null;
  }
}

export async function checkSystemStatus(): Promise<Record<string, unknown>> {
  const uptimeSeconds = os.uptime();
  const totalMem = os.totalmem();
  const freeMem = os.freemem();
  const usedMem = totalMem - freeMem;
  const load = os.loadavg();
  const disk = await getRootDiskUsage();

  return {
    status: "ok",
    platform: os.platform(),
    hostname: os.hostname(),
    uptime_seconds: uptimeSeconds,
    uptime_human: formatUptime(uptimeSeconds),
    cpu_count: os.cpus().length,
    load_avg_1m: load[0],
    load_avg_5m: load[1],
    load_avg_15m: load[2],
    memory: {
      total_bytes: totalMem,
      used_bytes: usedMem,
      free_bytes: freeMem,
      utilization: totalMem > 0 ? usedMem / totalMem : 0,
    },
    disk_root: disk,
  };
}
