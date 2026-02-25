"""Gateway supervisor daemon."""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger


class GatewayDaemon:
    """Supervisor process that manages the gateway worker."""

    PID_FILE = Path.home() / ".nanobot" / "gateway.pid"
    WORKER_PID_FILE = Path.home() / ".nanobot" / "gateway-worker.pid"

    def __init__(self, port: int, verbose: bool):
        self.port = port
        self.verbose = verbose
        self.worker_process: subprocess.Popen | None = None
        self._restart_requested = False
        self._shutdown_requested = False
        self._shutdown_signal = signal.SIGTERM

    def start(self, daemonize: bool = False) -> None:
        """Write PID file, install signal handlers, and supervise worker."""
        existing = self.read_pid()
        if existing:
            logger.error("Gateway supervisor already running (pid={})", existing)
            return

        if daemonize and os.environ.get("NANOBOT_SUPERVISOR") != "1":
            self._start_detached_supervisor()
            return

        self.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._write_pid_file()
        atexit.register(self.cleanup_pid_file)
        atexit.register(self._cleanup_worker_pid_file)

        signal.signal(signal.SIGUSR1, self._handle_restart)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        logger.info("Gateway supervisor started (pid={})", os.getpid())

        backoff_s = 1
        while not self._shutdown_requested:
            started_at = time.monotonic()
            forced_restart = False
            self.worker_process = self._spawn_worker()
            if self.worker_process is None:
                backoff_s = self._sleep_with_backoff(backoff_s)
                continue

            while self.worker_process and self.worker_process.poll() is None:
                if self._restart_requested:
                    logger.info("Restart requested; stopping worker")
                    forced_restart = True
                    self._restart_requested = False
                    self._stop_worker(signal_to_send=signal.SIGTERM)
                    break
                if self._shutdown_requested:
                    self._stop_worker(signal_to_send=self._shutdown_signal)
                    break
                time.sleep(0.25)

            if self._shutdown_requested:
                break

            if forced_restart:
                self._cleanup_worker_pid_file()
                self.worker_process = None
                backoff_s = 1
                continue

            runtime_s = time.monotonic() - started_at
            return_code = None if self.worker_process is None else self.worker_process.poll()
            self._cleanup_worker_pid_file()
            self.worker_process = None

            if return_code is not None:
                if return_code == 0:
                    logger.info("Gateway worker exited cleanly")
                    if runtime_s > 30:
                        backoff_s = 1
                else:
                    logger.warning("Gateway worker exited with code {}", return_code)
                    if runtime_s <= 30:
                        backoff_s = self._sleep_with_backoff(backoff_s)
                    else:
                        backoff_s = 1

        self._stop_worker(signal_to_send=self._shutdown_signal)
        self.cleanup_pid_file()
        self._cleanup_worker_pid_file()
        logger.info("Gateway supervisor stopped")

    def _start_detached_supervisor(self) -> None:
        cmd = [sys.executable, "-m", "nanobot", "gateway", "--port", str(self.port)]
        if self.verbose:
            cmd.append("--verbose")
        env = os.environ.copy()
        env["NANOBOT_SUPERVISOR"] = "1"
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("Gateway daemonized (supervisor pid={})", proc.pid)

    def _write_pid_file(self) -> None:
        self.PID_FILE.write_text(f"{os.getpid()}\n", encoding="utf-8")

    def _write_worker_pid_file(self, worker_pid: int) -> None:
        self.WORKER_PID_FILE.write_text(f"{worker_pid}\n", encoding="utf-8")

    def _spawn_worker(self) -> subprocess.Popen | None:
        """Start the gateway worker as a subprocess."""
        cmd = [
            sys.executable,
            "-m",
            "nanobot",
            "gateway-worker",
            "--port",
            str(self.port),
        ]
        if self.verbose:
            cmd.append("--verbose")

        try:
            proc = subprocess.Popen(cmd)
        except Exception as exc:
            logger.exception("Failed to start gateway worker: {}", exc)
            return None

        self._write_worker_pid_file(proc.pid)
        logger.info("Gateway worker started (pid={})", proc.pid)
        return proc

    def _stop_worker(self, timeout: int = 10, signal_to_send: signal.Signals = signal.SIGTERM) -> None:
        """Gracefully stop worker: signal, wait, then SIGKILL if needed."""
        if not self.worker_process:
            return

        proc = self.worker_process
        if proc.poll() is not None:
            self._cleanup_worker_pid_file()
            return

        try:
            proc.send_signal(signal_to_send)
            proc.wait(timeout=timeout)
            logger.info("Gateway worker stopped")
        except subprocess.TimeoutExpired:
            logger.warning("Gateway worker did not stop in {}s; force killing", timeout)
            proc.kill()
            proc.wait(timeout=2)
        except Exception as exc:
            logger.warning("Error while stopping gateway worker: {}", exc)
        finally:
            self._cleanup_worker_pid_file()

    def _sleep_with_backoff(self, current_backoff_s: int) -> int:
        wait_s = min(current_backoff_s, 60)
        logger.info("Respawning worker in {}s", wait_s)
        end = time.monotonic() + wait_s
        while time.monotonic() < end:
            if self._restart_requested or self._shutdown_requested:
                return 1
            time.sleep(0.1)
        return min(wait_s * 2, 60)

    def _handle_restart(self, signum: int, frame) -> None:  # noqa: ARG002
        """SIGUSR1 handler: request a worker restart."""
        if signum == signal.SIGUSR1:
            self._restart_requested = True

    def _handle_shutdown(self, signum: int, frame) -> None:  # noqa: ARG002
        """SIGTERM/SIGINT handler: request supervisor shutdown."""
        self._shutdown_requested = True
        self._shutdown_signal = signal.SIGINT if signum == signal.SIGINT else signal.SIGTERM

    @classmethod
    def _read_pid_from_file(cls, pid_file: Path) -> int | None:
        if not pid_file.exists():
            return None
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (TypeError, ValueError):
            try:
                pid_file.unlink()
            except FileNotFoundError:
                pass
            return None

        try:
            os.kill(pid, 0)
            return pid
        except ProcessLookupError:
            try:
                pid_file.unlink()
            except FileNotFoundError:
                pass
            return None
        except PermissionError:
            return pid

    @classmethod
    def read_pid(cls) -> int | None:
        """Read supervisor PID from file; return None when not running."""
        return cls._read_pid_from_file(cls.PID_FILE)

    @classmethod
    def read_worker_pid(cls) -> int | None:
        """Read worker PID from file; return None when not running."""
        return cls._read_pid_from_file(cls.WORKER_PID_FILE)

    @classmethod
    def send_signal(cls, sig: signal.Signals) -> bool:
        """Read supervisor PID and send signal."""
        pid = cls.read_pid()
        if not pid:
            return False
        try:
            os.kill(pid, sig)
            return True
        except ProcessLookupError:
            cls.cleanup_pid_file()
            return False
        except PermissionError:
            return False

    @classmethod
    def cleanup_pid_file(cls) -> None:
        """Remove supervisor PID file."""
        try:
            cls.PID_FILE.unlink()
        except FileNotFoundError:
            return

    @classmethod
    def _cleanup_worker_pid_file(cls) -> None:
        try:
            cls.WORKER_PID_FILE.unlink()
        except FileNotFoundError:
            return
