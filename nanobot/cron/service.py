"""Cron service for scheduling agent tasks."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

import yaml
from loguru import logger

from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronSchedule

_DURATION_RE = re.compile(r"(\d+)([smhd])")
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ms_to_iso(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iso_to_ms(value: str) -> int:
    text = value.strip()
    if not text:
        raise ValueError("empty datetime")
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _parse_duration_seconds(value: str, *, allow_days: bool = True) -> int:
    text = value.strip().lower().replace(" ", "")
    if not text:
        raise ValueError("empty duration")

    pos = 0
    total = 0
    for match in _DURATION_RE.finditer(text):
        if match.start() != pos:
            raise ValueError(f"invalid duration '{value}'")
        amount = int(match.group(1))
        unit = match.group(2)
        if unit == "d":
            if not allow_days:
                raise ValueError(f"invalid duration '{value}'")
            total += amount * 86400
        elif unit == "h":
            total += amount * 3600
        elif unit == "m":
            total += amount * 60
        elif unit == "s":
            total += amount
        pos = match.end()

    if pos != len(text) or total <= 0:
        raise ValueError(f"invalid duration '{value}'")
    return total


def _coerce_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        out = int(value)
        return out if out >= 0 else default
    except Exception:
        return default


def _compute_next_run_ms(
    schedule: CronSchedule,
    *,
    created_at_ms: int,
    last_run_at_ms: int | None,
    runs: int,
) -> int | None:
    if schedule.kind == "at":
        if runs > 0:
            return None
        return schedule.at_ms

    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        base = last_run_at_ms if last_run_at_ms is not None else created_at_ms
        if base <= 0:
            base = _now_ms()
        return base + schedule.every_ms

    if schedule.kind == "cron" and schedule.expr:
        try:
            from croniter import croniter
            from zoneinfo import ZoneInfo

            base_ms = last_run_at_ms if last_run_at_ms is not None else created_at_ms
            if base_ms <= 0:
                base_ms = _now_ms()
            tz = ZoneInfo(schedule.tz) if schedule.tz else datetime.now().astimezone().tzinfo
            base_dt = datetime.fromtimestamp(base_ms / 1000, tz=tz)
            return int(croniter(schedule.expr, base_dt).get_next(datetime).timestamp() * 1000)
        except Exception:
            return None

    return None


def _validate_schedule_for_add(schedule: CronSchedule) -> None:
    if schedule.tz and schedule.kind != "cron":
        raise ValueError("tz can only be used with cron schedules")

    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            raise ValueError("every schedule requires every_ms > 0")

    if schedule.kind == "at":
        if not schedule.at_ms:
            raise ValueError("at schedule requires at_ms")

    if schedule.kind == "cron":
        if not schedule.expr:
            raise ValueError("cron schedule requires expr")
        if schedule.tz:
            try:
                from zoneinfo import ZoneInfo

                ZoneInfo(schedule.tz)
            except Exception:
                raise ValueError(f"unknown timezone '{schedule.tz}'") from None


def _normalize_job_id(job_id: str) -> str:
    stem = Path(job_id.strip()).stem
    if not stem or not _JOB_ID_RE.match(stem):
        raise ValueError(f"invalid job id '{job_id}'")
    return stem


class CronService:
    """Service for managing and executing scheduled jobs."""

    def __init__(
        self,
        jobs_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None,
    ):
        p = Path(jobs_path)
        if p.suffix == ".json":
            self.legacy_store_path = p
            self.jobs_dir = p.parent / "jobs"
        else:
            self.jobs_dir = p
            self.legacy_store_path = p.parent / "jobs.json"

        self.on_job = on_job
        self._timer_task: asyncio.Task | None = None
        self._running = False
        self._initialized = False

    def _ensure_storage(self) -> None:
        if self._initialized:
            return
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_store_if_needed()
        self._initialized = True

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.yaml"

    def _lock_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.lock"

    def _read_job_dict(self, path: Path) -> dict[str, Any] | None:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read cron job {}: {}", path, e)
            return None
        if not isinstance(payload, dict):
            logger.warning("Cron job {} is not a YAML object", path)
            return None
        return payload

    def _write_job_dict(self, path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        data = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        tmp_path.write_text(data, encoding="utf-8")
        tmp_path.replace(path)

    def _parse_timeout_seconds(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            timeout_s = int(value)
            return timeout_s if timeout_s > 0 else None
        if isinstance(value, str):
            try:
                return _parse_duration_seconds(value)
            except ValueError:
                logger.warning("Ignoring invalid timeout '{}'", value)
        return None

    def _parse_max_runs(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            max_runs = int(value)
            return max_runs if max_runs > 0 else None
        except Exception:
            logger.warning("Ignoring invalid max_runs '{}'", value)
            return None

    def _parse_schedule(self, schedule_text: str, tz: str | None) -> CronSchedule:
        text = schedule_text.strip()
        if not text:
            raise ValueError("missing schedule")

        if text.startswith("every "):
            interval = text[len("every "):].strip()
            seconds = _parse_duration_seconds(interval, allow_days=False)
            if tz:
                raise ValueError("tz can only be used with cron schedules")
            return CronSchedule(kind="every", every_ms=seconds * 1000, raw=text)

        if text.startswith("at "):
            at_value = text[len("at "):].strip()
            at_ms = _iso_to_ms(at_value)
            if tz:
                raise ValueError("tz can only be used with cron schedules")
            return CronSchedule(kind="at", at_ms=at_ms, raw=text)

        schedule = CronSchedule(kind="cron", expr=text, tz=tz, raw=text)
        _validate_schedule_for_add(schedule)
        return schedule

    def _schedule_to_text(self, schedule: CronSchedule) -> str:
        if schedule.kind == "every":
            if not schedule.every_ms or schedule.every_ms <= 0:
                raise ValueError("every schedule requires every_ms > 0")
            seconds = max(1, int(round(schedule.every_ms / 1000)))
            return f"every {seconds}s"

        if schedule.kind == "at":
            if not schedule.at_ms:
                raise ValueError("at schedule requires at_ms")
            return f"at {_ms_to_iso(schedule.at_ms)}"

        if schedule.kind == "cron":
            if not schedule.expr:
                raise ValueError("cron schedule requires expr")
            return schedule.expr

        raise ValueError(f"unsupported schedule kind '{schedule.kind}'")

    def _load_job_from_path(self, path: Path, now_ms: int | None = None) -> CronJob | None:
        payload = self._read_job_dict(path)
        if payload is None:
            return None

        try:
            schedule = self._parse_schedule(
                str(payload.get("schedule") or ""),
                payload.get("tz") if isinstance(payload.get("tz"), str) else None,
            )
        except Exception as e:
            logger.warning("Invalid schedule in {}: {}", path, e)
            return None

        current_ms = now_ms or _now_ms()

        created_at_raw = payload.get("created_at")
        try:
            created_at_ms = _iso_to_ms(str(created_at_raw)) if created_at_raw else int(path.stat().st_mtime * 1000)
        except Exception:
            created_at_ms = int(path.stat().st_mtime * 1000)

        updated_at_raw = payload.get("updated_at")
        try:
            updated_at_ms = _iso_to_ms(str(updated_at_raw)) if updated_at_raw else created_at_ms
        except Exception:
            updated_at_ms = created_at_ms

        last_run_raw = payload.get("last_run")
        try:
            last_run_at_ms = _iso_to_ms(str(last_run_raw)) if last_run_raw else None
        except Exception:
            last_run_at_ms = None

        runs = _coerce_non_negative_int(payload.get("runs"), default=0)
        timeout_s = self._parse_timeout_seconds(payload.get("timeout"))
        max_runs = self._parse_max_runs(payload.get("max_runs"))
        enabled = bool(payload.get("enabled", True))

        state = CronJobState(
            next_run_at_ms=(
                _compute_next_run_ms(
                    schedule,
                    created_at_ms=created_at_ms,
                    last_run_at_ms=last_run_at_ms,
                    runs=runs,
                )
                if enabled
                else None
            ),
            last_run_at_ms=last_run_at_ms,
            last_status=payload.get("last_status"),
            last_error=payload.get("last_error"),
            runs=runs,
        )

        message = str(payload.get("message") or "")
        name = str(payload.get("name") or message[:30] or path.stem)

        job = CronJob(
            id=path.stem,
            name=name,
            enabled=enabled,
            schedule=schedule,
            payload=CronPayload(
                kind="agent_turn",
                message=message,
                deliver=bool(payload.get("deliver", True)),
                channel=payload.get("channel"),
                to=payload.get("chat_id") or payload.get("to"),
                agent_id=payload.get("agent") or payload.get("agent_id"),
            ),
            state=state,
            created_at_ms=created_at_ms,
            updated_at_ms=updated_at_ms,
            delete_after_run=schedule.kind == "at",
            timeout_s=timeout_s,
            max_runs=max_runs,
            file_path=path,
        )

        # Opportunistic cleanup of malformed data in memory view.
        if timeout_s is not None and current_ms > created_at_ms + timeout_s * 1000:
            job.state.next_run_at_ms = current_ms
        if max_runs is not None and runs >= max_runs:
            job.state.next_run_at_ms = current_ms

        return job

    def _load_job(self, job_id: str, now_ms: int | None = None) -> CronJob | None:
        return self._load_job_from_path(self._job_path(job_id), now_ms=now_ms)

    def _delete_job_file(self, job_id: str) -> bool:
        path = self._job_path(job_id)
        if not path.exists():
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def _acquire_lock(self, job_id: str) -> bool:
        lock_path = self._lock_path(job_id)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8"))
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            return False

    def _release_lock(self, job_id: str) -> None:
        self._lock_path(job_id).unlink(missing_ok=True)

    def _should_delete_without_run(self, job: CronJob, now_ms: int) -> bool:
        if job.timeout_s is not None and now_ms > job.created_at_ms + job.timeout_s * 1000:
            return True
        if job.max_runs is not None and job.state.runs >= job.max_runs:
            return True
        return False

    def _migrate_legacy_store_if_needed(self) -> None:
        if not self.legacy_store_path.exists():
            return

        try:
            raw = json.loads(self.legacy_store_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read legacy cron store {}: {}", self.legacy_store_path, e)
            return

        jobs = raw.get("jobs") if isinstance(raw, dict) else None
        if not isinstance(jobs, list):
            logger.warning("Legacy cron store {} has invalid schema", self.legacy_store_path)
            return

        migrated = 0
        for row in jobs:
            if not isinstance(row, dict):
                continue

            try:
                job_id = _normalize_job_id(str(row.get("id") or uuid.uuid4().hex[:8]))
            except ValueError:
                job_id = uuid.uuid4().hex[:8]

            path = self._job_path(job_id)
            if path.exists():
                continue

            schedule_row = row.get("schedule") if isinstance(row.get("schedule"), dict) else {}
            kind = str(schedule_row.get("kind") or "").strip()

            try:
                if kind == "every":
                    every_ms = int(schedule_row.get("everyMs") or 0)
                    if every_ms <= 0:
                        continue
                    schedule_text = f"every {max(1, int(round(every_ms / 1000)))}s"
                elif kind == "at":
                    at_ms = int(schedule_row.get("atMs") or 0)
                    if at_ms <= 0:
                        continue
                    schedule_text = f"at {_ms_to_iso(at_ms)}"
                elif kind == "cron":
                    schedule_text = str(schedule_row.get("expr") or "").strip()
                    if not schedule_text:
                        continue
                else:
                    continue
            except Exception:
                continue

            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            state = row.get("state") if isinstance(row.get("state"), dict) else {}

            created_at_ms = _coerce_non_negative_int(row.get("createdAtMs"), default=_now_ms())
            updated_at_ms = _coerce_non_negative_int(row.get("updatedAtMs"), default=created_at_ms)
            last_run_ms = _coerce_non_negative_int(state.get("lastRunAtMs"), default=0)

            payload_dict: dict[str, Any] = {
                "schedule": schedule_text,
                "message": str(payload.get("message") or ""),
                "agent": payload.get("agentId"),
                "channel": payload.get("channel"),
                "chat_id": payload.get("to"),
                "deliver": bool(payload.get("deliver", True)),
                "name": str(row.get("name") or "")[:120],
                "created_at": _ms_to_iso(created_at_ms),
                "updated_at": _ms_to_iso(updated_at_ms),
                "runs": _coerce_non_negative_int(state.get("runs"), default=0),
                "last_run": _ms_to_iso(last_run_ms) if last_run_ms else None,
                "enabled": bool(row.get("enabled", True)),
                "last_status": state.get("lastStatus"),
                "last_error": state.get("lastError"),
            }
            if kind == "cron" and schedule_row.get("tz"):
                payload_dict["tz"] = schedule_row.get("tz")

            self._write_job_dict(path, payload_dict)
            migrated += 1

        try:
            self.legacy_store_path.unlink()
        except Exception as e:
            logger.warning("Failed to delete legacy cron store {}: {}", self.legacy_store_path, e)

        if migrated:
            logger.info("Cron: migrated {} legacy job(s) to {}", migrated, self.jobs_dir)

    async def start(self) -> None:
        """Start the cron service."""
        self._ensure_storage()
        self._running = True
        self._arm_timer()
        logger.info("Cron service started with {} jobs", len(self.list_jobs(include_disabled=False)))

    def stop(self) -> None:
        """Stop the cron service."""
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

    def _get_next_wake_ms(self) -> int | None:
        jobs = self.list_jobs(include_disabled=False)
        if not jobs:
            return None

        now_ms = _now_ms()
        next_wake: int | None = None

        for job in jobs:
            if self._should_delete_without_run(job, now_ms):
                return now_ms
            next_run = job.state.next_run_at_ms
            if next_run is None:
                continue
            if next_wake is None or next_run < next_wake:
                next_wake = next_run

        return next_wake

    def _arm_timer(self) -> None:
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

        if not self._running:
            return

        next_wake = self._get_next_wake_ms()
        if next_wake is None:
            return

        delay_s = max(0, next_wake - _now_ms()) / 1000

        async def _tick() -> None:
            try:
                await asyncio.sleep(delay_s)
                if self._running:
                    await self._on_timer()
            except asyncio.CancelledError:
                return

        self._timer_task = asyncio.create_task(_tick())

    async def _on_timer(self) -> None:
        if not self._running:
            return

        now_ms = _now_ms()
        jobs = self.list_jobs(include_disabled=False)

        for job in jobs:
            if self._should_delete_without_run(job, now_ms):
                self._delete_job_file(job.id)
                continue

            next_run = job.state.next_run_at_ms
            if next_run is not None and now_ms >= next_run:
                await self._execute_job(job, now_ms=now_ms)

        self._arm_timer()

    def _finalize_execution(
        self,
        job_id: str,
        *,
        started_ms: int,
        status: str,
        error: str | None,
    ) -> None:
        path = self._job_path(job_id)
        payload = self._read_job_dict(path)
        if payload is None:
            # The file may have been deleted during callback execution.
            return

        runs = _coerce_non_negative_int(payload.get("runs"), default=0) + 1
        payload["runs"] = runs
        payload["last_run"] = _ms_to_iso(started_ms)
        payload["updated_at"] = _ms_to_iso(_now_ms())
        payload["last_status"] = status
        payload["last_error"] = error

        schedule_text = str(payload.get("schedule") or "").strip().lower()
        max_runs = self._parse_max_runs(payload.get("max_runs"))

        if schedule_text.startswith("at "):
            path.unlink(missing_ok=True)
            return

        if max_runs is not None and runs >= max_runs:
            path.unlink(missing_ok=True)
            return

        timeout_s = self._parse_timeout_seconds(payload.get("timeout"))
        if timeout_s is not None:
            try:
                created_at_ms = _iso_to_ms(str(payload.get("created_at")))
                if _now_ms() > created_at_ms + timeout_s * 1000:
                    path.unlink(missing_ok=True)
                    return
            except Exception:
                pass

        if path.exists():
            self._write_job_dict(path, payload)

    async def _execute_job(
        self,
        job: CronJob,
        *,
        now_ms: int | None = None,
        manual: bool = False,
    ) -> bool:
        latest = self._load_job(job.id, now_ms=now_ms)
        if latest is None:
            return False

        if not latest.enabled and not manual:
            return False

        current_ms = now_ms or _now_ms()

        if self._should_delete_without_run(latest, current_ms):
            self._delete_job_file(latest.id)
            return False

        if not manual:
            next_run = latest.state.next_run_at_ms
            if next_run is None or current_ms < next_run:
                return False

        if not self._acquire_lock(latest.id):
            logger.debug("Cron: skipping job {} tick (already running)", latest.id)
            return False

        started_ms = _now_ms()
        status = "ok"
        error: str | None = None

        try:
            logger.info("Cron: executing job '{}' ({})", latest.name, latest.id)
            if self.on_job:
                await self.on_job(latest)
            logger.info("Cron: job '{}' completed", latest.name)
        except Exception as exc:
            status = "error"
            error = str(exc)
            logger.exception("Cron: job '{}' failed", latest.name)
        finally:
            try:
                self._finalize_execution(
                    latest.id,
                    started_ms=started_ms,
                    status=status,
                    error=error,
                )
            finally:
                self._release_lock(latest.id)

        return True

    # ========== Public API ==========

    def list_jobs(
        self,
        include_disabled: bool = False,
        agent_id: str | None = None,
    ) -> list[CronJob]:
        """List jobs, optionally filtered by agent_id."""
        self._ensure_storage()
        now_ms = _now_ms()

        jobs: list[CronJob] = []
        for path in sorted(self.jobs_dir.glob("*.yaml")):
            job = self._load_job_from_path(path, now_ms=now_ms)
            if job is None:
                continue
            if not include_disabled and not job.enabled:
                continue
            if agent_id and job.payload.agent_id != agent_id:
                continue
            jobs.append(job)

        jobs.sort(key=lambda j: j.state.next_run_at_ms if j.state.next_run_at_ms is not None else 10**18)
        return jobs

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
        agent_id: str | None = None,
        timeout: str | int | None = None,
        max_runs: int | None = None,
    ) -> CronJob:
        """Add a new job."""
        self._ensure_storage()
        _validate_schedule_for_add(schedule)

        now_ms = _now_ms()
        created_at = _ms_to_iso(now_ms)

        job_id = ""
        while not job_id:
            candidate = uuid.uuid4().hex[:8]
            path = self._job_path(candidate)
            if not path.exists():
                job_id = candidate

        payload: dict[str, Any] = {
            "schedule": self._schedule_to_text(schedule),
            "message": message,
            "agent": agent_id,
            "channel": channel,
            "chat_id": to,
            "deliver": bool(deliver),
            "name": name,
            "created_at": created_at,
            "updated_at": created_at,
            "runs": 0,
            "last_run": None,
            "enabled": True,
        }

        if schedule.kind == "cron" and schedule.tz:
            payload["tz"] = schedule.tz

        if timeout is not None:
            if isinstance(timeout, (int, float)):
                timeout_s = int(timeout)
                if timeout_s <= 0:
                    raise ValueError("timeout must be > 0")
                payload["timeout"] = f"{timeout_s}s"
            elif isinstance(timeout, str):
                _parse_duration_seconds(timeout)
                payload["timeout"] = timeout
            else:
                raise ValueError("timeout must be a duration string or integer seconds")

        if max_runs is not None:
            max_runs_int = int(max_runs)
            if max_runs_int <= 0:
                raise ValueError("max_runs must be > 0")
            payload["max_runs"] = max_runs_int

        if delete_after_run and schedule.kind == "at":
            payload["delete_after_run"] = True

        path = self._job_path(job_id)
        self._write_job_dict(path, payload)
        self._arm_timer()

        job = self._load_job(job_id)
        if job is None:
            raise RuntimeError("failed to load newly created cron job")

        logger.info("Cron: added job '{}' ({})", job.name, job.id)
        return job

    def remove_job(self, job_id: str, agent_id: str | None = None) -> bool:
        """Remove a job by ID. If agent_id is provided, only remove if it matches."""
        self._ensure_storage()
        try:
            normalized = _normalize_job_id(job_id)
        except ValueError:
            return False

        path = self._job_path(normalized)
        if not path.exists():
            return False

        if agent_id:
            job = self._load_job(normalized)
            if job is None or job.payload.agent_id != agent_id:
                return False

        removed = self._delete_job_file(normalized)
        if removed:
            logger.info("Cron: removed job {}", normalized)
            self._arm_timer()
        return removed

    def enable_job(self, job_id: str, enabled: bool = True) -> CronJob | None:
        """Enable or disable a job."""
        self._ensure_storage()
        try:
            normalized = _normalize_job_id(job_id)
        except ValueError:
            return None

        path = self._job_path(normalized)
        payload = self._read_job_dict(path)
        if payload is None:
            return None

        payload["enabled"] = bool(enabled)
        payload["updated_at"] = _ms_to_iso(_now_ms())
        self._write_job_dict(path, payload)
        self._arm_timer()
        return self._load_job(normalized)

    async def run_job(self, job_id: str, force: bool = False) -> bool:
        """Manually run a job."""
        self._ensure_storage()
        try:
            normalized = _normalize_job_id(job_id)
        except ValueError:
            return False

        job = self._load_job(normalized)
        if job is None:
            return False

        if not force and not job.enabled:
            return False

        ran = await self._execute_job(job, manual=True)
        self._arm_timer()
        return ran

    def status(self) -> dict[str, Any]:
        """Get service status."""
        self._ensure_storage()
        return {
            "enabled": self._running,
            "jobs": len(self.list_jobs(include_disabled=False)),
            "next_wake_at_ms": self._get_next_wake_ms(),
        }
