import json
from pathlib import Path

import pytest
import yaml

from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_add_job_rejects_unknown_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs")

    with pytest.raises(ValueError, match="unknown timezone 'America/Vancovuer'"):
        service.add_job(
            name="tz typo",
            schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancovuer"),
            message="hello",
        )

    assert service.list_jobs(include_disabled=True) == []


def test_add_job_creates_yaml_file(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs")

    job = service.add_job(
        name="health check",
        schedule=CronSchedule(kind="every", every_ms=120_000),
        message="check status",
        deliver=True,
        channel="discord",
        to="123",
        agent_id="nanobot-dev",
        timeout="20m",
        max_runs=5,
    )

    path = tmp_path / "cron" / "jobs" / f"{job.id}.yaml"
    assert path.exists()

    payload = _read_yaml(path)
    assert payload["schedule"] == "every 120s"
    assert payload["message"] == "check status"
    assert payload["agent"] == "nanobot-dev"
    assert payload["channel"] == "discord"
    assert payload["chat_id"] == "123"
    assert payload["timeout"] == "20m"
    assert payload["max_runs"] == 5
    assert payload["runs"] == 0


def test_remove_job_deletes_file_and_respects_agent_scope(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs")
    job = service.add_job(
        name="demo",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="hello",
        agent_id="agent-a",
    )

    path = tmp_path / "cron" / "jobs" / f"{job.id}.yaml"
    assert path.exists()

    assert service.remove_job(job.id, agent_id="agent-b") is False
    assert path.exists()

    assert service.remove_job(job.id, agent_id="agent-a") is True
    assert not path.exists()


@pytest.mark.asyncio
async def test_timeout_auto_deletes_before_execution(tmp_path) -> None:
    calls = 0

    async def on_job(_job):
        nonlocal calls
        calls += 1

    service = CronService(tmp_path / "cron" / "jobs", on_job=on_job)

    job = service.add_job(
        name="timeout job",
        schedule=CronSchedule(kind="every", every_ms=1_000),
        message="hello",
    )
    path = tmp_path / "cron" / "jobs" / f"{job.id}.yaml"

    payload = _read_yaml(path)
    payload["created_at"] = "2000-01-01T00:00:00Z"
    payload["timeout"] = "1s"
    _write_yaml(path, payload)

    service._running = True
    await service._on_timer()
    service.stop()

    assert calls == 0
    assert not path.exists()


@pytest.mark.asyncio
async def test_max_runs_auto_deletes_after_execution(tmp_path) -> None:
    calls = 0

    async def on_job(_job):
        nonlocal calls
        calls += 1

    service = CronService(tmp_path / "cron" / "jobs", on_job=on_job)
    job = service.add_job(
        name="max runs",
        schedule=CronSchedule(kind="every", every_ms=1_000),
        message="hello",
        max_runs=1,
    )
    path = tmp_path / "cron" / "jobs" / f"{job.id}.yaml"

    assert await service.run_job(job.id)

    assert calls == 1
    assert not path.exists()


@pytest.mark.asyncio
async def test_one_shot_at_job_deletes_after_execution(tmp_path) -> None:
    calls = 0

    async def on_job(_job):
        nonlocal calls
        calls += 1

    service = CronService(tmp_path / "cron" / "jobs", on_job=on_job)
    job = service.add_job(
        name="one-shot",
        schedule=CronSchedule(kind="at", at_ms=2_000_000_000_000),
        message="hello once",
    )
    path = tmp_path / "cron" / "jobs" / f"{job.id}.yaml"

    assert await service.run_job(job.id, force=True)

    assert calls == 1
    assert not path.exists()


@pytest.mark.asyncio
async def test_lockfile_skips_due_tick(tmp_path) -> None:
    calls = 0

    async def on_job(_job):
        nonlocal calls
        calls += 1

    service = CronService(tmp_path / "cron" / "jobs", on_job=on_job)
    job = service.add_job(
        name="skip if locked",
        schedule=CronSchedule(kind="every", every_ms=1_000),
        message="hello",
    )
    path = tmp_path / "cron" / "jobs" / f"{job.id}.yaml"

    payload = _read_yaml(path)
    payload["created_at"] = "2000-01-01T00:00:00Z"
    _write_yaml(path, payload)

    lock_path = tmp_path / "cron" / "jobs" / f"{job.id}.lock"
    lock_path.write_text("locked", encoding="utf-8")

    service._running = True
    await service._on_timer()
    service.stop()

    assert calls == 0
    assert path.exists()
    assert _read_yaml(path)["runs"] == 0


def test_list_jobs_filters_agent_id(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs")
    service.add_job(
        name="a-job",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="a",
        agent_id="agent-a",
    )
    service.add_job(
        name="b-job",
        schedule=CronSchedule(kind="every", every_ms=60_000),
        message="b",
        agent_id="agent-b",
    )

    jobs_a = service.list_jobs(agent_id="agent-a")
    jobs_b = service.list_jobs(agent_id="agent-b")

    assert len(jobs_a) == 1
    assert len(jobs_b) == 1
    assert jobs_a[0].payload.agent_id == "agent-a"
    assert jobs_b[0].payload.agent_id == "agent-b"


@pytest.mark.asyncio
async def test_start_migrates_legacy_jobs_json(tmp_path) -> None:
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = cron_dir / "jobs.json"

    legacy_payload = {
        "version": 1,
        "jobs": [
            {
                "id": "abc123",
                "name": "legacy",
                "enabled": True,
                "schedule": {"kind": "cron", "expr": "0 9 * * *", "tz": "America/Vancouver"},
                "payload": {
                    "message": "legacy message",
                    "deliver": True,
                    "channel": "discord",
                    "to": "777",
                    "agentId": "legacy-agent",
                },
                "state": {"lastRunAtMs": None, "lastStatus": None, "lastError": None},
                "createdAtMs": 1_700_000_000_000,
                "updatedAtMs": 1_700_000_000_000,
                "deleteAfterRun": False,
            }
        ],
    }
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    service = CronService(legacy_path)
    await service.start()
    service.stop()

    assert not legacy_path.exists()

    path = cron_dir / "jobs" / "abc123.yaml"
    assert path.exists()

    payload = _read_yaml(path)
    assert payload["schedule"] == "0 9 * * *"
    assert payload["tz"] == "America/Vancouver"
    assert payload["message"] == "legacy message"
    assert payload["agent"] == "legacy-agent"
