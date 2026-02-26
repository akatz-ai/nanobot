from pathlib import Path

import yaml
from typer.testing import CliRunner

from nanobot.cli.commands import app

runner = CliRunner()


def test_cron_add_rejects_invalid_timezone(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)

    result = runner.invoke(
        app,
        [
            "cron",
            "add",
            "--name",
            "demo",
            "--message",
            "hello",
            "--cron",
            "0 9 * * *",
            "--tz",
            "America/Vancovuer",
        ],
    )

    assert result.exit_code == 1
    assert "Error: unknown timezone 'America/Vancovuer'" in result.stdout

    jobs_dir = tmp_path / "cron" / "jobs"
    assert not jobs_dir.exists() or not list(jobs_dir.glob("*.yaml"))


def test_cron_add_writes_yaml_job_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)

    result = runner.invoke(
        app,
        [
            "cron",
            "add",
            "--name",
            "demo",
            "--message",
            "hello world",
            "--every",
            "60",
            "--timeout",
            "20m",
            "--max-runs",
            "3",
            "--deliver",
            "--channel",
            "discord",
            "--to",
            "12345",
        ],
    )

    assert result.exit_code == 0
    assert "Added job 'demo'" in result.stdout

    jobs_dir = tmp_path / "cron" / "jobs"
    files = list(jobs_dir.glob("*.yaml"))
    assert len(files) == 1

    payload = yaml.safe_load(files[0].read_text(encoding="utf-8"))
    assert payload["name"] == "demo"
    assert payload["schedule"] == "every 60s"
    assert payload["message"] == "hello world"
    assert payload["timeout"] == "20m"
    assert payload["max_runs"] == 3
    assert payload["channel"] == "discord"
    assert payload["chat_id"] == "12345"
