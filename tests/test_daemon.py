import signal

import pytest

from nanobot.daemon import GatewayDaemon


@pytest.fixture
def daemon_paths(tmp_path, monkeypatch):
    pid_file = tmp_path / "gateway.pid"
    worker_pid_file = tmp_path / "gateway-worker.pid"
    monkeypatch.setattr(GatewayDaemon, "PID_FILE", pid_file)
    monkeypatch.setattr(GatewayDaemon, "WORKER_PID_FILE", worker_pid_file)
    return pid_file, worker_pid_file


def test_read_pid_cleans_stale_pid_file(daemon_paths, monkeypatch) -> None:
    pid_file, _ = daemon_paths
    pid_file.write_text("999999\n", encoding="utf-8")

    def _kill(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("os.kill", _kill)
    assert GatewayDaemon.read_pid() is None
    assert not pid_file.exists()


def test_send_signal_uses_supervisor_pid(daemon_paths, monkeypatch) -> None:
    pid_file, _ = daemon_paths
    pid_file.write_text("1234\n", encoding="utf-8")
    calls: list[tuple[int, int | signal.Signals]] = []

    def _kill(pid: int, sig: int | signal.Signals) -> None:
        calls.append((pid, sig))

    monkeypatch.setattr("os.kill", _kill)

    assert GatewayDaemon.send_signal(signal.SIGUSR1) is True
    assert calls == [(1234, 0), (1234, signal.SIGUSR1)]


def test_read_worker_pid_returns_running_pid(daemon_paths, monkeypatch) -> None:
    _, worker_pid_file = daemon_paths
    worker_pid_file.write_text("4321\n", encoding="utf-8")
    calls: list[tuple[int, int | signal.Signals]] = []

    def _kill(pid: int, sig: int | signal.Signals) -> None:
        calls.append((pid, sig))

    monkeypatch.setattr("os.kill", _kill)

    assert GatewayDaemon.read_worker_pid() == 4321
    assert calls == [(4321, 0)]


def test_cleanup_pid_file_removes_file(daemon_paths) -> None:
    pid_file, _ = daemon_paths
    pid_file.write_text("77\n", encoding="utf-8")
    GatewayDaemon.cleanup_pid_file()
    assert not pid_file.exists()
