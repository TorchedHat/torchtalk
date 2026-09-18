"""Tests for pool worker sizing and parent-death cleanup."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from torchtalk.analysis.cpp_call_graph import (
    _default_workers,
    _proc_start_time,
    _worker_init,
)


def test_default_workers_uses_affinity_not_host_cpus(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setattr(os, "cpu_count", lambda: 160)
    assert _default_workers() == 6


def test_default_workers_follows_affinity(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(10)))
    assert _default_workers() == 8


def test_default_workers_never_zero(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    assert _default_workers() == 1


def test_worker_init_ignores_sigint():
    original = signal.getsignal(signal.SIGINT)
    try:
        _worker_init()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, original)


POOL_PARENT = Path(__file__).with_name("helpers") / "pool_parent.py"


def _alive(pid: int) -> bool:
    return os.path.isdir(f"/proc/{pid}")


@pytest.mark.skipif(sys.platform != "linux", reason="needs /proc and PDEATHSIG")
@pytest.mark.parametrize("start_method", ["fork", "forkserver"])
def test_workers_die_when_parent_is_killed(start_method):
    proc = subprocess.Popen(
        [sys.executable, str(POOL_PARENT), start_method],
        stdout=subprocess.PIPE,
        text=True,
    )
    workers: list[int] = []
    try:
        workers = [int(pid) for pid in proc.stdout.readline().split()]
        assert workers
        time.sleep(2)  # let workers finish _worker_init
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)

        deadline = time.time() + 15
        while time.time() < deadline and any(_alive(pid) for pid in workers):
            time.sleep(0.25)
        survivors = [pid for pid in workers if _alive(pid)]
        assert not survivors, f"orphaned workers survived: {survivors}"
    finally:
        if proc.poll() is None:
            proc.kill()
        for pid in workers:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)


def test_proc_start_time_of_self_and_missing_pid():
    st = _proc_start_time(os.getpid())
    if sys.platform != "linux":
        assert st is None
        return
    assert st is not None and st.isdigit()
    assert _proc_start_time(2**22 + 12345) is None
