"""Tests for pool worker sizing and parent-death cleanup."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time

import pytest

from torchtalk.analysis.cpp_call_graph import _default_workers, _worker_init


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


PARENT = """
import os, signal, sys, time
from multiprocessing import Pool
from torchtalk.analysis.cpp_call_graph import _worker_init

def work(i):
    time.sleep(300)

if __name__ == "__main__":
    pool = Pool(processes=2, initializer=_worker_init)
    print(" ".join(str(p.pid) for p in pool._pool), flush=True)
    pool.map_async(work, range(2))
    time.sleep(300)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="PDEATHSIG is Linux-only")
def test_workers_die_when_parent_is_killed():
    proc = subprocess.Popen(
        [sys.executable, "-c", PARENT], stdout=subprocess.PIPE, text=True
    )
    try:
        workers = [int(pid) for pid in proc.stdout.readline().split()]
        assert workers
        time.sleep(2)
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)

        deadline = time.time() + 15
        while time.time() < deadline:
            if not any(os.path.isdir(f"/proc/{pid}") for pid in workers):
                break
            time.sleep(0.25)

        survivors = [pid for pid in workers if os.path.isdir(f"/proc/{pid}")]
        assert not survivors, f"orphaned workers survived: {survivors}"
    finally:
        if proc.poll() is None:
            proc.kill()
        for pid in workers:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
