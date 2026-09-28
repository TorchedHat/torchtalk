"""Tests for the harness smoke script's pinned sparse checkout."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "harness_smoke.py"
_SPEC = importlib.util.spec_from_file_location("torchtalk_harness_smoke", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
harness_smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(harness_smoke)


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_sparse_clone_checks_out_full_commit_sha(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _git("init", "--initial-branch=main", cwd=source)
    _git("config", "user.email", "torchtalk-tests@example.com", cwd=source)
    _git("config", "user.name", "TorchTalk tests", cwd=source)

    package_file = source / "pkg" / "data.txt"
    package_file.parent.mkdir()
    package_file.write_text("pinned commit\n")
    (source / "outside").mkdir()
    (source / "outside" / "marker.txt").write_text("outside sparse path\n")
    _git("add", ".", cwd=source)
    _git("commit", "-m", "test pinned checkout", cwd=source)
    commit = _git("rev-parse", "HEAD", cwd=source)

    destination = tmp_path / "checkout"
    harness_smoke.sparse_clone(source.as_uri(), commit, ["pkg"], destination)

    assert _git("rev-parse", "HEAD", cwd=destination) == commit
    assert (destination / "pkg" / "data.txt").read_text() == "pinned commit\n"
    assert not (destination / "outside").exists()
