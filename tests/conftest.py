"""Shared pytest helpers and fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from torchtalk import indexer


def get_pytorch_path() -> Path | None:
    """Resolve PyTorch source from PYTORCH_SOURCE or PYTORCH_PATH env vars."""
    for var in ("PYTORCH_SOURCE", "PYTORCH_PATH"):
        if path := os.environ.get(var):
            p = Path(path)
            if p.exists() and (p / "torch").exists():
                return p
    return None


@pytest.fixture
def mock_state():
    """Generic save/restore fixture for indexer._state.

    Yields the state object. Set fields on it directly in each test or
    in a per-file fixture that depends on this one. Calls _build_indexes
    on teardown to restore derived lookup dicts.
    """
    s = indexer._state
    fields = [f.name for f in s.__dataclass_fields__.values()]
    saved = {f: getattr(s, f) for f in fields}
    try:
        yield s
    finally:
        for f, val in saved.items():
            setattr(s, f, val)
        indexer._build_indexes(s)
