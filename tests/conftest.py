"""Shared pytest helpers."""

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


def get_vllm_path() -> Path | None:
    """Resolve vLLM source from VLLM_SOURCE env var."""

    if path := os.environ.get("VLLM_SOURCE"):
        p = Path(path)
        if p.exists() and (p / "vllm").exists():
            return p
    return None


@pytest.fixture
def vllm_state():
    """Bootstrap the vLLM adapter against the local checkout when available."""

    vllm_path = get_vllm_path()
    if vllm_path is None:
        pytest.skip("VLLM_SOURCE environment variable not set")

    prior_state = dict(indexer._state.__dict__)
    try:
        indexer.init_via_adapter(framework="vllm", source=str(vllm_path))
        yield indexer._state
    finally:
        for field_name, value in prior_state.items():
            setattr(indexer._state, field_name, value)
