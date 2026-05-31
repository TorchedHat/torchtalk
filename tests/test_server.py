"""Tests for TorchTalk server status output."""

from __future__ import annotations

from torchtalk import indexer
from torchtalk.server import get_status


class TestStatus:
    async def test_get_status_includes_framework_and_source_root(self):
        prior_framework = indexer._state.framework
        prior_source_root = indexer._state.source_root
        prior_pytorch_source = indexer._state.pytorch_source

        try:
            indexer._state.framework = "pytorch"
            indexer._state.source_root = "/tmp/pytorch-src"
            indexer._state.pytorch_source = "/tmp/pytorch-src"

            status = await get_status()

            assert "Framework" in status
            assert "PyTorch" in status
            assert "/tmp/pytorch-src" in status
        finally:
            indexer._state.framework = prior_framework
            indexer._state.source_root = prior_source_root
            indexer._state.pytorch_source = prior_pytorch_source
