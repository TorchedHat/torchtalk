"""Integration tests for the vLLM adapter and static index."""

from __future__ import annotations

import pytest

from torchtalk.adapters import get_adapter, list_adapters
from torchtalk.tools import vllm as vllm_tools

from .conftest import get_vllm_path

VLLM_PATH = get_vllm_path()

pytestmark = pytest.mark.skipif(
    VLLM_PATH is None,
    reason="VLLM_SOURCE environment variable not set",
)

class TestVllmAdapter:
    def test_vllm_adapter_is_registered(self):
        assert "vllm" in list_adapters()
        adapter = get_adapter("vllm")
        assert adapter.framework_id == "vllm"

    def test_vllm_bootstrap_populates_entities(self, vllm_state):
        assert vllm_state.framework == "vllm"
        assert vllm_state.source_root == str(VLLM_PATH.resolve())
        assert "trace_apis" in vllm_state.capabilities
        assert vllm_state.entity_counts["api_entrypoints"] > 0
        assert vllm_state.entity_counts["model_architectures"] > 0
        assert vllm_state.entity_counts["attention_backends"] > 0
        assert vllm_state.entity_counts["ir_ops"] > 0
        assert vllm_state.entity_counts["torch_custom_ops"] > 0
        assert len(vllm_state.proof_traces) >= 5
        assert len(vllm_state.graph_edges) > 0
        assert vllm_state.pytorch_source is None


class TestVllmTools:
    @pytest.mark.asyncio
    async def test_vllm_search_finds_rms_norm(self, vllm_state):
        result = await vllm_tools.search("rms_norm", mode="ops", limit=10)
        assert "rms_norm" in result
        assert "vllm_c.py" in result

    @pytest.mark.asyncio
    async def test_vllm_trace_for_offline_generate(self, vllm_state):
        result = await vllm_tools.trace("LLM.generate")
        assert "Offline LLM.generate()" in result
        assert "LLMEngine.add_request" in result

    @pytest.mark.asyncio
    async def test_vllm_graph_for_attention_selector(self, vllm_state):
        result = await vllm_tools.graph(
            "_cached_get_attn_backend",
            mode="calls",
            depth=2,
        )
        assert "CudaPlatformBase.get_attn_backend_cls" in result
        assert "FLASH_ATTN" in result
