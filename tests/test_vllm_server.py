"""MCP-facing vLLM integration tests for TorchTalk."""

from __future__ import annotations

import pytest

from torchtalk import server

from .conftest import get_vllm_path

pytestmark = pytest.mark.skipif(
    get_vllm_path() is None,
    reason="VLLM_SOURCE environment variable not set",
)


class TestVllmStatus:
    @pytest.mark.asyncio
    async def test_get_status_reports_vllm_capabilities_and_entities(self, vllm_state):
        status = await server.get_status()
        assert "Framework: vLLM" in status
        assert "Capabilities:" in status
        assert "api_entrypoints: 12" in status
        assert "attention_backends: 31" in status
        assert "Dynamic Areas" in status
        assert "`modules`  Not supported" in status
        assert "`affected`  Not supported" in status


class TestVllmSearchModes:
    @pytest.mark.asyncio
    async def test_search_apis_mode(self, vllm_state):
        result = await server.search("OpenAIServingChat", mode="apis")
        assert "OpenAIServingChat._create_chat_completion" in result
        assert "chat_completion/serving.py" in result

    @pytest.mark.asyncio
    async def test_search_models_mode(self, vllm_state):
        result = await server.search("LlamaForCausalLM", mode="models")
        assert "impl_class=LlamaForCausalLM" in result
        assert "model_executor/models/registry.py" in result

    @pytest.mark.asyncio
    async def test_search_backends_mode(self, vllm_state):
        result = await server.search("FLASH_ATTN", mode="backends")
        assert "FLASH_ATTN" in result
        assert "attention/backends/registry.py" in result

    @pytest.mark.asyncio
    async def test_search_bindings_mode(self, vllm_state):
        result = await server.search("rms_norm", mode="bindings")
        assert "torch.ops._C.rms_norm" in result
        assert "libtorch_stable/torch_bindings.cpp" in result

    @pytest.mark.asyncio
    async def test_search_kernels_not_supported_for_vllm(self, vllm_state):
        with pytest.raises(RuntimeError, match="Kernel search is not available"):
            await server.search("anything", mode="kernels")


class TestVllmTraceAndGraph:
    @pytest.mark.asyncio
    async def test_trace_chat_completion_to_rms_norm(self, vllm_state):
        result = await server.trace("OpenAIServingChat._create_chat_completion")
        assert "Chat completion to rms_norm" in result
        assert "AsyncLLM.generate" in result
        assert "torch.ops._C.rms_norm" in result

    @pytest.mark.asyncio
    async def test_trace_pooling_encode(self, vllm_state):
        result = await server.trace("PoolingServingBase._prepare_generators")
        assert "Pooling encode flow" in result
        assert "AsyncLLM.encode" in result

    @pytest.mark.asyncio
    async def test_graph_calls_for_attention_selector(self, vllm_state):
        result = await server.graph("_cached_get_attn_backend", mode="calls")
        assert "CudaPlatformBase.get_attn_backend_cls" in result
        assert "confidence=conditional" in result
        assert "FLASH_ATTN" in result

    @pytest.mark.asyncio
    async def test_graph_impact_for_rms_norm(self, vllm_state):
        result = await server.graph("rms_norm", mode="impact", depth=3)
        assert "RMSNorm.forward_native" in result
        assert "Scheduler.schedule" in result


class TestUnsupportedVllmToolModes:
    @pytest.mark.asyncio
    async def test_modules_not_supported(self, vllm_state):
        with pytest.raises(
            RuntimeError,
            match="Python module tracing is not available",
        ):
            await server.modules("nn")

    @pytest.mark.asyncio
    async def test_tests_not_supported(self, vllm_state):
        with pytest.raises(RuntimeError, match="Test discovery is not available"):
            await server.tests("rms_norm")

    @pytest.mark.asyncio
    async def test_affected_not_supported(self, vllm_state):
        with pytest.raises(
            RuntimeError,
            match="Affected-test analysis is not available",
        ):
            await server.affected("rms_norm")
