"""Integration tests for the vLLM adapter and static index."""

from __future__ import annotations

import pytest

from torchtalk.adapters import get_adapter, list_adapters
from torchtalk.analysis.vllm_index import _build_graph, _build_lookup_indexes
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

    def test_hybrid_discovery_adds_non_curated_api_inventory(self, vllm_state):
        api_names = {
            record["name"] for record in vllm_state.entities["api_entrypoints"]
        }
        assert "LLM.enqueue" in api_names
        assert "LLM.wait_for_completion" in api_names

    def test_graph_builder_handles_offline_mixin_layout(self):
        def record(family: str, name: str, *, suffix: str | None = None):
            record_id = f"{family}:{name}"
            if suffix:
                record_id += f":{suffix}"
            return {
                "id": record_id,
                "family": family,
                "name": name,
                "file_path": f"/tmp/{family}.py",
                "line_number": 1,
                "kind": family,
                "details": {},
            }

        records_by_family = {
            "api_entrypoints": [
                record("api_entrypoints", "LLM.generate"),
                record(
                    "api_entrypoints",
                    "OpenAIServingChat._create_chat_completion",
                ),
                record(
                    "api_entrypoints",
                    "PoolingServingBase._prepare_generators",
                ),
            ],
            "request_pipeline_nodes": [
                record(
                    "request_pipeline_nodes",
                    "OfflineInferenceMixin._run_completion",
                ),
                record(
                    "request_pipeline_nodes",
                    "OfflineInferenceMixin._add_completion_requests",
                ),
                record("request_pipeline_nodes", "OfflineInferenceMixin._add_request"),
                record("request_pipeline_nodes", "OfflineInferenceMixin._run_engine"),
                record("request_pipeline_nodes", "LLMEngine.add_request"),
                record("request_pipeline_nodes", "LLMEngine.step"),
                record("request_pipeline_nodes", "AsyncLLM.generate"),
                record("request_pipeline_nodes", "AsyncLLM.add_request"),
                record("request_pipeline_nodes", "AsyncLLM.encode"),
                record("request_pipeline_nodes", "InputProcessor.process_inputs"),
                record("request_pipeline_nodes", "EngineCore.add_request"),
                record("request_pipeline_nodes", "Scheduler.schedule"),
                record("request_pipeline_nodes", "get_attn_backend"),
                record("request_pipeline_nodes", "_cached_get_attn_backend"),
            ],
            "layer_nodes": [record("layer_nodes", "RMSNorm.forward_native")],
            "platform_defaults": [
                record("platform_defaults", "Platform.get_attn_backend_cls"),
                record("platform_defaults", "CudaPlatformBase.get_attn_backend_cls"),
                record("platform_defaults", "RocmPlatform.get_attn_backend_cls"),
                record("platform_defaults", "XPUPlatform.get_attn_backend_cls"),
                record("platform_defaults", "CpuPlatform.get_attn_backend_cls"),
            ],
            "model_architectures": [],
            "attention_backends": [
                record("attention_backends", "FLASH_ATTN"),
            ],
            "ir_ops": [record("ir_ops", "rms_norm")],
            "ir_providers": [record("ir_providers", "rms_norm::vllm_c")],
            "custom_ops": [
                record("custom_ops", "unquantized_fused_moe"),
            ],
            "pluggable_layers": [record("pluggable_layers", "fused_moe")],
            "torch_custom_ops": [
                record("torch_custom_ops", "rms_norm", suffix="libtorch_stable"),
                record("torch_custom_ops", "moe_sum", suffix="moe"),
            ],
        }

        lookup_indexes = _build_lookup_indexes(records_by_family)
        graph_payload = _build_graph(records_by_family, lookup_indexes)
        offline_trace = graph_payload["proof_traces"]["offline_generate"]["steps"]
        offline_nodes = [
            graph_payload["graph_nodes"][node_id]["name"] for node_id in offline_trace
        ]

        assert "OfflineInferenceMixin._run_completion" in offline_nodes
        assert "OfflineInferenceMixin._add_completion_requests" in offline_nodes
        assert "OfflineInferenceMixin._run_engine" in offline_nodes


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

    @pytest.mark.asyncio
    async def test_hybrid_discovery_searches_new_api_nodes(self, vllm_state):
        result = await vllm_tools.search("wait_for_completion", mode="apis", limit=10)
        assert "LLM.wait_for_completion" in result
