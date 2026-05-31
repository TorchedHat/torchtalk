"""Quality gates for TorchTalk's vLLM mapping."""

from __future__ import annotations

import pytest

from torchtalk import server

from .conftest import get_vllm_path

pytestmark = pytest.mark.skipif(
    get_vllm_path() is None,
    reason="VLLM_SOURCE environment variable not set",
)

MIN_ENTITY_COUNTS = {
    "api_entrypoints": 10,
    "request_pipeline_nodes": 10,
    "layer_nodes": 1,
    "platform_defaults": 5,
    "model_architectures": 300,
    "attention_backends": 25,
    "ir_ops": 2,
    "ir_providers": 6,
    "custom_ops": 20,
    "pluggable_layers": 10,
    "torch_custom_ops": 100,
}

EXPECTED_PROOF_TRACES = {
    "attention_backend_selection",
    "chat_completion_to_rms_norm",
    "fused_moe_family",
    "offline_generate",
    "pooling_encode",
}

MIN_GRAPH_EDGES = 40
MIN_CONDITIONAL_EDGES = 20
MIN_QUALITY_SCORE = 90


def compute_quality_metrics(vllm_state) -> dict[str, int | float]:
    graph_edges = len(vllm_state.graph_edges)
    conditional_edges = sum(
        1 for edge in vllm_state.graph_edges if edge.get("conditions")
    )
    evidence_edges = sum(1 for edge in vllm_state.graph_edges if edge.get("evidence"))
    entities_meeting_thresholds = sum(
        1
        for family, threshold in MIN_ENTITY_COUNTS.items()
        if vllm_state.entity_counts.get(family, 0) >= threshold
    )

    return {
        "graph_edges": graph_edges,
        "conditional_edges": conditional_edges,
        "evidence_edges": evidence_edges,
        "entity_threshold_checks": entities_meeting_thresholds,
        "entity_threshold_total": len(MIN_ENTITY_COUNTS),
        "proof_trace_count": len(vllm_state.proof_traces),
    }


class TestVllmQualityGates:
    def test_entity_counts_meet_thresholds(self, vllm_state):
        metrics = compute_quality_metrics(vllm_state)
        assert metrics["entity_threshold_checks"] == metrics["entity_threshold_total"]

    def test_proof_traces_match_expected_set(self, vllm_state):
        assert set(vllm_state.proof_traces) == EXPECTED_PROOF_TRACES

    def test_graph_has_enough_conditional_and_evidenced_edges(self, vllm_state):
        metrics = compute_quality_metrics(vllm_state)
        assert metrics["graph_edges"] >= MIN_GRAPH_EDGES
        assert metrics["conditional_edges"] >= MIN_CONDITIONAL_EDGES
        assert metrics["evidence_edges"] == metrics["graph_edges"]

    @pytest.mark.asyncio
    async def test_quality_oracle_queries(self, vllm_state):
        chat_trace = await server.trace("OpenAIServingChat._create_chat_completion")
        offline_trace = await server.trace("LLM.generate")
        pooling_trace = await server.trace("PoolingServingBase._prepare_generators")
        selector_graph = await server.graph("_cached_get_attn_backend", mode="calls")
        model_search = await server.search("LlamaForCausalLM", mode="models")

        assert "AsyncLLM.generate" in chat_trace
        assert "torch.ops._C.rms_norm" in chat_trace
        assert "LLMEngine.add_request" in offline_trace
        assert "AsyncLLM.encode" in pooling_trace
        assert "confidence=conditional" in selector_graph
        assert "FLASH_ATTN" in selector_graph
        assert "model_executor/models/registry.py" in model_search

    @pytest.mark.asyncio
    async def test_composite_quality_score_is_high(self, vllm_state):
        metrics = compute_quality_metrics(vllm_state)
        checks = [
            metrics["entity_threshold_checks"] == metrics["entity_threshold_total"],
            set(vllm_state.proof_traces) == EXPECTED_PROOF_TRACES,
            metrics["graph_edges"] >= MIN_GRAPH_EDGES,
            metrics["conditional_edges"] >= MIN_CONDITIONAL_EDGES,
            metrics["evidence_edges"] == metrics["graph_edges"],
            "AsyncLLM.generate"
            in await server.trace("OpenAIServingChat._create_chat_completion"),
            "LLMEngine.add_request" in await server.trace("LLM.generate"),
            "FLASH_ATTN"
            in await server.graph("_cached_get_attn_backend", mode="calls"),
            "torch.ops._C.rms_norm" in await server.search("rms_norm", mode="bindings"),
        ]
        quality_score = round(100 * (sum(checks) / len(checks)))
        assert quality_score >= MIN_QUALITY_SCORE
