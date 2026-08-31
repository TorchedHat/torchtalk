"""Tests for operator trace/search tools."""

import asyncio

import pytest

from torchtalk import indexer
from torchtalk.tools.ops import _do_cuda_kernels, trace


@pytest.fixture
def state_without_call_graph():
    """Loaded index with no C++ call graph (background build / no libclang)."""
    s = indexer._state
    saved = (
        s.native_functions,
        s.bindings,
        s.cuda_kernels,
        s.cpp_extractor,
        s.source,
    )
    s.native_functions = {
        "add": {
            "name": "add",
            "base_name": "add",
            "signature": "add(Tensor self, Tensor other) -> Tensor",
            "dispatch": {"CPU": "add_cpu"},
            "variants": "function",
            "python_module": "",
            "structured": False,
            "structured_delegate": None,
            "tags": [],
        }
    }
    s.bindings = []
    s.cuda_kernels = [
        {"name": "add_kernel", "file_path": "/src/a.cu", "line_number": 3}
    ]
    s.cpp_extractor = None
    s.source = "/src"
    indexer._build_indexes(s)
    try:
        yield s
    finally:
        (
            s.native_functions,
            s.bindings,
            s.cuda_kernels,
            s.cpp_extractor,
            s.source,
        ) = saved
        indexer._build_indexes(s)


class TestTraceWithoutCallGraph:
    @pytest.mark.parametrize("focus", ["full", "yaml", "dispatch"])
    def test_trace_does_not_crash(self, state_without_call_graph, focus):
        out = asyncio.run(trace("add", focus=focus))
        assert "add" in out

    def test_trace_unknown_function(self, state_without_call_graph):
        out = asyncio.run(trace("definitely_not_an_op_xyz"))
        assert isinstance(out, str)


class TestCudaKernelsWithoutCallGraph:
    def test_kernel_search_does_not_crash(self, state_without_call_graph):
        out = asyncio.run(_do_cuda_kernels("add"))
        assert "add_kernel" in out


class TestTraceImplDedupe:
    def test_duplicates_deduped_before_slice(self, state_without_call_graph):
        s = state_without_call_graph
        dup = {"function_name": "add_cpu", "file_path": "/a.cpp", "line_number": 1}
        uniq = [
            {"function_name": f"add_v{i}", "file_path": "/b.cpp", "line_number": i}
            for i in range(2)
        ]
        s.native_implementations = {"add": [dup] * 10 + uniq}
        out = asyncio.run(trace("add"))
        assert "add_v0" in out
        assert "add_v1" in out


class TestTraceFuzzyLabeling:
    def test_fuzzy_dispatch_labeled_and_no_contradiction(
        self, state_without_call_graph
    ):
        s = state_without_call_graph
        s.native_functions = {}
        s.bindings = [
            {
                "python_name": "hardswish",
                "cpp_name": "qhardswish",
                "dispatch_key": "QuantizedCPU",
                "file_path": "/q.cpp",
                "line_number": 5,
            }
        ]
        from torchtalk import indexer

        indexer._build_indexes(s)
        out = asyncio.run(trace("hardswishh", focus="dispatch"))
        assert "fuzzy match" in out
        assert "Function Not Found" not in out
