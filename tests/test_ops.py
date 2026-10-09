"""Tests for operator trace/search tools."""

import asyncio

import pytest

from torchtalk import indexer
from torchtalk.tools.ops import (
    _do_cuda_kernels,
    _do_search_bindings,
    _get_native_func,
    _similar_functions,
    trace,
)


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
        s.registrations,
        s.py_functions,
        s.py_classes,
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
            s.registrations,
            s.py_functions,
            s.py_classes,
        ) = saved
        indexer._build_indexes(s)


class TestTraceWithoutCallGraph:
    @pytest.mark.parametrize("focus", ["full", "yaml", "dispatch"])
    def test_trace_does_not_crash(self, state_without_call_graph, focus):
        out = asyncio.run(trace("add", focus=focus))
        assert "add" in out

    def test_trace_unknown_function(self, state_without_call_graph):
        out = asyncio.run(trace("definitely_not_an_op_xyz"))
        assert "definitely_not_an_op_xyz" in out
        assert "not found" in out.lower()


class TestTracePythonRegistrations:
    """A registration record links an op name to its Python target and onward."""

    @staticmethod
    def _record(key, target, file, line=40):
        return {
            "registry": "custom_ops",
            "key": key,
            "target": target,
            "kind": "resolved",
            "via": "call",
            "file": file,
            "line": line,
        }

    @staticmethod
    def _func(name, module, line, calls=()):
        from torchtalk.analysis.python_analyzer import PyBinding, PyFunction

        return PyFunction(
            name=name,
            qualified_name=f"{module}.{name}",
            file_path=f"/src/{module.replace('.', '/')}.py",
            line_number=line,
            cpp_bindings=[PyBinding(cpp_symbol=c, line=line + 3) for c in calls],
        )

    @staticmethod
    def _binding(python_name, cpp_name, line, dispatch_key="CUDA"):
        return {
            "python_name": python_name,
            "cpp_name": cpp_name,
            "type": "torch_library_impl",
            "dispatch_key": dispatch_key,
            "file_path": "/src/csrc/bindings.cpp",
            "line_number": line,
        }

    def test_registered_op_traces(self, state_without_call_graph):
        s = state_without_call_graph
        s.registrations = {
            "records": [self._record("fused_norm", "_fused_norm_impl", "pkg/ops.py")]
        }
        impl = self._func("_fused_norm_impl", "pkg.ops", 12, ["_C::rms_norm"])
        s.py_functions = {"_fused_norm_impl": [impl]}
        s.bindings = [self._binding("_C.rms_norm", "rms_norm", 7)]
        indexer._build_indexes(s)

        out = asyncio.run(trace("fused_norm"))
        assert "Python Registrations" in out
        assert "`custom_ops` → `pkg.ops._fused_norm_impl` (`pkg/ops.py:12`)" in out
        assert "csrc/bindings.cpp:7" in out
        assert "not found" not in out.lower()

        assert "csrc/bindings.cpp:7" in asyncio.run(trace("fused_norm", "dispatch"))
        assert "Python Registrations" not in asyncio.run(trace("fused_norm", "yaml"))

    def test_bare_pybind_call(self, state_without_call_graph):
        s = state_without_call_graph
        s.registrations = {"records": [self._record("op", "op_impl", "pkg/ops.py")]}
        s.py_functions = {"op_impl": [self._func("op_impl", "pkg.ops", 5, ["foo"])]}
        s.bindings = [
            {
                "python_name": "foo",
                "cpp_name": "foo_cuda_impl",
                "type": "pybind_function",
                "file_path": "/src/csrc/pybind.cpp",
                "line_number": 9,
            }
        ]
        indexer._build_indexes(s)
        assert "csrc/pybind.cpp:9" in asyncio.run(trace("op"))

    def test_class_target_methods(self, state_without_call_graph):
        from torchtalk.analysis.python_analyzer import PyClass

        s = state_without_call_graph
        s.registrations = {
            "records": [self._record("norm", "pkg.layers.Norm", "pkg/layers.py")]
        }
        fwd = self._func("forward_cuda", "pkg.layers", 30, ["_C::rms_norm"])
        cls = PyClass(
            name="Norm",
            qualified_name="pkg.layers.Norm",
            file_path="/src/pkg/layers.py",
            line_number=20,
            bases=["nn.Module"],
            decorators=[],
            docstring=None,
            methods=[fwd],
        )
        s.py_functions = {}
        s.py_classes = {"Norm": [cls]}
        s.bindings = [self._binding("_C.rms_norm", "rms_norm", 7)]
        indexer._build_indexes(s)

        out = asyncio.run(trace("norm"))
        assert "`pkg.layers.Norm` (`pkg/layers.py:20`)" in out
        assert "csrc/bindings.cpp:7" in out

    def test_other_file_not_guessed(self, state_without_call_graph):
        s = state_without_call_graph
        s.registrations = {"records": [self._record("op", "impl", "pkg/a.py", 3)]}
        s.py_functions = {
            "impl": [self._func("impl", "pkg.b", 8), self._func("impl", "pkg.c", 9)]
        }
        out = asyncio.run(trace("op"))
        assert "`custom_ops` → `impl` (`pkg/a.py:3`)" in out
        assert "pkg/b.py" not in out and "pkg/c.py" not in out

        s.py_functions["impl"].append(self._func("impl", "pkg.a", 3))
        out = asyncio.run(trace("op"))
        assert "`pkg.a.impl` (`pkg/a.py:3`)" in out

    def test_unresolved_target(self, state_without_call_graph):
        s = state_without_call_graph
        s.registrations = {
            "records": [self._record("triton_only", "triton_impl", "pkg/triton.py", 3)]
        }
        s.py_functions = {}
        out = asyncio.run(trace("triton_only"))
        assert "Python Registrations" in out
        assert "`custom_ops` → `triton_impl` (`pkg/triton.py:3`)" in out
        assert "not found" not in out.lower()


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


@pytest.fixture
def ops_state(mock_state):
    s = mock_state
    s.native_functions = {
        "addmm": {"name": "addmm", "base_name": "addmm"},
        "add_out": {"name": "add_out", "base_name": "add_out"},
        "softmax.Tensor": {"name": "softmax.Tensor", "base_name": "softmax"},
        "relu": {"name": "relu", "base_name": "relu"},
        "softmin": {"name": "softmin", "base_name": "softmin"},
        "softmax": {"name": "softmax", "base_name": "softmax"},
    }
    s.bindings = [
        {
            "python_name": "aten.add",
            "cpp_name": "add_cpu",
            "dispatch_key": "CPU",
            "file_path": "/src/a.cpp",
            "line_number": 10,
        },
        {
            "python_name": "aten.add",
            "cpp_name": "add_cuda",
            "dispatch_key": "CUDA",
            "file_path": "/src/b.cu",
            "line_number": 20,
        },
        {
            "python_name": "aten.add",
            "cpp_name": "add_cpu",
            "dispatch_key": "CPU",
            "file_path": "/src/a_dup.cpp",
            "line_number": 99,
        },
        {
            "python_name": "aten.mul",
            "cpp_name": "mul_cpu",
            "dispatch_key": "CPU",
            "file_path": "/src/c.cpp",
            "line_number": 30,
        },
        {
            "python_name": "aten.add",
            "cpp_name": "add_mps",
            "dispatch_key": "MPS",
            "file_path": "/src/d.mm",
            "line_number": 40,
        },
        {
            "python_name": "aten.add",
            "cpp_name": "add_quantized",
            "dispatch_key": "QuantizedCPU",
            "file_path": "/src/e.cpp",
            "line_number": 50,
        },
        {
            "python_name": "aten.add",
            "cpp_name": "add_mkldnn",
            "dispatch_key": "MkldnnCPU",
            "file_path": "/src/f.cpp",
            "line_number": 60,
        },
    ]
    s.pytorch_source = "/src"
    indexer._build_indexes(s)
    return s


class TestGetNativeFunc:
    def test_exact_key(self, ops_state):
        result = _get_native_func("relu")
        assert result is not None
        assert result["name"] == "relu"

    def test_case_insensitive_key(self, ops_state):
        # Key and base_name differ, so only the key arm can match.
        ops_state.native_functions["Foo_Bar"] = {"name": "Foo_Bar", "base_name": "zzz"}
        result = _get_native_func("foo_bar")
        assert result is not None
        assert result["name"] == "Foo_Bar"

    def test_base_name_fallback(self, ops_state):
        # Drop the exact "softmax" key so lookup walks to base_name.
        del ops_state.native_functions["softmax"]
        result = _get_native_func("softmax")
        assert result is not None
        assert result["name"] == "softmax.Tensor"
        assert result["base_name"] == "softmax"

    def test_substring_shortest_key_wins(self, ops_state):
        result = _get_native_func("add")
        assert result is not None
        assert result["name"] == "addmm"

    def test_no_match_returns_none(self, ops_state):
        assert _get_native_func("zzz_not_an_op") is None


class TestSimilarFunctions:
    def test_substring_suggestions(self, ops_state):
        result = _similar_functions("soft")
        assert "softmax" in result
        assert "softmin" in result
        assert "relu" not in result

    def test_typo_via_levenshtein(self, ops_state):
        result = _similar_functions("sofmax")
        assert "softmax" in result

    def test_skips_keys_of_length_fifty_or_more(self, ops_state):
        long_key = "a" * 49 + "c"  # 50 chars — excluded by len(key) < 50
        short_key = "a" * 47 + "b"
        ops_state.native_functions[long_key] = {"name": long_key, "base_name": long_key}
        ops_state.native_functions[short_key] = {
            "name": short_key,
            "base_name": short_key,
        }
        query = "a" * 47 + "x"
        result = _similar_functions(query)
        assert long_key not in result
        assert short_key in result

    def test_dedups_key_matched_by_both_passes(self, ops_state):
        # "softmaxx" matches on substring and again on levenshtein distance 1.
        ops_state.native_functions["softmaxx"] = {
            "name": "softmaxx",
            "base_name": "softmaxx",
        }
        result = _similar_functions("softmax")
        assert result.count("softmaxx") == 1

    def test_limit(self, ops_state):
        for i in range(15):
            name = f"fn_{i:02d}"
            ops_state.native_functions[name] = {"name": name, "base_name": name}
        assert len(_similar_functions("fn_", limit=3)) == 3


class TestSearchBindings:
    def test_matches_python_name(self, ops_state):
        out = asyncio.run(_do_search_bindings("aten.add"))
        assert "aten.add" in out
        assert "add_cpu" in out
        assert "aten.mul" not in out

    def test_matches_cpp_name(self, ops_state):
        out = asyncio.run(_do_search_bindings("add_cuda"))
        assert "add_cuda" in out
        assert "Found 1 binding(s)" in out
        assert "add_cpu" not in out

    def test_backend_filter(self, ops_state):
        out = asyncio.run(_do_search_bindings("add", backend="CUDA"))
        assert "(backend: CUDA)" in out
        assert "add_cuda" in out
        assert "add_cpu" not in out
        assert "aten.mul" not in out

    def test_dedup_by_name_and_dispatch_key(self, ops_state):
        out = asyncio.run(_do_search_bindings("add_cpu"))
        assert "Found 1 binding(s)" in out
        assert "a_dup.cpp" not in out

    def test_truncation_footer(self, ops_state):
        out = asyncio.run(_do_search_bindings("add", limit=2))
        assert "*Showing 2 of 5 results*" in out

    def test_no_match_with_backend(self, ops_state):
        out = asyncio.run(_do_search_bindings("gelu", backend="mps"))
        assert out == "No bindings found matching 'gelu' with backend 'mps'."

    def test_no_match_without_backend(self, ops_state):
        out = asyncio.run(_do_search_bindings("gelu"))
        assert out == "No bindings found matching 'gelu'."
