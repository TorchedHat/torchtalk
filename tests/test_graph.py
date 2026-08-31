"""Tests for graph tool clamps and traversal config."""

from __future__ import annotations

import asyncio

import pytest

from torchtalk import indexer
from torchtalk.tools import graph as graph_mod
from torchtalk.tools.graph import (
    _GRAPH_HARD_DEPTH_CAP,
    _do_impact,
    _max_depth,
    _py_name_to_cpp_symbol,
    _python_callers_for,
)


class TestMaxDepth:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("TORCHTALK_GRAPH_MAX_DEPTH", raising=False)
        assert _max_depth() == 5

    def test_reads_env_override(self, monkeypatch):
        monkeypatch.setenv("TORCHTALK_GRAPH_MAX_DEPTH", "8")
        assert _max_depth() == 8

    def test_clamps_to_hard_cap(self, monkeypatch):
        monkeypatch.setenv("TORCHTALK_GRAPH_MAX_DEPTH", "100")
        assert _max_depth() == _GRAPH_HARD_DEPTH_CAP

    def test_floors_below_one(self, monkeypatch):
        monkeypatch.setenv("TORCHTALK_GRAPH_MAX_DEPTH", "0")
        assert _max_depth() == 1

    def test_falls_back_on_invalid(self, monkeypatch):
        monkeypatch.setenv("TORCHTALK_GRAPH_MAX_DEPTH", "not_a_number")
        assert _max_depth() == 5


class _FakeExtractor:
    def __init__(
        self,
        edges: dict[str, list[dict]],
        fuzzy_only: set[str] | None = None,
        coverage: dict[str, int] | None = None,
        matches: dict[str, list[str]] | None = None,
    ):
        self._edges = edges
        self._fuzzy_only = fuzzy_only or set()
        self._coverage = coverage or {}
        self._matches = matches or {}

    def get_callers(self, name: str, fuzzy: bool = True) -> list[dict]:
        if name in self._fuzzy_only and not fuzzy:
            return []
        return self._edges.get(name, [])

    def coverage_summary(self) -> dict[str, int]:
        return self._coverage

    def match_functions(self, name: str, fuzzy: bool = True) -> list[str]:
        return self._matches.get(name, [name])


@pytest.fixture
def reset_extractor():
    prior = indexer._state.cpp_extractor
    prior_src = indexer._state.source
    indexer._state.source = "/fake/source"
    indexer._state.bindings = [{"python_name": "fake"}]  # satisfies _ensure_loaded
    try:
        yield
    finally:
        indexer._state.cpp_extractor = prior
        indexer._state.source = prior_src
        indexer._state.bindings = []


class TestImpactFuzzyAllLevels:
    def test_default_fuzzy_only_at_level_one(self, reset_extractor, monkeypatch):
        # 'leaf' is reachable only via fuzzy lookup; with default
        # `fuzzy_all_levels=False` it must be reached at level 1 but not
        # propagate to level 2 lookups.
        edges = {
            "root": [{"caller": "mid", "caller_file": "/a.cpp", "caller_line": 1}],
            "mid": [
                {"caller": "fuzzy_only", "caller_file": "/b.cpp", "caller_line": 2}
            ],
        }
        indexer._state.cpp_extractor = _FakeExtractor(edges, fuzzy_only={"mid"})
        monkeypatch.setattr(graph_mod, "_cpp_status", lambda: "")
        monkeypatch.setattr(graph_mod, "coverage_note", lambda _: "", raising=False)

        out = asyncio.run(_do_impact("root", depth=3))
        # mid found at level 1 (fuzzy); but mid's edges only return data with
        # fuzzy=True. At level 2, fuzzy=False (default), so no leaf found.
        assert "`mid`" in out
        assert "`fuzzy_only`" not in out

    def test_fuzzy_all_levels_propagates(self, reset_extractor, monkeypatch):
        edges = {
            "root": [{"caller": "mid", "caller_file": "/a.cpp", "caller_line": 1}],
            "mid": [
                {"caller": "fuzzy_only", "caller_file": "/b.cpp", "caller_line": 2}
            ],
        }
        indexer._state.cpp_extractor = _FakeExtractor(edges, fuzzy_only={"mid"})
        monkeypatch.setattr(graph_mod, "_cpp_status", lambda: "")
        monkeypatch.setattr(graph_mod, "coverage_note", lambda _: "", raising=False)

        out = asyncio.run(_do_impact("root", depth=3, fuzzy_all_levels=True))
        assert "`mid`" in out
        assert "`fuzzy_only`" in out


class TestPyNameToCppSymbol:
    def test_dotted_two_part(self):
        assert _py_name_to_cpp_symbol("aten.add") == "aten::add"

    def test_drops_overload_tag(self):
        assert _py_name_to_cpp_symbol("aten.add.Tensor") == "aten::add"

    def test_bare_passes_through(self):
        assert _py_name_to_cpp_symbol("foo") == "foo"


class TestPythonCallersFor:
    @pytest.fixture(autouse=True)
    def reset(self):
        prior_edges = indexer._state.py_to_cpp_edges
        prior_by_cpp = indexer._state.by_cpp_name
        try:
            yield
        finally:
            indexer._state.py_to_cpp_edges = prior_edges
            indexer._state.by_cpp_name = prior_by_cpp

    def test_no_edges_returns_empty(self):
        indexer._state.py_to_cpp_edges = {}
        indexer._state.by_cpp_name = {}
        assert _python_callers_for("at::native::add") == []

    def test_resolves_via_known_binding(self):
        indexer._state.by_cpp_name = {"add": [{"python_name": "aten.add"}]}
        indexer._state.py_to_cpp_edges = {
            "aten::add": [{"caller_qualname": "torch.x.f", "file": "/x.py", "line": 12}]
        }
        result = _python_callers_for("at::native::add")
        assert result == [{"caller_qualname": "torch.x.f", "file": "/x.py", "line": 12}]

    def test_falls_back_to_aten_prefix(self):
        # No binding but bare-name + aten:: guess hits.
        indexer._state.by_cpp_name = {}
        indexer._state.py_to_cpp_edges = {
            "aten::relu": [
                {"caller_qualname": "torch.nn.f", "file": "/nn.py", "line": 5}
            ]
        }
        result = _python_callers_for("at::native::relu")
        assert len(result) == 1
        assert result[0]["caller_qualname"] == "torch.nn.f"


class TestImpactWalkPython:
    def test_walk_python_emits_source_callers(self, reset_extractor, monkeypatch):
        edges = {
            "add": [{"caller": "outer_fn", "caller_file": "/.cpp", "caller_line": 1}]
        }
        indexer._state.cpp_extractor = _FakeExtractor(edges, fuzzy_only=set())
        indexer._state.by_cpp_name = {"outer_fn": [{"python_name": "aten.add"}]}
        indexer._state.py_to_cpp_edges = {
            "aten::add": [
                {
                    "caller_qualname": "torch.functional.relu",
                    "file": "/torch/functional.py",
                    "line": 42,
                }
            ]
        }
        monkeypatch.setattr(graph_mod, "_cpp_status", lambda: "")
        monkeypatch.setattr(graph_mod, "coverage_note", lambda _: "", raising=False)

        out = asyncio.run(_do_impact("add", depth=2, walk_python=True))
        assert "Python Source Callers" in out
        assert "torch.functional.relu" in out

    def test_walk_python_off_omits_section(self, reset_extractor, monkeypatch):
        edges = {
            "add": [{"caller": "outer_fn", "caller_file": "/.cpp", "caller_line": 1}]
        }
        indexer._state.cpp_extractor = _FakeExtractor(edges, fuzzy_only=set())
        indexer._state.by_cpp_name = {"outer_fn": [{"python_name": "aten.add"}]}
        indexer._state.py_to_cpp_edges = {
            "aten::add": [
                {
                    "caller_qualname": "torch.functional.relu",
                    "file": "/torch/functional.py",
                    "line": 42,
                }
            ]
        }
        monkeypatch.setattr(graph_mod, "_cpp_status", lambda: "")
        monkeypatch.setattr(graph_mod, "coverage_note", lambda _: "", raising=False)

        out = asyncio.run(_do_impact("add", depth=2, walk_python=False))
        assert "Python Source Callers" not in out


class TestImpactDepthClamp:
    def test_caller_above_max_depth_is_truncated(self, reset_extractor, monkeypatch):
        # Chain a -> b -> c -> d -> e -> f. With env cap at 3, only the first
        # three callers should appear.
        edges = {
            "a": [{"caller": "b", "caller_file": "/.cpp", "caller_line": 1}],
            "b": [{"caller": "c", "caller_file": "/.cpp", "caller_line": 1}],
            "c": [{"caller": "d", "caller_file": "/.cpp", "caller_line": 1}],
            "d": [{"caller": "e", "caller_file": "/.cpp", "caller_line": 1}],
            "e": [{"caller": "f", "caller_file": "/.cpp", "caller_line": 1}],
        }
        indexer._state.cpp_extractor = _FakeExtractor(edges, fuzzy_only=set())
        monkeypatch.setenv("TORCHTALK_GRAPH_MAX_DEPTH", "3")
        monkeypatch.setattr(graph_mod, "_cpp_status", lambda: "")
        monkeypatch.setattr(graph_mod, "coverage_note", lambda _: "", raising=False)

        out = asyncio.run(_do_impact("a", depth=10, fuzzy_all_levels=True))
        assert "`b`" in out and "`c`" in out and "`d`" in out
        assert "`e`" not in out and "`f`" not in out


class TestCalledByCudaGap:
    """_do_called_by must disclose unindexed .cu TUs instead of a confident
    empty/narrow answer."""

    def _setup(self, monkeypatch, edges, coverage):
        indexer._state.cpp_extractor = _FakeExtractor(edges, coverage=coverage)
        monkeypatch.setattr(graph_mod, "_cpp_status", lambda: "")
        monkeypatch.setattr(graph_mod, "coverage_note", lambda _: "", raising=False)

    def test_empty_result_mentions_cuda_gap(self, reset_extractor, monkeypatch):
        self._setup(monkeypatch, {}, {"cu_unindexed": 397})
        out = asyncio.run(graph_mod._do_called_by("GeluCUDAKernelImpl"))
        assert "No inbound callers" in out
        assert "397 .cu TUs unindexed" in out
        assert "CUDA callers unknown" in out

    def test_cuda_adjacent_result_mentions_gap(self, reset_extractor, monkeypatch):
        edges = {
            "gelu_out": [
                {
                    "caller": "GeluKernelImpl",
                    "caller_file": "/src/aten/native/cuda/Gelu.cu",
                    "caller_line": 5,
                }
            ]
        }
        self._setup(monkeypatch, edges, {"cu_unindexed": 10})
        out = asyncio.run(graph_mod._do_called_by("gelu_out"))
        assert "GeluKernelImpl" in out
        assert "10 .cu TUs unindexed" in out

    def test_cuda_named_query_mentions_gap(self, reset_extractor, monkeypatch):
        edges = {
            "launch_cuda_gemm": [
                {"caller": "gemm_entry", "caller_file": "/a.cpp", "caller_line": 1}
            ]
        }
        self._setup(monkeypatch, edges, {"cu_unindexed": 3})
        out = asyncio.run(graph_mod._do_called_by("launch_cuda_gemm"))
        assert "3 .cu TUs unindexed" in out

    def test_non_cuda_result_omits_gap(self, reset_extractor, monkeypatch):
        edges = {"foo": [{"caller": "bar", "caller_file": "/a.cpp", "caller_line": 1}]}
        self._setup(monkeypatch, edges, {"cu_unindexed": 10})
        out = asyncio.run(graph_mod._do_called_by("foo"))
        assert "CUDA callers unknown" not in out

    def test_no_gap_no_note(self, reset_extractor, monkeypatch):
        self._setup(monkeypatch, {}, {})
        out = asyncio.run(graph_mod._do_called_by("GeluCUDAKernelImpl"))
        assert "CUDA callers unknown" not in out


class TestFuzzyResolutionDisclosure:
    """Fuzzy graph lookups disclose resolution instead of merging neighborhoods."""

    def _setup(self, monkeypatch, ext):
        indexer._state.cpp_extractor = ext
        monkeypatch.setattr(graph_mod, "_cpp_status", lambda: "")
        monkeypatch.setattr(graph_mod, "coverage_note", lambda _: "", raising=False)

    def test_fuzzy_resolution_disclosed(self, reset_extractor, monkeypatch):
        edges = {
            "at::native::gemm": [
                {"caller": "linear_fwd", "caller_file": "/a.cpp", "caller_line": 1}
            ]
        }
        matches = {"gemm": ["at::native::gemm", "at::cpublas::gemm"]}
        self._setup(monkeypatch, _FakeExtractor(edges, matches=matches))

        out = asyncio.run(graph_mod._do_called_by("gemm"))
        assert "Resolved 'gemm' → 'at::native::gemm' (fuzzy)" in out
        assert "at::cpublas::gemm" in out
        assert "linear_fwd" in out

    def test_multiple_matches_not_merged(self, reset_extractor, monkeypatch):
        edges = {
            "at::native::gemm": [
                {"caller": "linear_fwd", "caller_file": "/a.cpp", "caller_line": 1}
            ],
            "at::cpublas::gemm": [
                {"caller": "blas_entry", "caller_file": "/b.cpp", "caller_line": 2}
            ],
        }
        matches = {"gemm": ["at::native::gemm", "at::cpublas::gemm"]}
        self._setup(monkeypatch, _FakeExtractor(edges, matches=matches))

        out = asyncio.run(graph_mod._do_called_by("gemm"))
        assert "linear_fwd" in out
        assert "blas_entry" not in out

    def test_exact_match_has_no_disclosure(self, reset_extractor, monkeypatch):
        edges = {
            "addmm": [{"caller": "linear", "caller_file": "/a.cpp", "caller_line": 1}]
        }
        self._setup(monkeypatch, _FakeExtractor(edges))

        out = asyncio.run(graph_mod._do_called_by("addmm"))
        assert "Resolved" not in out
        assert "linear" in out

    def test_impact_discloses_resolution(self, reset_extractor, monkeypatch):
        edges = {
            "at::native::gemm": [
                {"caller": "linear_fwd", "caller_file": "/a.cpp", "caller_line": 1}
            ]
        }
        matches = {"gemm": ["at::native::gemm"]}
        self._setup(monkeypatch, _FakeExtractor(edges, matches=matches))

        out = asyncio.run(_do_impact("gemm", depth=1))
        assert "Resolved 'gemm' → 'at::native::gemm' (fuzzy)" in out
        assert "linear_fwd" in out


class TestResolutionSkipsEmptyCandidates:
    def test_first_candidate_with_callers_wins(self, reset_extractor, monkeypatch):
        # rank-1 candidate has no callers; rank-2 holds the real answer
        edges = {
            "std::at::addmm": [
                {
                    "caller": "at::native::linear",
                    "caller_file": "/l.cpp",
                    "caller_line": 9,
                }
            ]
        }
        matches = {"addmm": ["at::addmm", "std::at::addmm"]}
        indexer._state.cpp_extractor = _FakeExtractor(edges, matches=matches)
        monkeypatch.setattr(graph_mod, "_cpp_status", lambda: "")
        monkeypatch.setattr(graph_mod, "coverage_note", lambda _: "", raising=False)

        out = asyncio.run(graph_mod._do_called_by("addmm"))
        assert "Resolved 'addmm' → 'std::at::addmm' (fuzzy)" in out
        assert "at::native::linear" in out
        assert "`at::addmm`" in out  # listed as another match, not merged

    def test_empty_result_still_disclosed(self, reset_extractor, monkeypatch):
        matches = {"addmm": ["at::addmm", "std::at::addmm"]}
        indexer._state.cpp_extractor = _FakeExtractor({}, matches=matches)
        monkeypatch.setattr(graph_mod, "_cpp_status", lambda: "")
        monkeypatch.setattr(graph_mod, "coverage_note", lambda _: "", raising=False)

        out = asyncio.run(graph_mod._do_called_by("addmm"))
        assert "No inbound callers" in out
        assert "Resolved 'addmm'" in out
        assert "std::at::addmm" in out


class TestImpactIncludesSeed:
    def test_walk_python_includes_seed_edges(self, reset_extractor, monkeypatch):
        # Seed has a Python source caller but zero C++ callers at depth>0.
        edges = {
            "at::native::gelu": [
                {"caller": "gelu_meta", "caller_file": "/m.cpp", "caller_line": 1}
            ]
        }
        indexer._state.cpp_extractor = _FakeExtractor(edges)
        monkeypatch.setattr(
            indexer._state,
            "py_to_cpp_edges",
            {
                "aten::gelu": [
                    {
                        "caller_qualname": "torch.nn.GELU.forward",
                        "file": "/g.py",
                        "line": 7,
                    }
                ]
            },
        )
        monkeypatch.setattr(
            indexer._state,
            "by_cpp_name",
            {"gelu": [{"python_name": "aten.gelu", "cpp_name": "gelu"}]},
        )
        monkeypatch.setattr(graph_mod, "_cpp_status", lambda: "")
        monkeypatch.setattr(graph_mod, "coverage_note", lambda _: "", raising=False)

        out = asyncio.run(
            _do_impact("at::native::gelu", depth=1, walk_python=True, focus="full")
        )
        assert "Python Source Callers" in out
        assert "torch.nn.GELU.forward" in out
