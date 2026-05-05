"""Tests for graph tool clamps and traversal config."""

from __future__ import annotations

import asyncio

import pytest

from torchtalk import indexer
from torchtalk.tools import graph as graph_mod
from torchtalk.tools.graph import _GRAPH_HARD_DEPTH_CAP, _do_impact, _max_depth


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
    def __init__(self, edges: dict[str, list[dict]], fuzzy_only: set[str]):
        self._edges = edges
        self._fuzzy_only = fuzzy_only

    def get_callers(self, name: str, fuzzy: bool = True) -> list[dict]:
        if name in self._fuzzy_only and not fuzzy:
            return []
        return self._edges.get(name, [])


@pytest.fixture
def reset_extractor():
    prior = indexer._state.cpp_extractor
    prior_src = indexer._state.pytorch_source
    indexer._state.pytorch_source = "/fake/source"
    indexer._state.bindings = [{"python_name": "fake"}]  # satisfies _ensure_loaded
    try:
        yield
    finally:
        indexer._state.cpp_extractor = prior
        indexer._state.pytorch_source = prior_src
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
