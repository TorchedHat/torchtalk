"""Tests for the affected-tests tool wrapper."""

import asyncio
from unittest.mock import MagicMock

import pytest

from torchtalk import indexer
from torchtalk.tools.affected import _do_affected, _split_funcs


class TestSplitFuncs:
    def test_simple_split(self):
        assert _split_funcs("add,mul,relu") == ["add", "mul", "relu"]

    def test_template_preserved(self):
        assert _split_funcs("std::vector<int>,foo") == ["std::vector<int>", "foo"]

    def test_nested_templates(self):
        assert _split_funcs("A<B<C>>,D") == ["A<B<C>>", "D"]

    def test_whitespace_stripped(self):
        assert _split_funcs(" add , mul ") == ["add", "mul"]

    def test_empty_returns_empty(self):
        assert _split_funcs("") == []

    def test_trailing_comma(self):
        assert _split_funcs("add,") == ["add"]

    def test_unbalanced_open_bracket_keeps_nesting(self):
        result = _split_funcs("A<B,C")
        assert result == ["A<B,C"]

    def test_unbalanced_close_bracket_floors_at_zero(self):
        result = _split_funcs("A>,B")
        assert result == ["A>", "B"]


@pytest.fixture
def affected_state(mock_state):
    s = mock_state
    s.bindings = [{"python_name": "x"}]
    s.native_functions = {}
    s.cpp_extractor = MagicMock()
    s.cpp_building = False
    s.test_classes = {}
    s.test_files = {}
    s.opinfo_registry = {}
    s.opinfo_alias_map = {}
    s.opinfo_test_files = set()
    s.test_attr_index = {}
    s.pytorch_source = "/fake"
    indexer._build_indexes(s)
    return s


def _canned_result(**overrides):
    base = {
        "input_functions": ["foo_kernel"],
        "callers_walked": 5,
        "bindings_matched": [
            {"python_name": "foo", "cpp_name": "foo_kernel", "dispatch_key": "CPU"}
        ],
        "python_apis": ["foo", "bar"],
        "api_tier": {"foo": "precise", "bar": "fuzzy"},
        "api_sources": {"foo": ["call_graph"], "bar": ["cohort"]},
        "test_runs": [
            {"file": "test/test_x.py", "included_classes": ["TestFoo"]},
        ],
        "opinfo_runs": [],
        "function_hits": {"test/test_x.py": {"TestFoo": ["test_bar"]}},
    }
    base.update(overrides)
    return base


class TestDoAffected:
    def test_renders_tier_counts(self, affected_state, monkeypatch):
        monkeypatch.setattr(
            "torchtalk.tools.affected.affected_tests",
            lambda **kw: _canned_result(),
        )
        out = asyncio.run(_do_affected("foo_kernel"))
        assert "1 precise" in out
        assert "1 fuzzy" in out

    def test_pytest_node_ids(self, affected_state, monkeypatch):
        monkeypatch.setattr(
            "torchtalk.tools.affected.affected_tests",
            lambda **kw: _canned_result(),
        )
        out = asyncio.run(_do_affected("foo_kernel"))
        assert "test/test_x.py::TestFoo::test_bar" in out

    def test_opinfo_run_rendered(self, affected_state, monkeypatch):
        monkeypatch.setattr(
            "torchtalk.tools.affected.affected_tests",
            lambda **kw: _canned_result(
                opinfo_runs=[{"files": ["test/test_ops.py"], "k": "softmax"}],
            ),
        )
        out = asyncio.run(_do_affected("foo_kernel"))
        assert '-k "softmax"' in out

    def test_no_runs_message(self, affected_state, monkeypatch):
        monkeypatch.setattr(
            "torchtalk.tools.affected.affected_tests",
            lambda **kw: _canned_result(test_runs=[], opinfo_runs=[], function_hits={}),
        )
        out = asyncio.run(_do_affected("foo_kernel"))
        assert "No matching test runs" in out

    def test_cpp_unavailable(self, affected_state):
        affected_state.cpp_extractor = None
        affected_state.cpp_building = False
        out = asyncio.run(_do_affected("foo_kernel"))
        assert "compile_commands.json" in out
