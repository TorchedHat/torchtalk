"""Regression tests: empty-input guards and limit clamps on tool entry points."""

import asyncio

import pytest

from torchtalk import indexer
from torchtalk.tools.graph import _do_called_by, _do_calls, _do_impact
from torchtalk.tools.modules import _do_trace_module
from torchtalk.tools.ops import _do_cuda_kernels, _do_search_bindings, trace
from torchtalk.tools.tests import _do_find_similar_tests, _do_test_file_info


@pytest.fixture
def loaded_state():
    s = indexer._state
    saved = (
        s.native_functions,
        s.bindings,
        s.cuda_kernels,
        s.py_classes,
        s.test_files,
        s.cpp_extractor,
        s.test_functions,
        s.test_classes,
        s.opinfo_registry,
    )
    s.native_functions = {"add": {"base_name": "add", "dispatch": {}}}
    s.bindings = [
        {
            "python_name": "aten.add",
            "cpp_name": "add",
            "dispatch_key": "CPU",
            "file_path": "/f.cpp",
            "line_number": 1,
        }
    ]
    s.cuda_kernels = [
        {"name": "add_kernel", "file_path": "/a.cu", "line_number": 1},
        {"name": "mul_kernel", "file_path": "/a.cu", "line_number": 9},
    ]
    s.py_classes = {"Linear": [{"name": "Linear"}]}
    s.test_files = {"test/test_x.py": {"path": "test/test_x.py"}}
    s.cpp_extractor = None
    indexer._build_indexes(s)
    try:
        yield s
    finally:
        (
            s.native_functions,
            s.bindings,
            s.cuda_kernels,
            s.py_classes,
            s.test_files,
            s.cpp_extractor,
            s.test_functions,
            s.test_classes,
            s.opinfo_registry,
        ) = saved
        indexer._build_indexes(s)


class TestEmptyInputGuards:
    def test_trace_empty(self, loaded_state):
        assert "Provide a function name" in asyncio.run(trace(""))

    def test_search_empty(self, loaded_state):
        assert "Provide a query" in asyncio.run(_do_search_bindings("  "))

    def test_graph_entry_points_empty(self, loaded_state):
        for fn in (_do_calls, _do_called_by, _do_impact):
            assert "Provide a function name" in asyncio.run(fn(""))

    def test_modules_trace_empty(self, loaded_state):
        assert "Provide a module" in asyncio.run(_do_trace_module(" "))

    def test_tests_file_info_empty(self, loaded_state):
        assert "Provide a test file name" in asyncio.run(_do_test_file_info(""))


class TestLimitClamps:
    def test_search_negative_limit(self, loaded_state):
        out = asyncio.run(_do_search_bindings("add", limit=-1))
        assert "Showing -1" not in out
        assert "aten.add" in out

    def test_kernels_zero_limit(self, loaded_state):
        out = asyncio.run(_do_cuda_kernels("kernel", limit=0))
        assert "Showing 0" not in out
        assert "add_kernel" in out

    def test_find_tests_negative_limit(self, loaded_state):
        loaded_state.test_functions = {}
        loaded_state.test_classes = {}
        loaded_state.opinfo_registry = {}
        out = asyncio.run(_do_find_similar_tests("add", limit=-5))
        assert "and -" not in out
