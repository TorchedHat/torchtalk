"""Tests for the test-discovery tool implementations."""

import asyncio

import pytest

from torchtalk import indexer
from torchtalk.tools.tests import (
    _do_find_similar_tests,
    _do_list_test_utils,
    _do_test_file_info,
)


@pytest.fixture
def test_tool_state(mock_state):
    s = mock_state
    s.bindings = [{"python_name": "x"}]
    s.native_functions = {}
    s.test_functions = {
        "test_softmax": [
            {
                "name": "test_softmax",
                "class": "TestOps",
                "file": "test/test_ops.py",
                "line": 10,
            },
            {
                "name": "test_softmax",
                "class": "TestCUDA",
                "file": "test/test_cuda.py",
                "line": 20,
            },
        ],
        "test_softmax_backward": [
            {
                "name": "test_softmax_backward",
                "class": "TestOps",
                "file": "test/test_ops.py",
                "line": 30,
            },
        ],
        "test_padding": [
            {
                "name": "test_padding",
                "class": "TestNN",
                "file": "test/test_nn.py",
                "line": 40,
            },
        ],
        "test_add": [
            {
                "name": "test_add",
                "class": "TestMath",
                "file": "test/test_math.py",
                "line": 50,
            },
        ],
    }
    s.test_classes = {
        "TestNN_softmax": [
            {
                "name": "TestNN_softmax",
                "file": "test/test_ops.py",
                "line": 5,
                "bases": ["TestCase"],
                "is_test_class": True,
            },
        ],
        "TestPadding": [
            {
                "name": "TestPadding",
                "file": "test/test_nn.py",
                "line": 100,
                "bases": ["TestCase"],
                "is_test_class": True,
            },
        ],
    }
    s.test_files = {
        "test/test_ops.py": {
            "path": "test/test_ops.py",
            "classes": [{"name": "TestOps", "line": 1, "bases": ["TestCase"]}],
            "functions": [
                {"name": "test_softmax", "class": "TestOps", "line": 10},
                {"name": "test_softmax_backward", "class": "TestOps", "line": 30},
            ],
        },
        "test/test_nn.py": {
            "path": "test/test_nn.py",
            "classes": [{"name": "TestNN", "line": 1, "bases": ["NNTestCase"]}],
            "functions": [{"name": "test_padding", "class": "TestNN", "line": 40}],
        },
        "test/test_cuda.py": {
            "path": "test/test_cuda.py",
            "classes": [],
            "functions": [{"name": "test_softmax", "class": "TestCUDA", "line": 20}],
        },
        "test/test_math.py": {
            "path": "test/test_math.py",
            "classes": [],
            "functions": [{"name": "test_add", "class": "TestMath", "line": 50}],
        },
    }
    s.opinfo_registry = {
        "softmax": {
            "name": "softmax",
            "file": "torch/testing/_internal/opinfo/definitions/nn.py",
            "line": 42,
            "aliases": [],
            "aten_name": "_softmax",
        },
    }
    s.test_utilities = {
        "torch/testing/_internal/common_utils.py": {
            "path": "torch/testing/_internal/common_utils.py",
            "full_path": "/fake/torch/testing/_internal/common_utils.py",
            "exists": True,
        },
    }
    s.source = None
    indexer._build_indexes(s)
    return s


class TestFindSimilarTests:
    def test_matches_function_by_word_boundary(self, test_tool_state):
        out = asyncio.run(_do_find_similar_tests("softmax"))
        assert "test_softmax" in out
        assert "Test Functions" in out

    def test_rejects_substring_non_boundary(self, test_tool_state):
        out = asyncio.run(_do_find_similar_tests("add"))
        assert "test_add" in out
        assert "test_padding" not in out

    def test_matches_class_by_name(self, test_tool_state):
        out = asyncio.run(_do_find_similar_tests("softmax"))
        assert "TestNN_softmax" in out
        assert "Test Classes" in out

    def test_matches_file_by_substring(self, test_tool_state):
        out = asyncio.run(_do_find_similar_tests("test_ops"))
        assert "test/test_ops.py" in out

    def test_matches_opinfo_by_name(self, test_tool_state):
        out = asyncio.run(_do_find_similar_tests("softmax"))
        assert "OpInfo Definitions" in out
        assert "softmax" in out

    def test_focus_functions_only(self, test_tool_state):
        out = asyncio.run(_do_find_similar_tests("softmax", focus="functions"))
        assert "Test Functions" in out
        assert "Test Classes" not in out
        assert "Test Files" not in out

    def test_focus_classes_only(self, test_tool_state):
        out = asyncio.run(_do_find_similar_tests("softmax", focus="classes"))
        assert "Test Classes" in out
        assert "Test Functions" not in out

    def test_focus_files_only(self, test_tool_state):
        out = asyncio.run(_do_find_similar_tests("test_ops", focus="files"))
        assert "Test Files" in out
        assert "Test Functions" not in out
        assert "Test Classes" not in out

    def test_limit_truncation(self, test_tool_state):
        out = asyncio.run(_do_find_similar_tests("softmax", limit=1))
        assert "... and" in out

    def test_zero_results(self, test_tool_state):
        out = asyncio.run(_do_find_similar_tests("nonexistent_xyz_op"))
        assert "No tests found" in out


class TestListTestUtils:
    def test_renders_utility_names(self, test_tool_state):
        out = asyncio.run(_do_list_test_utils())
        assert "common_utils" in out
        assert "opinfo" in out

    def test_stats_displayed(self, test_tool_state):
        out = asyncio.run(_do_list_test_utils())
        assert "Test files indexed:" in out


class TestTestFileInfo:
    def test_shows_classes_and_functions(self, test_tool_state):
        out = asyncio.run(_do_test_file_info("test_ops"))
        assert "TestOps" in out
        assert "test_softmax" in out

    def test_multiple_matches_truncated(self, test_tool_state):
        s = test_tool_state
        for i in range(4):
            s.test_files[f"test/test_extra_{i}.py"] = {
                "path": f"test/test_extra_{i}.py",
                "classes": [],
                "functions": [],
            }
        out = asyncio.run(_do_test_file_info("test_extra"))
        assert "Showing 3 of" in out

    def test_no_match_message(self, test_tool_state):
        out = asyncio.run(_do_test_file_info("nonexistent_file"))
        assert "No test file found" in out
