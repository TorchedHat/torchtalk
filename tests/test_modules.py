"""Tests for the modules list tool grouping."""

import asyncio
from types import SimpleNamespace

import pytest

from torchtalk import indexer
from torchtalk.tools.modules import _do_list_modules


def _cls(name, qualified):
    return SimpleNamespace(name=name, qualified_name=qualified)


@pytest.fixture
def module_state():
    s = indexer._state
    saved = (s.py_classes, s.nn_modules, s.bindings, s.native_functions)
    s.bindings = [{"python_name": "x"}]
    s.native_functions = {}
    s.nn_modules = [
        _cls("Linear", "torch.nn.Linear"),
        _cls("MSELoss", "torch.nn.MSELoss"),
        _cls("ModuleList", "torch.nn.ModuleList"),
        _cls("ModuleDict", "torch.nn.ModuleDict"),
    ]
    s.py_classes = {
        "SGD": [_cls("SGD", "torch.optim.SGD")],
        "ArgMappingException": [
            _cls("ArgMappingException", "torch.distributed.optim.ArgMappingException")
        ],
        **{
            f"Opt{i:02d}": [_cls(f"Opt{i:02d}", f"torch.optim.Opt{i:02d}")]
            for i in range(22)
        },
    }
    try:
        yield s
    finally:
        (s.py_classes, s.nn_modules, s.bindings, s.native_functions) = saved


class TestNnGrouping:
    def test_sections_partition_no_double_count(self, module_state):
        out = asyncio.run(_do_list_modules("nn"))
        assert "Found 4 nn.Module subclasses" in out
        assert "Layers (1)" in out
        assert "Loss Functions (1)" in out
        assert "Containers (2)" in out


class TestOptimListing:
    def test_truncation_footer_present(self, module_state):
        out = asyncio.run(_do_list_modules("optim"))
        assert "more.*" in out or "more." in out

    def test_non_optimizers_excluded(self, module_state):
        out = asyncio.run(_do_list_modules("optim"))
        assert "ArgMappingException" not in out
