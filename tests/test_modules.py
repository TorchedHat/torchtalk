"""Tests for the modules list and trace tools."""

import asyncio
from types import SimpleNamespace

import pytest

from torchtalk.tools.modules import _do_list_modules, _do_trace_module


def _cls(name, qualified):
    return SimpleNamespace(name=name, qualified_name=qualified)


def _method(name, signature="()"):
    return SimpleNamespace(name=name, signature=signature)


def _trace_cls(
    name,
    qualified,
    *,
    file_path="/src/torch/nn/modules/linear.py",
    line_number=98,
    bases=None,
    is_module=False,
    docstring=None,
    methods=None,
):
    return SimpleNamespace(
        name=name,
        qualified_name=qualified,
        file_path=file_path,
        line_number=line_number,
        bases=bases or [],
        is_module=is_module,
        docstring=docstring,
        methods=methods or [],
    )


@pytest.fixture
def module_state(mock_state):
    s = mock_state
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
    return s


@pytest.fixture
def trace_module_state(mock_state):
    s = mock_state
    s.bindings = [{"python_name": "x"}]
    s.native_functions = {}
    s.pytorch_source = "/src"
    linear = _trace_cls(
        "Linear",
        "torch.nn.Linear",
        file_path="/src/torch/nn/modules/linear.py",
        line_number=98,
        bases=["Module"],
        is_module=True,
        docstring="Applies a linear transformation to the incoming data.",
        methods=[
            _method("forward", "(self, input)"),
            _method("__init__", "(self, in_features, out_features)"),
        ],
    )
    s.py_classes = {
        "Linear": [linear],
        "LinearReLU": [
            _trace_cls(
                "LinearReLU",
                "torch.nn.intrinsic.LinearReLU",
                file_path="/src/torch/nn/intrinsic/modules/fused.py",
                line_number=12,
                is_module=True,
            )
        ],
        "AvgPool2d": [
            _trace_cls(
                "AvgPool2d",
                "torch.nn.AvgPool2d",
                file_path="/src/torch/nn/modules/pooling.py",
                line_number=40,
                is_module=True,
            )
        ],
        "MaxPool2d": [
            _trace_cls(
                "MaxPool2d",
                "torch.nn.MaxPool2d",
                file_path="/src/torch/nn/modules/pooling.py",
                line_number=80,
                is_module=True,
            )
        ],
        "SGD": [
            _trace_cls(
                "SGD",
                "torch.optim.SGD",
                file_path="/src/torch/optim/sgd.py",
                line_number=5,
                is_module=False,
            )
        ],
    }
    return s


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


class TestTraceModule:
    def test_exact_match_renders_file_and_name(self, trace_module_state):
        out = asyncio.run(_do_trace_module("Linear"))
        assert "torch.nn.Linear" in out
        assert "torch/nn/modules/linear.py:98" in out
        assert "LinearReLU" not in out

    def test_qualified_name_uses_last_segment(self, trace_module_state):
        out = asyncio.run(_do_trace_module("torch.nn.Linear"))
        assert "torch.nn.Linear" in out
        assert "torch/nn/modules/linear.py:98" in out

    def test_exact_match_beats_substring(self, trace_module_state):
        out = asyncio.run(_do_trace_module("Linear"))
        assert "torch.nn.Linear" in out
        assert "Showing top match" not in out
        assert "intrinsic.LinearReLU" not in out

    def test_substring_fallback(self, trace_module_state):
        out = asyncio.run(_do_trace_module("Pool"))
        assert "AvgPool2d" in out or "MaxPool2d" in out
        assert "Showing top match of 2 total" in out

    def test_focus_full_includes_bases_type_docstring(self, trace_module_state):
        out = asyncio.run(_do_trace_module("Linear", focus="full"))
        assert "**Bases:** Module" in out
        assert "**Type:** torch.nn.Module" in out
        assert "Applies a linear transformation" in out

    def test_default_focus_omits_full_metadata(self, trace_module_state):
        out = asyncio.run(_do_trace_module("Linear"))
        assert "**Methods:**" in out
        assert "`forward(self, input)`" in out
        assert "**Bases:**" not in out
        assert "**Type:**" not in out
        assert "Applies a linear transformation" not in out

    def test_method_truncation(self, trace_module_state):
        linear = trace_module_state.py_classes["Linear"][0]
        linear.methods = [_method(f"m{i}", "()") for i in range(12)]
        out = asyncio.run(_do_trace_module("Linear"))
        assert "`m0()`" in out
        assert "`m9()`" in out
        assert "`m10()`" not in out
        assert "... and 2 more" in out

    def test_multiple_exact_matches_footer(self, trace_module_state):
        extra = _trace_cls(
            "Linear",
            "torch.nn.modules.linear.Linear",
            file_path="/src/torch/nn/modules/linear.py",
            line_number=200,
        )
        trace_module_state.py_classes["Linear"].append(extra)
        out = asyncio.run(_do_trace_module("Linear"))
        assert "Showing top match of 2 total" in out

    def test_not_found_with_similar_suggestion(self, trace_module_state):
        out = asyncio.run(_do_trace_module("Linaer"))
        assert "Module `Linaer` not found. Similar:" in out
        assert "Linear" in out

    def test_not_found_without_similar(self, trace_module_state):
        out = asyncio.run(_do_trace_module("zzz_no_such_module_xyz"))
        assert out == "Module `zzz_no_such_module_xyz` not found."

    def test_python_analysis_unavailable(self, trace_module_state):
        trace_module_state.py_classes = {}
        out = asyncio.run(_do_trace_module("Linear"))
        assert out == (
            "Python module analysis not available. Ensure PyTorch source is loaded."
        )
