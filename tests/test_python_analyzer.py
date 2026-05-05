"""Tests for python_analyzer C++ binding resolution."""

from __future__ import annotations

from torchtalk.analysis.python_analyzer import (
    PythonAnalyzer,
    resolve_cpp_symbol,
)


class TestResolveCppSymbol:
    def test_torch_ops_aten_two_part(self):
        assert resolve_cpp_symbol("torch.ops.aten.add") == "aten::add"

    def test_torch_ops_aten_with_overload_tag_dropped(self):
        assert resolve_cpp_symbol("torch.ops.aten.add.Tensor") == "aten::add"

    def test_torch_ops_other_namespace(self):
        assert resolve_cpp_symbol("torch.ops.profiler.foo") == "profiler::foo"

    def test_torch_underscore_C_returns_inner(self):
        assert resolve_cpp_symbol("torch._C._tensor_op") == "_tensor_op"

    def test_bare_C_alias(self):
        assert resolve_cpp_symbol("_C.foo") == "foo"

    def test_method_call_returns_none(self):
        # `t.add(1)` parses as Attribute(Name('t'), 'add') — no torch prefix.
        assert resolve_cpp_symbol("t.add") is None

    def test_unrelated_call_returns_none(self):
        assert resolve_cpp_symbol("random.seed") is None

    def test_empty_returns_none(self):
        assert resolve_cpp_symbol("") is None

    def test_torch_ops_one_part_returns_none(self):
        # `torch.ops.aten` (no op name) is malformed — drop.
        assert resolve_cpp_symbol("torch.ops.aten") is None


class TestFindCppBindings:
    def _analyze(self, tmp_path, body: str):
        path = tmp_path / "torch" / "fake_mod.py"
        path.parent.mkdir(parents=True)
        path.write_text(body)
        analyzer = PythonAnalyzer()
        return analyzer.analyze_file(str(path))

    def test_function_with_aten_call_records_binding(self, tmp_path):
        module = self._analyze(
            tmp_path,
            "import torch\ndef f(x):\n    return torch.ops.aten.add(x, 1)\n",
        )
        assert module is not None
        bindings = module.functions[0].cpp_bindings
        assert len(bindings) == 1
        assert bindings[0].cpp_symbol == "aten::add"

    def test_method_call_does_not_create_false_edge(self, tmp_path):
        module = self._analyze(
            tmp_path,
            "def f(t):\n    return t.add(1)\n",
        )
        assert module is not None
        assert module.functions[0].cpp_bindings == []

    def test_multiple_calls_same_symbol_dedup_per_line(self, tmp_path):
        module = self._analyze(
            tmp_path,
            "import torch\n"
            "def f(x):\n"
            "    a = torch.ops.aten.add(x, 1)\n"
            "    b = torch.ops.aten.add(x, 2)\n"
            "    return a + b\n",
        )
        symbols = [b.cpp_symbol for b in module.functions[0].cpp_bindings]
        # Two distinct call sites — both kept.
        assert symbols == ["aten::add", "aten::add"]

    def test_torch_C_tensor_op_recorded(self, tmp_path):
        module = self._analyze(
            tmp_path,
            "import torch\ndef f(x):\n    return torch._C._tensor_op(x)\n",
        )
        bindings = module.functions[0].cpp_bindings
        assert len(bindings) == 1
        assert bindings[0].cpp_symbol == "_tensor_op"

    def test_method_function_captures_bindings(self, tmp_path):
        module = self._analyze(
            tmp_path,
            "import torch\n"
            "class M:\n"
            "    def forward(self, x):\n"
            "        return torch.ops.aten.relu(x)\n",
        )
        cls = module.classes[0]
        assert cls.methods[0].cpp_bindings[0].cpp_symbol == "aten::relu"
