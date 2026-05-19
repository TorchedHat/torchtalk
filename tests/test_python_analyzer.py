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

    def test_alias_map_resolves_torch_dot_op(self):
        alias_map = {"torch.add": "aten::add", "torch.relu": "aten::relu"}
        assert resolve_cpp_symbol("torch.add", alias_map) == "aten::add"
        assert resolve_cpp_symbol("torch.relu", alias_map) == "aten::relu"

    def test_alias_map_unknown_call_returns_none(self):
        alias_map = {"torch.add": "aten::add"}
        assert resolve_cpp_symbol("torch.unknown_op", alias_map) is None

    def test_alias_map_does_not_intercept_torch_ops_form(self):
        # `torch.ops.aten.add` must still resolve via the existing path even
        # if the alias map happens to contain a literal entry for it.
        alias_map = {"torch.ops.aten.add": "BOGUS"}
        assert resolve_cpp_symbol("torch.ops.aten.add", alias_map) == "aten::add"

    def test_no_alias_map_falls_through_for_torch_dot_op(self):
        # Without the alias map, `torch.add` is unresolvable (back-compat).
        assert resolve_cpp_symbol("torch.add") is None

    def test_import_alias_rewrites_leading_segment(self):
        alias_map = {"torch.linalg.cross": "aten::linalg_cross"}
        # `from torch import linalg` -> `linalg` -> `torch.linalg`.
        import_aliases = {"linalg": "torch.linalg"}
        assert (
            resolve_cpp_symbol("linalg.cross", alias_map, import_aliases)
            == "aten::linalg_cross"
        )

    def test_import_alias_for_module_renamed_via_as(self):
        alias_map = {"torch.nn.functional.linear": "aten::linear"}
        # `from torch.nn import functional as F` -> `F` -> `torch.nn.functional`.
        import_aliases = {"F": "torch.nn.functional"}
        assert (
            resolve_cpp_symbol("F.linear", alias_map, import_aliases) == "aten::linear"
        )

    def test_import_alias_for_directly_imported_function(self):
        alias_map = {"torch.nn.functional.linear": "aten::linear"}
        # `from torch.nn.functional import linear` -> `linear` -> full path.
        import_aliases = {"linear": "torch.nn.functional.linear"}
        assert resolve_cpp_symbol("linear", alias_map, import_aliases) == "aten::linear"

    def test_import_alias_unknown_local_returns_none(self):
        alias_map = {"torch.linalg.cross": "aten::linalg_cross"}
        import_aliases = {"linalg": "torch.linalg"}
        # `unknown_mod.foo` is not imported — falls through to None.
        assert resolve_cpp_symbol("unknown_mod.foo", alias_map, import_aliases) is None

    def test_import_alias_expansion_misses_alias_map(self):
        # Local imports that don't lead to a torch alias remain unresolvable.
        alias_map = {"torch.linalg.cross": "aten::linalg_cross"}
        import_aliases = {"path": "os.path"}  # unrelated alias
        assert resolve_cpp_symbol("path.join", alias_map, import_aliases) is None

    def test_import_alias_does_not_affect_torch_ops_form(self):
        # `torch.ops.aten.add` already resolves directly; import_aliases must
        # not interfere even if it has a misleading entry.
        alias_map = {"torch.add": "BOGUS"}
        import_aliases = {"torch": "BOGUS_PATH"}
        assert (
            resolve_cpp_symbol("torch.ops.aten.add", alias_map, import_aliases)
            == "aten::add"
        )


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


class TestAnalyzerAliasMap:
    def _analyze(self, tmp_path, body: str, alias_map=None):
        path = tmp_path / "torch" / "fake_mod.py"
        path.parent.mkdir(parents=True)
        path.write_text(body)
        analyzer = PythonAnalyzer(alias_map=alias_map)
        return analyzer.analyze_file(str(path))

    def test_torch_dot_op_resolves_with_alias_map(self, tmp_path):
        module = self._analyze(
            tmp_path,
            "import torch\ndef f(x):\n    return torch.add(x, 1)\n",
            alias_map={"torch.add": "aten::add"},
        )
        bindings = module.functions[0].cpp_bindings
        assert len(bindings) == 1
        assert bindings[0].cpp_symbol == "aten::add"

    def test_torch_dot_op_without_alias_map_yields_no_binding(self, tmp_path):
        # Back-compat: omitting alias_map preserves prior behavior — `torch.add`
        # is silently skipped rather than misresolved.
        module = self._analyze(
            tmp_path,
            "import torch\ndef f(x):\n    return torch.add(x, 1)\n",
        )
        assert module.functions[0].cpp_bindings == []

    def test_method_call_still_does_not_create_false_edge(self, tmp_path):
        # `t.add(1)` must NOT resolve through the alias map even when `torch.add`
        # is registered — receiver type is unknown so we stay conservative.
        module = self._analyze(
            tmp_path,
            "def f(t):\n    return t.add(1)\n",
            alias_map={"torch.add": "aten::add"},
        )
        assert module.functions[0].cpp_bindings == []


class TestImportAwareResolution:
    def _analyze(self, tmp_path, body: str, alias_map=None):
        path = tmp_path / "torch" / "fake_mod.py"
        path.parent.mkdir(parents=True)
        path.write_text(body)
        analyzer = PythonAnalyzer(alias_map=alias_map)
        return analyzer.analyze_file(str(path))

    def test_from_torch_import_namespace_module(self, tmp_path):
        # `from torch import linalg` then `linalg.cross(x, y)` should resolve
        # via the alias map's `torch.linalg.cross` entry.
        module = self._analyze(
            tmp_path,
            "from torch import linalg\ndef f(x, y):\n    return linalg.cross(x, y)\n",
            alias_map={"torch.linalg.cross": "aten::linalg_cross"},
        )
        bindings = module.functions[0].cpp_bindings
        assert len(bindings) == 1
        assert bindings[0].cpp_symbol == "aten::linalg_cross"

    def test_import_module_with_as_alias(self, tmp_path):
        # `import torch.linalg as L` then `L.cross(x, y)`.
        module = self._analyze(
            tmp_path,
            "import torch.linalg as L\ndef f(x, y):\n    return L.cross(x, y)\n",
            alias_map={"torch.linalg.cross": "aten::linalg_cross"},
        )
        assert module.functions[0].cpp_bindings[0].cpp_symbol == "aten::linalg_cross"

    def test_from_module_import_function_directly(self, tmp_path):
        # `from torch.nn.functional import linear` then `linear(x, w)`.
        module = self._analyze(
            tmp_path,
            "from torch.nn.functional import linear\n"
            "def f(x, w):\n"
            "    return linear(x, w)\n",
            alias_map={"torch.nn.functional.linear": "aten::linear"},
        )
        assert module.functions[0].cpp_bindings[0].cpp_symbol == "aten::linear"

    def test_from_module_import_with_as_alias(self, tmp_path):
        # `from torch.nn import functional as F` then `F.linear(...)`.
        module = self._analyze(
            tmp_path,
            "from torch.nn import functional as F\n"
            "def g(x, w):\n"
            "    return F.linear(x, w)\n",
            alias_map={"torch.nn.functional.linear": "aten::linear"},
        )
        assert module.functions[0].cpp_bindings[0].cpp_symbol == "aten::linear"

    def test_relative_import_skipped(self, tmp_path):
        # `from . import linalg` is a relative import — package location
        # unknown at scrape time, so we don't expand it.
        module = self._analyze(
            tmp_path,
            "from . import linalg\ndef f(x, y):\n    return linalg.cross(x, y)\n",
            alias_map={"torch.linalg.cross": "aten::linalg_cross"},
        )
        assert module.functions[0].cpp_bindings == []

    def test_wildcard_import_skipped(self, tmp_path):
        # `from torch.linalg import *` doesn't enumerate names — skip.
        module = self._analyze(
            tmp_path,
            "from torch.linalg import *\ndef f(x, y):\n    return cross(x, y)\n",
            alias_map={"torch.linalg.cross": "aten::linalg_cross"},
        )
        assert module.functions[0].cpp_bindings == []

    def test_function_defined_before_import_still_resolves(self, tmp_path):
        # Function appears textually BEFORE the import — pre-pass collects
        # imports first so resolution still works.
        module = self._analyze(
            tmp_path,
            "def f(x, y):\n    return linalg.cross(x, y)\nfrom torch import linalg\n",
            alias_map={"torch.linalg.cross": "aten::linalg_cross"},
        )
        assert module.functions[0].cpp_bindings[0].cpp_symbol == "aten::linalg_cross"

    def test_unrelated_import_does_not_false_match(self, tmp_path):
        # `from os import path` — `path.join(...)` must NOT register as a
        # torch binding even though import_aliases has a `path` entry.
        module = self._analyze(
            tmp_path,
            "from os import path\ndef f(a, b):\n    return path.join(a, b)\n",
            alias_map={"torch.linalg.cross": "aten::linalg_cross"},
        )
        assert module.functions[0].cpp_bindings == []
