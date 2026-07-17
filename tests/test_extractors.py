"""Tests for config-driven registration extractors and qualname resolver."""

from __future__ import annotations

import dataclasses

import pytest

from tests.conftest import get_pytorch_path
from torchtalk.analysis.extractors import (
    extract_registrations,
    resolve_qualname_literals,
)
from torchtalk.harness import PYTORCH_MANIFEST, CallRegistration, ConventionManifest


def _manifest(**kwargs) -> ConventionManifest:
    return ConventionManifest(package="fake", cpp_search_dirs=(), **kwargs)


def _write(tmp_path, rel: str, body: str):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


class TestDecoratorRegistries:
    def test_string_key_decorator_on_class(self, tmp_path):
        _write(
            tmp_path,
            "pkg/ops.py",
            "@CustomOp.register('silu_and_mul')\nclass SiluAndMul:\n    pass\n",
        )
        m = _manifest(decorator_registries={"CustomOp.register": "custom_ops"})
        out = extract_registrations(tmp_path, m)
        assert out["records"] == [
            {
                "registry": "custom_ops",
                "key": "silu_and_mul",
                "target": "pkg.ops.SiluAndMul",
                "kind": "resolved",
                "via": "decorator",
                "file": "pkg/ops.py",
                "line": 2,
            }
        ]

    def test_dotted_and_list_keys(self, tmp_path):
        _write(
            tmp_path,
            "pkg/decomp.py",
            "@register_decomposition(aten.gelu)\ndef gelu(x):\n    pass\n\n"
            "@register_decomposition([aten.abs, aten.neg])\ndef absneg(x):\n    pass\n",
        )
        m = _manifest(decorator_registries={"register_decomposition": "decomp"})
        keys = {r["key"] for r in extract_registrations(tmp_path, m)["records"]}
        assert keys == {"aten.gelu", "aten.abs", "aten.neg"}

    def test_bare_decorator_uses_decorated_name_as_key(self, tmp_path):
        _write(tmp_path, "pkg/b.py", "@register_backend\ndef inductor(g):\n    pass\n")
        m = _manifest(decorator_registries={"register_backend": "backends"})
        (rec,) = extract_registrations(tmp_path, m)["records"]
        assert rec["key"] == "inductor"
        assert rec["target"] == "pkg.b.inductor"


class TestStringRegistries:
    def test_vllm_tuple_shape_is_candidate(self, tmp_path):
        _write(
            tmp_path,
            "pkg/registry.py",
            "_VLLM_MODELS = {\n"
            "    'LlamaForCausalLM': ('llama', 'LlamaForCausalLM'),\n"
            "}\n",
        )
        m = _manifest(string_registries=("_VLLM_MODELS",))
        (rec,) = extract_registrations(tmp_path, m)["records"]
        assert rec["registry"] == "_VLLM_MODELS"
        assert rec["key"] == "LlamaForCausalLM"
        assert rec["target"] == "llama.LlamaForCausalLM"
        assert rec["kind"] == "candidate"
        assert rec["evidence"] == "llama.LlamaForCausalLM"

    def test_hf_import_structure_list_shape(self, tmp_path):
        _write(
            tmp_path,
            "pkg/init_stub.py",
            "_import_structure = {'models.llama': ['LlamaModel', 'LlamaConfig']}\n",
        )
        m = _manifest(string_registries=("_import_structure",))
        targets = {r["target"] for r in extract_registrations(tmp_path, m)["records"]}
        assert targets == {"models.llama.LlamaModel", "models.llama.LlamaConfig"}

    def test_qualname_string_value(self, tmp_path):
        _write(tmp_path, "pkg/r.py", "_REG = {'k': 'pkg.mod.Cls'}\n")
        m = _manifest(string_registries=("_REG",))
        (rec,) = extract_registrations(tmp_path, m)["records"]
        assert rec["target"] == "pkg.mod.Cls"

    def test_undeclared_registry_ignored(self, tmp_path):
        _write(tmp_path, "pkg/r.py", "_OTHER = {'k': 'pkg.mod.Cls'}\n")
        m = _manifest(string_registries=("_REG",))
        assert extract_registrations(tmp_path, m)["records"] == []


class TestRegistrationCalls:
    def test_kwarg_roles(self, tmp_path):
        _write(
            tmp_path,
            "pkg/c.py",
            "direct_register_custom_op(op_name='rms_norm', op_func=rms_norm_impl)\n",
        )
        m = _manifest(
            registration_calls=(
                CallRegistration(
                    "direct_register_custom_op", "op_name", "op_func", "custom_ops"
                ),
            )
        )
        (rec,) = extract_registrations(tmp_path, m)["records"]
        assert rec == {
            "registry": "custom_ops",
            "key": "rms_norm",
            "target": "rms_norm_impl",
            "kind": "resolved",
            "via": "call",
            "file": "pkg/c.py",
            "line": 1,
        }

    def test_positional_roles_and_default_registry(self, tmp_path):
        _write(tmp_path, "pkg/c.py", "register_op('fused_moe', fused_moe_kernel)\n")
        m = _manifest(registration_calls=(CallRegistration("register_op", 0, 1),))
        (rec,) = extract_registrations(tmp_path, m)["records"]
        assert rec["registry"] == "register_op"
        assert rec["key"] == "fused_moe"
        assert rec["target"] == "fused_moe_kernel"


class TestStringDispatchers:
    def test_dispatch_emits_candidate_edge_with_evidence(self, tmp_path):
        _write(
            tmp_path,
            "pkg/engine.py",
            "class Engine:\n"
            "    def start(self):\n"
            "        self.collective_rpc('init_worker')\n",
        )
        m = _manifest(string_dispatchers={"collective_rpc": 0})
        (edge,) = extract_registrations(tmp_path, m)["candidate_edges"]
        assert edge == {
            "kind": "candidate",
            "via": "string_dispatch",
            "source": "pkg.engine.Engine.start",
            "target": "init_worker",
            "evidence": "init_worker",
            "file": "pkg/engine.py",
            "line": 3,
        }

    def test_non_literal_arg_emits_nothing(self, tmp_path):
        _write(tmp_path, "pkg/e.py", "rpc.collective_rpc(method_name)\n")
        m = _manifest(string_dispatchers={"collective_rpc": 0})
        assert extract_registrations(tmp_path, m)["candidate_edges"] == []


class TestQualnameResolver:
    def test_literal_matching_indexed_qualname_becomes_candidate(self, tmp_path):
        _write(tmp_path, "pkg/models.py", "class Foo:\n    pass\n")
        _write(
            tmp_path,
            "pkg/engine.py",
            "def launch():\n    load('pkg.models.Foo')\n",
        )
        m = _manifest(string_dispatchers={"never_called": 0})
        edges = extract_registrations(tmp_path, m)["candidate_edges"]
        assert edges == [
            {
                "kind": "candidate",
                "via": "qualname_literal",
                "source": "pkg.engine.launch",
                "target": "pkg.models.Foo",
                "evidence": "pkg.models.Foo",
                "file": "pkg/engine.py",
                "line": 2,
            }
        ]

    def test_unmatched_literals_emit_nothing(self):
        lits = [{"value": "no.such.Thing", "scope": "m.f", "file": "f", "line": 1}]
        assert resolve_qualname_literals(lits, {"pkg.models.Foo"}) == []

    def test_plain_words_never_collected(self, tmp_path):
        _write(tmp_path, "pkg/w.py", "def f():\n    x = 'launch'\n")
        m = _manifest(string_dispatchers={"never_called": 0})
        assert extract_registrations(tmp_path, m)["candidate_edges"] == []


class TestManifestGating:
    def test_unconfigured_manifest_skips_walk(self, tmp_path):
        _write(tmp_path, "pkg/r.py", "_REG = {'k': 'pkg.mod.Cls'}\n")
        out = extract_registrations(tmp_path, _manifest())
        assert out == {"records": [], "candidate_edges": []}

    def test_exclude_patterns_respected(self, tmp_path):
        _write(
            tmp_path,
            "pkg/test_x.py",
            "@CustomOp.register('k')\nclass C:\n    pass\n",
        )
        m = _manifest(
            decorator_registries={"CustomOp.register": "custom_ops"},
            exclude_patterns=("test_",),
        )
        assert extract_registrations(tmp_path, m)["records"] == []


@pytest.mark.skipif(get_pytorch_path() is None, reason="PyTorch source not available")
class TestAgainstPyTorch:
    def test_register_decomposition_records_found(self):
        src = get_pytorch_path()
        m = dataclasses.replace(PYTORCH_MANIFEST, python_search_dirs=("torch/_decomp",))
        out = extract_registrations(src, m)
        decomp = [r for r in out["records"] if r["registry"] == "decompositions"]
        assert len(decomp) > 100
        assert all(r["kind"] == "resolved" and r["via"] == "decorator" for r in decomp)
        assert any(r["key"].startswith("aten.") for r in decomp)
