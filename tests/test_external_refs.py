"""Tests for cross-package import refs (analysis/external_refs.py)."""

from __future__ import annotations

import dataclasses

from torchtalk.analysis.external_refs import (
    REF_KINDS,
    ExternalRef,
    bridge_targets,
    collect_cpp_refs,
    collect_import_refs,
    collect_op_refs,
    op_namespace_targets,
    refs_by_target,
)
from torchtalk.analysis.python_analyzer import PyImport, PyModule
from torchtalk.harness import ConventionManifest, load_builtin_manifest


def _mod(name: str, imports: list[tuple[str, str]], path: str = "") -> PyModule:
    return PyModule(
        name=name,
        file_path=path or f"{name.replace('.', '/')}.py",
        classes=[],
        functions=[],
        imports=[
            PyImport(
                module=m, name=n, file_path=path or f"{name}.py", line_number=i + 1
            )
            for i, (m, n) in enumerate(imports)
        ],
        exports=[],
    )


def _manifest(**overrides) -> ConventionManifest:
    base = dataclasses.replace(
        load_builtin_manifest("vllm"),
        package="vllm",
        python_package_roots=("vllm",),
        depends_on=("pytorch",),
    )
    return dataclasses.replace(base, **overrides)


TARGETS = {"pytorch": ("torch",)}


class TestCollectImportRefs:
    def test_import_and_from_import_become_refs(self):
        mods = {
            "vllm.model_executor.layers.linear": _mod(
                "vllm.model_executor.layers.linear",
                [
                    ("torch.nn", "torch.nn"),
                    ("torch", "nn"),
                    ("torch.nn.functional", "torch.nn.functional"),
                ],
            )
        }
        refs = collect_import_refs(mods, _manifest(), TARGETS)
        names = [r.to_name for r in refs]
        assert names == ["torch.nn", "torch.nn", "torch.nn.functional"]
        assert all(r.kind == "import" for r in refs)
        assert all(r.to_package == "pytorch" for r in refs)
        assert all(r.from_symbol == "vllm.model_executor.layers.linear" for r in refs)
        assert refs[0].evidence.endswith(":1")

    def test_skips_relative_own_package_and_third_party(self):
        mods = {
            "vllm.utils": _mod(
                "vllm.utils",
                [
                    ("", "helpers"),
                    ("vllm.config", "Config"),
                    ("numpy", "numpy"),
                    ("torch", "torch"),
                ],
            )
        }
        refs = collect_import_refs(mods, _manifest(), TARGETS)
        assert [r.to_name for r in refs] == ["torch"]

    def test_star_import_refers_to_module(self):
        mods = {"m": _mod("m", [("torch.nn", "*")])}
        refs = collect_import_refs(mods, _manifest(), TARGETS)
        assert [r.to_name for r in refs] == ["torch.nn"]

    def test_no_depends_on_yields_nothing(self):
        mods = {"m": _mod("m", [("torch", "torch")])}
        assert collect_import_refs(mods, _manifest(depends_on=()), {}) == []

    def test_dedupes_same_target_same_line(self):
        mod = _mod("m", [("torch", "torch")])
        mod.imports.append(
            PyImport(module="torch", name="torch", file_path="m.py", line_number=1)
        )
        refs = collect_import_refs({"m": mod}, _manifest(), TARGETS)
        assert len(refs) == 1

    def test_deterministic_order(self):
        mods = {
            "b": _mod("b", [("torch", "torch")]),
            "a": _mod("a", [("torch.nn", "torch.nn"), ("torch", "torch")]),
        }
        refs = collect_import_refs(mods, _manifest(), TARGETS)
        assert [(r.from_symbol, r.to_name) for r in refs] == [
            ("a", "torch.nn"),
            ("a", "torch"),
            ("b", "torch"),
        ]

    def test_to_dict_roundtrip(self):
        ref = ExternalRef("a", "torch", "import", "a.py:1", to_package="pytorch")
        d = ref.to_dict()
        assert d == {
            "from_symbol": "a",
            "to_name": "torch",
            "kind": "import",
            "evidence": "a.py:1",
            "confidence": 1.0,
            "to_package": "pytorch",
        }
        assert "import" in REF_KINDS and "provides" in REF_KINDS

    def test_refs_by_target_groups(self):
        refs = [
            ExternalRef("a", "torch", "import", "a.py:1"),
            ExternalRef("b", "torch", "import", "b.py:1"),
            ExternalRef("b", "torch.nn", "import", "b.py:2"),
        ]
        grouped = refs_by_target(refs)
        assert sorted(grouped) == ["torch", "torch.nn"]
        assert len(grouped["torch"]) == 2


class TestBridgeTargets:
    def test_resolves_depends_on_via_builtin_manifests(self):
        vllm = load_builtin_manifest("vllm")
        assert bridge_targets(vllm) == {"pytorch": ("torch",)}

    def test_unknown_dependency_is_skipped(self):
        assert bridge_targets(_manifest(depends_on=("does-not-exist",))) == {}

    def test_end_to_end_with_builtin_vllm_manifest(self):
        vllm = load_builtin_manifest("vllm")
        mods = {"vllm.x": _mod("vllm.x", [("torch", "torch"), ("vllm.y", "z")])}
        refs = collect_import_refs(mods, vllm)
        assert [(r.to_name, r.to_package) for r in refs] == [("torch", "pytorch")]


class TestCollectOpRefs:
    def test_dependency_namespace_becomes_op_ref(self):
        edges = {
            "aten::silu": [
                {
                    "caller_qualname": "vllm.act.SiluAndMul.forward",
                    "file": "/s/vllm/act.py",
                    "line": 7,
                },
                {
                    "caller_qualname": "vllm.act.SiluAndMul.forward",
                    "file": "/s/vllm/act.py",
                    "line": 7,
                },
            ],
            "_C::rms_norm": [
                {"caller_qualname": "vllm.ln.f", "file": "/s/vllm/ln.py", "line": 3}
            ],
            "_tensor_op": [
                {"caller_qualname": "vllm.ln.g", "file": "/s/vllm/ln.py", "line": 4}
            ],
        }
        refs = collect_op_refs(edges, _manifest(), {"aten": "pytorch"}, source="/s")
        assert [
            (r.from_symbol, r.to_name, r.evidence, r.kind, r.to_package) for r in refs
        ] == [
            (
                "vllm.act.SiluAndMul.forward",
                "aten::silu",
                "vllm/act.py:7",
                "op",
                "pytorch",
            )
        ]

    def test_namespaces_come_from_dependency_manifests(self):
        assert op_namespace_targets(load_builtin_manifest("vllm")) == {
            "aten": "pytorch"
        }
        assert op_namespace_targets(_manifest(depends_on=())) == {}

    def test_no_targets_yields_nothing(self):
        edges = {"aten::silu": [{"caller_qualname": "f", "file": "f.py", "line": 1}]}
        assert collect_op_refs(edges, _manifest(depends_on=())) == []


CALLEES = {
    "rms_norm": {"torch::stable::contiguous", "at::empty", "helper", "TORCH_CHECK"},
    "helper": {"c10::cuda::getCurrentCUDAStream"},
    "torch::stable::Tensor::stride": {"at::Tensor::stride"},
}
LOCATIONS = {
    "rms_norm": ("/s/csrc/layernorm.cu", 40),
    "helper": ("/s/csrc/utils.h", 9),
    "torch::stable::Tensor::stride": ("/site/torch/include/tensor_struct.h", 80),
}


class TestCollectCppRefs:
    def test_callees_in_bridge_namespaces_become_refs(self):
        refs = collect_cpp_refs(CALLEES, LOCATIONS, _manifest(), source="/s")
        assert [(r.from_symbol, r.to_name, r.evidence) for r in refs] == [
            ("helper", "c10::cuda::getCurrentCUDAStream", "csrc/utils.h:9"),
            ("rms_norm", "at::empty", "csrc/layernorm.cu:40"),
            ("rms_norm", "torch::stable::contiguous", "csrc/layernorm.cu:40"),
        ]
        assert {r.kind for r in refs} == {"cpp"}
        assert {r.to_package for r in refs} == {"pytorch"}

    def test_callers_outside_source_are_skipped(self):
        refs = collect_cpp_refs(CALLEES, LOCATIONS, _manifest(), source="/s")
        assert "torch::stable::Tensor::stride" not in {r.from_symbol for r in refs}

    def test_needs_namespaces_and_dependency(self):
        assert collect_cpp_refs(CALLEES, LOCATIONS, _manifest(cpp_namespaces=())) == []
        assert collect_cpp_refs(CALLEES, LOCATIONS, _manifest(depends_on=())) == []
