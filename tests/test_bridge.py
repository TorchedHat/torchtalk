"""Tests for the dependency index and the bridge tool."""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from torchtalk import indexer
from torchtalk.analysis.python_analyzer import PyClass, PyFunction
from torchtalk.harness import load_builtin_manifest, set_active_harness
from torchtalk.indexer import ServerState, dependency_index, dependency_status
from torchtalk.server import get_status
from torchtalk.symbols import PackageIdentity
from torchtalk.tools.bridge import (
    _do_bridge,
    _installed_path,
    refs_from,
    refs_to,
    resolve,
)
from torchtalk.tools.ops import trace

REFS = [
    {
        "from_symbol": "vllm.act.SiluAndMul.forward_native",
        "to_name": "aten::silu",
        "kind": "op",
        "evidence": "vllm/act.py:12",
        "confidence": 1.0,
        "to_package": "pytorch",
    },
    {
        "from_symbol": "rms_norm",
        "to_name": "at::empty",
        "kind": "cpp",
        "evidence": "csrc/layernorm.cu:40",
        "confidence": 1.0,
        "to_package": "pytorch",
    },
    {
        "from_symbol": "rms_norm",
        "to_name": "torch::stable::contiguous",
        "kind": "cpp",
        "evidence": "csrc/layernorm.cu:40",
        "confidence": 1.0,
        "to_package": "pytorch",
    },
    {
        "from_symbol": "vllm.act",
        "to_name": "torch.nn.functional",
        "kind": "import",
        "evidence": "vllm/act.py:1",
        "confidence": 1.0,
        "to_package": "pytorch",
    },
    {
        "from_symbol": "vllm.cpu.act.SiluAndMul.forward_native",
        "to_name": "aten::rsqrt",
        "kind": "op",
        "evidence": "vllm/cpu/act.py:8",
        "confidence": 1.0,
        "to_package": "pytorch",
    },
]


def _pytorch_index() -> ServerState:
    dep = ServerState()
    dep.source = "/src/pytorch"
    dep.native_functions = {
        "silu": {"base_name": "silu", "dispatch": {"CPU": "silu_cpu"}},
        "empty": {"base_name": "empty", "dispatch": {}},
        "rsqrt": {
            "base_name": "rsqrt",
            "dispatch": {},
            "structured_delegate": "rsqrt.out",
        },
        "rsqrt.out": {
            "base_name": "rsqrt",
            "dispatch": {"CPU": "rsqrt_out", "CUDA": "rsqrt_out"},
        },
    }
    dep.native_implementations = {
        "silu_cpu": [
            {
                "function_name": "silu_cpu",
                "file_path": "/src/pytorch/aten/Act.cpp",
                "line_number": 9,
            }
        ],
        "empty": [
            {
                "function_name": "empty",
                "file_path": "/src/pytorch/aten/Empty.cpp",
                "line_number": 3,
            }
        ],
    }
    dep.bindings = [
        {
            "python_name": "aten.silu",
            "cpp_name": "silu_cpu",
            "file_path": "/src/pytorch/aten/Reg.cpp",
            "line_number": 5,
        }
    ]
    dep.cpp_locations = {
        "torch::stable::contiguous": ("/src/pytorch/torch/csrc/stable/ops.h", 90)
    }
    indexer._build_indexes(dep)
    return dep


@pytest.fixture
def vllm_state(mock_state, monkeypatch):
    set_active_harness("vllm")
    s = mock_state
    s.source = "/src/vllm"
    s.bindings = [
        {
            "python_name": "_C.rms_norm",
            "cpp_name": "rms_norm",
            "dispatch_key": "CUDA",
            "file_path": "/src/vllm/csrc/torch_bindings.cpp",
            "line_number": 20,
        }
    ]
    s.native_functions = {}
    s.native_implementations = {
        "rms_norm": [
            {
                "function_name": "rms_norm",
                "file_path": "/src/vllm/csrc/layernorm.cu",
                "line_number": 40,
            }
        ]
    }
    s.registrations = {}
    s.cpp_extractor = None
    s.cpp_building = False
    s.cuda_kernels = []
    s.derivatives = {}
    s.py_modules = {}
    s.nn_modules = []
    s.opinfo_registry = {}
    s.test_files = {}
    s.test_classes = {}
    s.test_functions = {}
    s.external_refs = list(REFS)
    s.alias_map = {"torch.silu": "aten::silu"}
    # Methods sit under their class, as the real index keeps them.
    cuda = PyFunction(
        "forward_native",
        "vllm.act.SiluAndMul.forward_native",
        "/src/vllm/vllm/act.py",
        10,
    )
    cpu = PyFunction(
        "forward_native",
        "vllm.cpu.act.SiluAndMul.forward_native",
        "/src/vllm/vllm/cpu/act.py",
        6,
    )
    s.py_functions = {}
    s.py_classes = {
        "SiluAndMul": [
            PyClass(
                "SiluAndMul",
                "vllm.act.SiluAndMul",
                "/src/vllm/vllm/act.py",
                8,
                methods=[cuda],
            ),
            PyClass(
                "SiluAndMul",
                "vllm.cpu.act.SiluAndMul",
                "/src/vllm/vllm/cpu/act.py",
                4,
                methods=[cpu],
            ),
        ]
    }
    indexer._build_indexes(s)
    monkeypatch.setattr(indexer, "_dependencies", {"pytorch": _pytorch_index()})
    try:
        yield s
    finally:
        set_active_harness("pytorch")


class TestLookups:
    def test_refs_from_matches_qualified_or_bare_cpp(self, vllm_state):
        assert [r["to_name"] for r in refs_from(["rms_norm"])] == [
            "at::empty",
            "torch::stable::contiguous",
        ]
        assert [
            r["to_name"] for r in refs_from(["vllm.act.SiluAndMul.forward_native"])
        ] == ["aten::silu"]
        assert refs_from(["forward_native"]) == []

    def test_refs_to_accepts_every_spelling(self, vllm_state):
        for name in ("aten::silu", "aten.silu", "torch.silu", "silu"):
            assert [r["from_symbol"] for r in refs_to(name)] == [
                "vllm.act.SiluAndMul.forward_native"
            ], name
        assert [r["to_name"] for r in refs_to("empty")] == ["at::empty"]
        assert [r["to_name"] for r in refs_to("torch.nn")] == ["torch.nn.functional"]
        assert [r["to_name"] for r in refs_to("torch::stable")] == [
            "torch::stable::contiguous"
        ]


class TestResolve:
    def test_op_resolves_through_native_functions(self, vllm_state):
        assert resolve(REFS[0]) == [
            {"symbol": "silu_cpu", "file": "/src/pytorch/aten/Act.cpp", "line": 9}
        ]

    def test_cpp_api_namespace_resolves_as_op(self, vllm_state):
        assert resolve(REFS[1]) == [
            {"symbol": "empty", "file": "/src/pytorch/aten/Empty.cpp", "line": 3}
        ]

    def test_cpp_symbol_resolves_through_call_graph_locations(self, vllm_state):
        assert resolve(REFS[2]) == [
            {
                "symbol": "torch::stable::contiguous",
                "file": "/src/pytorch/torch/csrc/stable/ops.h",
                "line": 90,
            }
        ]

    def test_op_falls_back_to_registrations(self, vllm_state):
        dep = indexer._dependencies["pytorch"]
        dep.native_implementations = {}
        assert resolve(REFS[0]) == [
            {"symbol": "silu_cpu", "file": "/src/pytorch/aten/Reg.cpp", "line": 5}
        ]

    def test_kernel_without_body_points_at_yaml(self, vllm_state):
        assert resolve(REFS[4]) == [
            {
                "symbol": "rsqrt_out",
                "file": "/src/pytorch/aten/src/ATen/native/native_functions.yaml",
                "line": None,
            }
        ]

    def test_imports_and_missing_index_stay_unresolved(self, vllm_state):
        assert resolve(REFS[3]) == []
        indexer._dependencies.clear()
        assert resolve(REFS[0]) == []


class TestBridgeTool:
    def test_uses_resolves_targets(self, vllm_state):
        out = asyncio.run(_do_bridge("rms_norm"))
        assert "Calls into pytorch" in out
        assert "`at::empty` → `aten/Empty.cpp:3`" in out
        assert "`torch::stable::contiguous` → `torch/csrc/stable/ops.h:90`" in out

    def test_bare_class_name_covers_every_definition(self, vllm_state):
        out = asyncio.run(_do_bridge("SiluAndMul"))
        assert "`aten::silu` → `silu_cpu` `aten/Act.cpp:9`" in out
        assert (
            "`aten::rsqrt` → `rsqrt_out` `aten/src/ATen/native/native_functions.yaml`"
            in out
        )

    def test_method_is_found_through_its_class(self, vllm_state):
        out = asyncio.run(_do_bridge("vllm.cpu.act.SiluAndMul.forward_native"))
        assert "`aten::rsqrt` → `rsqrt_out`" in out
        assert "aten::silu" not in out

    def test_rows_keep_a_fixed_order(self, vllm_state):
        vllm_state.external_refs = list(reversed(REFS))
        out = asyncio.run(_do_bridge("aten", mode="used_by"))
        assert out.index("vllm.act.SiluAndMul") < out.index("vllm.cpu.act.SiluAndMul")

    def test_used_by_lists_referencing_symbols(self, vllm_state):
        out = asyncio.run(_do_bridge("torch.silu", mode="used_by"))
        assert "Used by: `torch.silu`" in out
        assert "vllm.act.SiluAndMul.forward_native" in out
        assert "vllm/act.py:12" in out

    def test_unavailable_dependency_is_explained(self, vllm_state, monkeypatch):
        indexer._dependencies.clear()
        monkeypatch.setattr(indexer, "resolve_source", lambda name: None)
        out = asyncio.run(_do_bridge("rms_norm"))
        assert "no source configured for pytorch" in out
        assert "`at::empty`" in out

    def test_empty_and_unknown(self, vllm_state):
        assert asyncio.run(_do_bridge("  ")) == "Provide a symbol."
        assert "No references from `nope`" in asyncio.run(_do_bridge("nope"))
        assert "No references to `nope`" in asyncio.run(
            _do_bridge("nope", mode="used_by")
        )

    def test_no_depends_on(self, mock_state):
        set_active_harness("pytorch")
        mock_state.bindings = [{"python_name": "add"}]
        assert "no `depends_on`" in asyncio.run(_do_bridge("add"))


class TestTraceAndStatus:
    def test_trace_shows_calls_into_dependency(self, vllm_state):
        out = asyncio.run(trace("rms_norm"))
        assert "Calls into pytorch" in out
        assert "- `rms_norm` (`csrc/layernorm.cu:40`)\n" in out
        assert "  - `at::empty` → `aten/Empty.cpp:3`" in out

    def test_trace_of_python_function(self, vllm_state):
        out = asyncio.run(trace("vllm.act.SiluAndMul.forward_native"))
        assert "`aten::silu` → `silu_cpu` `aten/Act.cpp:9`" in out

    def test_status_reports_refs_and_index(self, vllm_state):
        out = asyncio.run(get_status())
        assert "cpp 2, import 1, op 2" in out
        assert "loaded from /src/pytorch" in out
        assert "`bridge`" in out

    def test_installed_header_path_is_shortened(self, vllm_state):
        assert (
            _installed_path("/usr/lib/python3/site-packages/torch/include/a.h")
            == "torch/include/a.h"
        )
        assert _installed_path("/src/vllm/csrc/a.h") == "csrc/a.h"


class TestDependencyIndex:
    @pytest.fixture(autouse=True)
    def isolated(self, monkeypatch, tmp_path):
        monkeypatch.setattr(indexer, "_dependencies", {})
        monkeypatch.setattr(indexer, "_source_fingerprint", lambda *_a: "fp")
        monkeypatch.setattr(
            indexer,
            "detect_package_identity",
            lambda s, n: PackageIdentity(n, "abc123"),
        )
        src = tmp_path / "pytorch"
        src.mkdir()
        monkeypatch.setattr(
            indexer,
            "resolve_source",
            lambda name: str(src) if name == "pytorch" else None,
        )
        monkeypatch.setattr(
            indexer,
            "cache_paths",
            lambda source, package=None: {
                "bindings": tmp_path / f"bindings_{package}.json",
                "callgraph": tmp_path / f"{package}_callgraph.json",
            },
        )
        self.src = src
        self.tmp = tmp_path

    def _write_cache(self):
        meta = indexer._cache_metadata(str(self.src), load_builtin_manifest("pytorch"))
        data = {
            "bindings": [],
            "native_functions": {"silu": {"base_name": "silu"}},
            "metadata": meta,
        }
        (self.tmp / "bindings_pytorch.json").write_text(json.dumps(data))

    def test_loads_current_cache_and_call_graph_locations(self):
        self._write_cache()
        from torchtalk.analysis.cpp_call_graph import _CALL_GRAPH_CACHE_FORMAT_VERSION

        (self.tmp / "pytorch_callgraph.json").write_text(
            json.dumps(
                {
                    "format_version": _CALL_GRAPH_CACHE_FORMAT_VERSION,
                    "source_fingerprint": "fp",
                    "function_locations": {"at::foo": ["/p/a.cpp", 3]},
                }
            )
        )
        dep = dependency_index("pytorch")
        assert dep is not None and dep.source == str(self.src)
        assert "silu" in dep.native_functions
        assert dep.cpp_locations == {"at::foo": ("/p/a.cpp", 3)}
        assert dependency_status("pytorch") == ""
        assert dependency_index("pytorch") is dep

    def test_missing_cache_gives_build_hint(self):
        assert dependency_index("pytorch") is None
        assert "torchtalk index build --harness pytorch" in dependency_status("pytorch")

    def test_missing_source_gives_config_hint(self):
        assert dependency_index("other") is None
        assert "TORCHTALK_SOURCE_OTHER" in dependency_status("other")

    @pytest.mark.parametrize(
        "meta",
        [{"format_version": 0}, {"source_fingerprint": "other"}],
    )
    def test_outdated_call_graph_cache_is_ignored(self, meta):
        from torchtalk.analysis.cpp_call_graph import _CALL_GRAPH_CACHE_FORMAT_VERSION

        self._write_cache()
        data = {
            "format_version": _CALL_GRAPH_CACHE_FORMAT_VERSION,
            "source_fingerprint": "fp",
            "function_locations": {"x": ["/a", 1]},
        }
        (self.tmp / "pytorch_callgraph.json").write_text(json.dumps(data | meta))
        assert dependency_index("pytorch").cpp_locations == {}


class TestAliasMapFromDependency:
    def test_extension_borrows_dependency_native_functions(
        self, mock_state, monkeypatch
    ):
        mock_state.native_functions = {}
        monkeypatch.setattr(indexer, "_dependencies", {"pytorch": _pytorch_index()})
        aliases = indexer._build_alias_map(load_builtin_manifest("vllm"))
        assert aliases == {
            "torch.silu": "aten::silu",
            "torch.empty": "aten::empty",
            "torch.rsqrt": "aten::rsqrt",
        }

    def test_own_op_namespace_stays_local(self, mock_state, monkeypatch):
        monkeypatch.setattr(indexer, "_dependencies", {"pytorch": _pytorch_index()})
        mock_state.native_functions = {"fused": {"base_name": "fused"}}
        vllm = load_builtin_manifest("vllm")
        namespaces = {**vllm.op_namespaces, "vllm": "vllm"}
        manifest = dataclasses.replace(vllm, op_namespaces=namespaces)
        aliases = indexer._build_alias_map(manifest)
        assert aliases["vllm.fused"] == "vllm::fused"
        assert aliases["torch.silu"] == "aten::silu"

    def test_without_dependency_index_only_own_functions(self, mock_state, monkeypatch):
        mock_state.native_functions = {}
        monkeypatch.setattr(indexer, "_dependencies", {})
        monkeypatch.setattr(indexer, "resolve_source", lambda name: None)
        assert indexer._build_alias_map(load_builtin_manifest("vllm")) == {}

    def test_edges_cache_rejects_other_alias_map(
        self, mock_state, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(indexer, "_source_fingerprint", lambda *_a: "fp")
        mock_state.source = "/fake"
        mock_state.alias_map = {"torch.silu": "aten::silu"}
        mock_state.py_to_cpp_edges = {"aten::silu": []}
        cache = tmp_path / "edges.json"
        indexer._save_py_cpp_edges_cache(cache)
        mock_state.alias_map = {"torch.gelu": "aten::gelu"}
        assert indexer._load_py_cpp_edges_cache(cache, "/fake") is False
        mock_state.alias_map = {"torch.silu": "aten::silu"}
        assert indexer._load_py_cpp_edges_cache(cache, "/fake") is True
