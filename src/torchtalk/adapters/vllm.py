"""vLLM adapter for static-first TorchTalk indexing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..analysis.vllm_index import build_vllm_index
from ..config import (
    framework_cache_path,
    resolve_framework_source,
    validate_framework_path,
)

_VLLM_CACHE_VERSION = 2
_VLLM_CAPABILITIES = frozenset(
    {
        "trace_apis",
        "trace_ops",
        "search_models",
        "search_backends",
        "search_ops",
        "search_bindings",
        "graph_flow",
    }
)


@dataclass(frozen=True)
class VllmAdapter:
    """Adapter that indexes the authoritative vLLM mapping layers."""

    framework_id: str = "vllm"
    display_name: str = "vLLM"

    def resolve_source(self, cli_flag: str | None = None) -> str | None:
        return resolve_framework_source(self.framework_id, cli_flag)

    def validate_source(self, path: str | Path) -> tuple[bool, str]:
        return validate_framework_path(self.framework_id, path)

    def bootstrap(
        self,
        source: str | None = None,
        *,
        index_path: str | None = None,
    ) -> None:
        if source:
            self._bootstrap_from_source(source)
            return
        if index_path:
            self._bootstrap_from_index(index_path)
            return
        raise ValueError("VllmAdapter.bootstrap requires source or index_path")

    def capabilities(self, state: Any | None = None) -> frozenset[str]:
        return _VLLM_CAPABILITIES

    def build_index(self, source: str, wait_for_cpp: bool = True) -> dict[str, int]:
        # `wait_for_cpp` is unused for vLLM because there is no call-graph build.
        self._bootstrap_from_source(source)
        from .. import indexer

        return {
            "bindings": len(indexer._state.entities.get("torch_custom_ops", [])),
            "cuda_kernels": 0,
            "native_functions": len(indexer._state.entities.get("ir_ops", [])),
            "derivatives": 0,
            "call_graph_functions": 0,
            "call_graph_building": False,
            "python_modules": len(indexer._state.entities.get("api_entrypoints", [])),
            "nn_modules": len(indexer._state.entities.get("pluggable_layers", [])),
            "test_files": 0,
            "test_functions": 0,
        }

    def _bootstrap_from_source(self, source: str) -> None:
        from .. import indexer

        root = Path(source).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        cache = framework_cache_path(root, self.framework_id)
        if self._cache_valid(cache, root):
            data = json.loads(cache.read_text())
        else:
            data = build_vllm_index(str(root))
            data["metadata"].update(
                {
                    "cache_version": _VLLM_CACHE_VERSION,
                    "source_fingerprint": self._source_fingerprint(root),
                }
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
        self._hydrate_state(indexer, data, str(root))

    def _bootstrap_from_index(self, index_path: str) -> None:
        from .. import indexer

        data = json.loads(Path(index_path).read_text())
        source_root = data.get("metadata", {}).get("source_path")
        self._hydrate_state(indexer, data, source_root)

    def _hydrate_state(
        self,
        indexer,
        data: dict[str, Any],
        source_root: str | None,
    ) -> None:
        state = indexer._state
        self._clear_legacy_state(indexer)

        normalized_root = str(Path(source_root).resolve()) if source_root else None
        state.framework = self.framework_id
        state.source_root = normalized_root
        state.pytorch_source = None
        state.capabilities = self.capabilities(state)
        state.entities = {
            family: data.get(family, [])
            for family in (
                "api_entrypoints",
                "request_pipeline_nodes",
                "layer_nodes",
                "platform_defaults",
                "model_architectures",
                "attention_backends",
                "ir_ops",
                "ir_providers",
                "custom_ops",
                "pluggable_layers",
                "torch_custom_ops",
            )
        }
        state.indexes = data.get("lookup_indexes", {})
        state.proof_traces = data.get("proof_traces", {})
        state.graph_nodes = data.get("graph_nodes", {})
        state.graph_edges = data.get("graph_edges", [])
        state.dynamic_notes = data.get("dynamic_notes", [])
        state.entity_counts = {
            family: len(records)
            for family, records in state.entities.items()
            if isinstance(records, list)
        }

    def _clear_legacy_state(self, indexer) -> None:
        state = indexer._state
        state.bindings = []
        state.cuda_kernels = []
        state.native_functions = {}
        state.derivatives = {}
        state.native_implementations = {}
        state.symbol_to_file = {}
        state.by_python_name = {}
        state.by_cpp_name = {}
        state.by_dispatch_key = {}
        state.bindings_by_file = {}
        state.ops_by_file = {}
        state.py_modules = {}
        state.py_classes = {}
        state.py_functions = {}
        state.nn_modules = []
        state.py_to_cpp_edges = {}
        state.alias_map = {}
        state.test_files = {}
        state.test_classes = {}
        state.test_functions = {}
        state.test_utilities = {}
        state.opinfo_registry = {}
        state.opinfo_alias_map = {}
        state.opinfo_test_files = set()
        state.test_attr_index = {}
        state.python_profiling = {}
        state.decomp_alias_map = {}
        state.backward_to_forward = {}
        state.kernel_impl_to_op = {}
        state.dispatch_to_op = {}
        state.cpp_extractor = None
        state.cpp_building = False
        state.cpp_thread = None

    def _cache_valid(self, cache: Path, source_root: Path) -> bool:
        if not cache.exists():
            return False
        try:
            payload = json.loads(cache.read_text())
        except json.JSONDecodeError:
            return False

        metadata = payload.get("metadata", {})
        if metadata.get("cache_version") != _VLLM_CACHE_VERSION:
            return False
        return metadata.get("source_fingerprint") == self._source_fingerprint(
            source_root
        )

    def _source_fingerprint(self, source_root: Path) -> str:
        parts = []
        for path in (
            source_root / "pyproject.toml",
            source_root / "vllm" / "version.py",
            source_root / ".git" / "HEAD",
        ):
            if path.exists():
                stat = path.stat()
                parts.append(f"{path.name}:{stat.st_mtime}:{stat.st_size}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]
