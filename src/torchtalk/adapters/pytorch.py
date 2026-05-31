"""PyTorch adapter for legacy TorchTalk bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import resolve_framework_source, validate_framework_path

_CORE_CAPABILITIES = frozenset({"trace_ops", "search_bindings", "search_kernels"})


@dataclass(frozen=True)
class PyTorchAdapter:
    """Thin adapter that delegates to the existing PyTorch bootstrap."""

    framework_id: str = "pytorch"
    display_name: str = "PyTorch"

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
        from .. import indexer

        if source:
            self._bootstrap_from_source(indexer, source)
            return
        if index_path:
            self._bootstrap_from_index(indexer, index_path)
            return
        raise ValueError("PyTorchAdapter.bootstrap requires source or index_path")

    def capabilities(self, state: Any | None = None) -> frozenset[str]:
        if state is None:
            return _CORE_CAPABILITIES

        capabilities = set(_CORE_CAPABILITIES)
        if state.cpp_extractor or state.cpp_building:
            capabilities.add("graph_cpp")
        if state.py_modules:
            capabilities.add("python_modules")
        if state.test_files:
            capabilities.add("tests")
        if state.cpp_extractor and state.test_files:
            capabilities.add("affected")
        return frozenset(capabilities)

    def build_index(self, source: str, wait_for_cpp: bool = True) -> dict[str, int]:
        from .. import indexer

        self._bootstrap_from_source(indexer, source)

        if (
            wait_for_cpp
            and indexer._state.cpp_thread is not None
            and indexer._state.cpp_thread.is_alive()
        ):
            indexer.log.info("Waiting for C++ call graph build to finish...")
            indexer._state.cpp_thread.join()

        cg_functions = 0
        if indexer._state.cpp_extractor is not None:
            cg_functions = len(indexer._state.cpp_extractor.function_locations)

        return {
            "bindings": len(indexer._state.bindings),
            "cuda_kernels": len(indexer._state.cuda_kernels),
            "native_functions": len(indexer._state.native_functions),
            "derivatives": len(indexer._state.derivatives),
            "call_graph_functions": cg_functions,
            "call_graph_building": indexer._state.cpp_building,
            "python_modules": len(indexer._state.py_modules),
            "nn_modules": len(indexer._state.nn_modules),
            "test_files": len(indexer._state.test_files),
            "test_functions": len(indexer._state.test_functions),
        }

    def _bootstrap_from_source(self, indexer, source: str) -> None:
        src = Path(source).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        resolved_source = str(src)
        self._set_framework_context(indexer, resolved_source)

        cache = indexer._cache_path(resolved_source)
        if indexer._cache_valid(cache, resolved_source):
            indexer.log.info(f"Using cached index from {cache}")
            indexer._load_from_json(str(cache), framework=self.framework_id)
        else:
            data = indexer._build_index(resolved_source)
            state = indexer._state
            state.bindings = data.get("bindings", [])
            state.cuda_kernels = data.get("cuda_kernels", [])
            state.native_functions = data.get("native_functions", {})
            state.derivatives = data.get("derivatives", {})
            state.native_implementations = data.get("native_implementations", {})
            state.symbol_to_file = data.get("symbol_to_file", {})
            indexer._build_indexes(state)

        self._finalize_loaded_state(indexer, resolved_source)

    def _bootstrap_from_index(self, indexer, index_path: str) -> None:
        indexer._load_from_json(index_path, framework=self.framework_id)
        source_root = indexer._state.source_root or indexer._state.pytorch_source
        if source_root and Path(source_root).exists():
            self._finalize_loaded_state(indexer, source_root)
        else:
            self._refresh_framework_views(indexer)

    def _set_framework_context(self, indexer, source_root: str | None) -> None:
        state = indexer._state
        state.framework = self.framework_id
        state.source_root = source_root
        state.pytorch_source = source_root

    def _finalize_loaded_state(self, indexer, source_root: str) -> None:
        self._set_framework_context(indexer, source_root)
        indexer._init_decomp_aliases(source_root)
        indexer._init_backward_bridge()
        indexer._init_dispatch_stubs(source_root)
        indexer._init_cpp_call_graph(source_root)
        indexer._init_python_modules(source_root)
        indexer._init_test_infrastructure(source_root)
        self._refresh_framework_views(indexer)

    def _refresh_framework_views(self, indexer) -> None:
        state = indexer._state
        state.capabilities = self.capabilities(state)
        state.entity_counts = {
            "bindings": len(state.bindings),
            "cuda_kernels": len(state.cuda_kernels),
            "native_functions": len(state.native_functions),
            "derivatives": len(state.derivatives),
            "python_modules": len(state.py_modules),
            "nn_modules": len(state.nn_modules),
            "test_files": len(state.test_files),
            "test_functions": len(state.test_functions),
            "opinfo_registry": len(state.opinfo_registry),
        }
        state.entities = {
            "bindings": state.bindings,
            "cuda_kernels": state.cuda_kernels,
            "native_functions": state.native_functions,
            "derivatives": state.derivatives,
            "python_modules": state.py_modules,
            "tests": state.test_files,
            "opinfo_registry": state.opinfo_registry,
        }
        state.indexes = {
            "by_python_name": state.by_python_name,
            "by_cpp_name": state.by_cpp_name,
            "by_dispatch_key": state.by_dispatch_key,
            "bindings_by_file": state.bindings_by_file,
            "dispatch_to_op": state.dispatch_to_op,
        }
