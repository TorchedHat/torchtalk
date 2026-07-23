"""Data loading, caching, and index initialization for TorchTalk."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .analysis.helpers import fuzzy_distance_limit, levenshtein_distance, truncate
from .analysis.patterns import (
    has_binding_patterns as _has_binding_patterns,
)
from .analysis.patterns import (
    has_test_patterns as _has_test_patterns,
)
from .analysis.patterns import is_vendor_path
from .analysis.patterns import (
    should_exclude as _should_exclude,
)
from .analysis.patterns import (
    should_include_dir as _should_include_dir,
)
from .config import CACHE_DIR, cache_paths, resolve_source
from .harness import ConventionManifest, active_manifest
from .symbols import PackageIdentity, content_fingerprint, detect_package_identity

log = logging.getLogger(__name__)


@dataclass
class ServerState:
    """Server state container."""

    bindings: list[dict] = field(default_factory=list)
    cuda_kernels: list[dict] = field(default_factory=list)
    native_functions: dict[str, dict] = field(default_factory=dict)
    derivatives: dict[str, dict] = field(default_factory=dict)
    native_implementations: dict[str, list[dict]] = field(default_factory=dict)
    symbol_to_file: dict[str, str] = field(default_factory=dict)
    registrations: dict[str, list[dict]] = field(default_factory=dict)

    by_python_name: dict[str, list[dict]] = field(default_factory=dict)
    by_cpp_name: dict[str, list[dict]] = field(default_factory=dict)
    by_dispatch_key: dict[str, list[dict]] = field(default_factory=dict)
    bindings_by_file: dict[str, list[dict]] = field(default_factory=dict)
    ops_by_file: dict[str, set[str]] = field(default_factory=dict)

    py_modules: dict[str, Any] = field(default_factory=dict)
    py_classes: dict[str, list[Any]] = field(default_factory=dict)
    py_functions: dict[str, list[Any]] = field(default_factory=dict)
    nn_modules: list[Any] = field(default_factory=list)
    py_to_cpp_edges: dict[str, list[dict]] = field(default_factory=dict)
    alias_map: dict[str, str] = field(default_factory=dict)

    test_files: dict[str, dict] = field(default_factory=dict)
    test_classes: dict[str, list[dict]] = field(default_factory=dict)
    test_functions: dict[str, list[dict]] = field(default_factory=dict)
    test_utilities: dict[str, dict] = field(default_factory=dict)
    opinfo_registry: dict[str, dict] = field(default_factory=dict)
    opinfo_alias_map: dict[str, list[dict]] = field(default_factory=dict)
    opinfo_test_files: set[str] = field(default_factory=set)
    test_attr_index: dict[str, list[dict]] = field(default_factory=dict)
    python_profiling: dict[str, dict[str, float]] = field(default_factory=dict)
    decomp_alias_map: dict[str, list[str]] = field(default_factory=dict)
    backward_to_forward: dict[str, list[str]] = field(default_factory=dict)
    kernel_impl_to_op: dict[str, str] = field(default_factory=dict)
    dispatch_to_op: dict[str, str] = field(default_factory=dict)

    pytorch_source: str | None = None
    package: PackageIdentity | None = None
    cpp_extractor: Any = None
    cpp_building: bool = False
    cpp_thread: Any = None


_state = ServerState()


def _cache_path(source: str) -> Path:
    """Get cache file path for source directory."""
    return cache_paths(source)["bindings"]


def _source_fingerprint(source: str) -> str:
    """Fingerprint source to detect changes.

    Git checkouts get a content fingerprint (HEAD tree + dirty diff), so
    commits, pulls, and uncommitted edits all invalidate caches. Non-git
    checkouts fall back to a coarse tree signal (file count + newest mtime
    across the manifest's search dirs) so edits still invalidate; version
    files alone would fingerprint every content state identically.
    """
    fp = content_fingerprint(source) if source else None
    if fp:
        return fp
    src = Path(source)
    manifest = active_manifest()
    newest, count = 0.0, 0
    for d in sorted({*manifest.cpp_search_dirs, *manifest.python_search_dirs}):
        root = src / d
        if not root.exists():
            continue
        for p in root.rglob("*"):
            try:
                if p.is_file():
                    count += 1
                    newest = max(newest, p.stat().st_mtime)
            except OSError:
                continue
    return hashlib.md5(f"tree:{count}:{newest}".encode()).hexdigest()[:16]


# Bump when the bindings-cache schema changes (e.g. a new field added to
# `native_functions` entries). v2 introduced `python_module`; v3 added
# package identity to metadata; v4 added the `registrations` section; v5
# renamed binding "line" → "line_number".
_BINDINGS_CACHE_FORMAT_VERSION = 5


def _cache_metadata(source: str, manifest: ConventionManifest | None = None) -> dict:
    """Metadata every bindings-cache writer must emit; `_cache_valid` checks it."""
    manifest = manifest or active_manifest()
    return {
        "source_path": source,
        "source_fingerprint": _source_fingerprint(source),
        "format_version": _BINDINGS_CACHE_FORMAT_VERSION,
        "package": asdict(detect_package_identity(source, manifest.package)),
    }


def _cache_valid(cache: Path, source: str) -> bool:
    """Check if cache is valid."""
    if not cache.exists():
        return False
    try:
        with open(cache) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        if meta.get("format_version") != _BINDINGS_CACHE_FORMAT_VERSION:
            return False
        if meta.get("package") != asdict(
            detect_package_identity(source, active_manifest().package)
        ):
            return False
        return meta.get("source_fingerprint") == _source_fingerprint(source)
    except Exception:
        return False


def _parse_native_functions(
    source: str, manifest: ConventionManifest | None = None
) -> tuple[dict, dict]:
    """Parse the manifest's YAML op/derivative sources; empty when undeclared."""
    import yaml

    manifest = manifest or active_manifest()
    src = Path(source)
    functions, derivatives = {}, {}

    nf_yaml = (
        src / manifest.native_functions_yaml if manifest.native_functions_yaml else None
    )
    if nf_yaml is not None and nf_yaml.exists():
        try:
            data = yaml.safe_load(nf_yaml.read_text())
            for entry in data or []:
                if not isinstance(entry, dict) or "func" not in entry:
                    continue

                sig = entry["func"]
                match = re.match(r"(\w+(?:\.\w+)?)\s*\(", sig)
                if not match:
                    continue

                name = match.group(1)
                base = name.split(".")[0]

                dispatch = {}
                if isinstance(entry.get("dispatch"), dict):
                    for key, impl in entry["dispatch"].items():
                        for k in key.split(","):
                            dispatch[k.strip()] = impl

                # YAML emits single-tag entries as a bare string, multi-tag
                # entries as a list. Downstream code iterates `tags` element-
                # wise — normalize so a single tag never iterates as chars.
                tags = entry.get("tags") or []
                if isinstance(tags, str):
                    tags = [tags]

                func = {
                    "name": name,
                    "base_name": base,
                    "signature": sig,
                    "dispatch": dispatch,
                    "variants": entry.get("variants", ""),
                    "python_module": entry.get("python_module", "") or "",
                    "structured": entry.get("structured", False),
                    "structured_delegate": entry.get("structured_delegate"),
                    "tags": tags,
                }
                functions[name] = func
                if name != base and base not in functions:
                    functions[base] = func

            log.info(f"Parsed {len(functions)} native functions")
        except Exception as e:
            log.warning(f"Failed to parse native_functions.yaml: {e}")

    deriv_yaml = src / manifest.derivatives_yaml if manifest.derivatives_yaml else None
    if deriv_yaml is not None and deriv_yaml.exists():
        try:
            data = yaml.safe_load(deriv_yaml.read_text())
            for entry in data or []:
                if not isinstance(entry, dict) or "name" not in entry:
                    continue

                match = re.match(r"(\w+(?:\.\w+)?)", entry["name"])
                if not match:
                    continue

                name = match.group(1)
                grads = {
                    k: v
                    for k, v in entry.items()
                    if k not in {"name", "dispatch", "output_differentiability"}
                    and isinstance(v, str)
                }

                derivatives[name] = {
                    "name": name,
                    "signature": entry["name"],
                    "gradients": grads,
                }

            log.info(f"Parsed {len(derivatives)} derivatives")
        except Exception as e:
            log.warning(f"Failed to parse derivatives.yaml: {e}")

    return functions, derivatives


# Identifier with optional namespace, optional template args (one nesting
# level), optional ptr/ref qualifier. Conservative — anything beyond two
# levels of nesting falls back to the libclang lookup below.
_TYPE_TOKEN = (
    r"[\w:]+"
    r"(?:\s*<[^<>]*(?:<[^<>]*>[^<>]*)*>)?"
    r"(?:\s*[*&]+)?"
)
_MODIFIERS = (
    r"(?:(?:static|inline|constexpr|TORCH_API|"
    r"C10_HOST_DEVICE|C10_HOST|C10_DEVICE|C10_API|"
    r"__forceinline__|virtual|explicit)\s+)*"
)
_TEMPLATE_PREFIX = r"(?:template\s*<[^>]+>\s*)?"
_FREE_FUNC_RE = re.compile(
    rf"^{_TEMPLATE_PREFIX}{_MODIFIERS}{_TYPE_TOKEN}\s+"
    rf"(\w+)\s*\([^;]*\)\s*(?:const\s*)?\{{",
    re.MULTILINE,
)
_METHOD_FUNC_RE = re.compile(
    rf"^{_TEMPLATE_PREFIX}{_MODIFIERS}{_TYPE_TOKEN}\s+"
    rf"(?:\w+::)+(\w+)\s*\([^;]*\)\s*(?:const\s*)?\{{",
    re.MULTILINE,
)


def _impls_from_extractor(target: str) -> list[dict]:
    """Look up `target` in the libclang call graph by bare-name suffix.

    Hybrid path: catches definitions the regex pre-filter misses (deeply
    templated returns, signatures whose layout the regex can't span).
    Returns [] when the extractor isn't ready. Matches outside the indexed
    source tree (stdlib/system headers the TU pulled in) are dropped.
    """
    extractor = _state.cpp_extractor
    if not extractor:
        return []
    src_prefix = (
        (_state.pytorch_source.rstrip("/") + "/") if _state.pytorch_source else None
    )
    out: list[dict] = []
    for qname, (file, line) in extractor.function_locations.items():
        if qname.rsplit("::", 1)[-1] != target:
            continue
        if src_prefix and not file.startswith(src_prefix):
            continue
        out.append(
            {
                "function_name": target,
                "file_path": file,
                "line_number": line,
                "signature": qname,
            }
        )
    return out


def _find_implementations(
    source: str, functions: dict, manifest: ConventionManifest | None = None
) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Find C++ implementations in source using configured search directories.

    Returns (impls, symbol_to_file). `impls` is YAML-targets-filtered as before.
    `symbol_to_file` indexes ALL function definitions found in vendor backend
    dirs (cudnn/miopen/mkldnn/...), bypassing the targets filter — needed for
    helper functions like `run_cudnn_SDP_fprop` that aren't YAML ops but still
    need a file-cohort bridge for test selection.
    """
    manifest = manifest or active_manifest()
    src = Path(source)

    search_dirs = [src / d for d in manifest.cpp_search_dirs]

    targets = set()
    for f in functions.values():
        targets.add(f["base_name"])
        targets.update(f.get("dispatch", {}).values())

    log.info(
        f"Searching for {len(targets)} implementations "
        f"in {len(search_dirs)} directories..."
    )

    patterns = [_FREE_FUNC_RE, _METHOD_FUNC_RE]

    impls: dict[str, list[dict]] = {}
    symbol_to_file: dict[str, str] = {}
    file_count = 0

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for cpp_file in search_dir.rglob("*.cpp"):
            file_str = str(cpp_file)

            # Apply exclusion patterns
            if _should_exclude(file_str, manifest.exclude_patterns):
                continue

            try:
                content = cpp_file.read_text(errors="ignore")
            except Exception:
                continue

            in_vendor = is_vendor_path(file_str)

            file_count += 1
            for pattern in patterns:
                for match in pattern.finditer(content):
                    func_name = match.group(1)
                    if in_vendor:
                        symbol_to_file.setdefault(func_name, file_str)
                    if func_name not in targets:
                        continue

                    line = content[: match.start()].count("\n") + 1
                    sig_end = content.find("\n", match.start())
                    sig = (
                        content[match.start() : sig_end].strip() if sig_end > 0 else ""
                    )

                    impl = {
                        "function_name": func_name,
                        "file_path": str(cpp_file),
                        "line_number": line,
                        "signature": truncate(sig, 100),
                    }
                    impls.setdefault(func_name, []).append(impl)

    log.info(
        f"Found {sum(len(v) for v in impls.values())} "
        f"implementations in {file_count} files; "
        f"{len(symbol_to_file)} vendor-helper symbols indexed"
    )
    return impls, symbol_to_file


def _fuzzy_find(name: str, data: dict[str, Any]) -> list[Any] | None:
    """Fuzzy match function name in a dict."""
    if name in data:
        return data[name] if isinstance(data[name], list) else [data[name]]

    name_lower = name.lower()

    suffix_keys = [
        k
        for k in data
        if k.lower().endswith(name_lower) or k.lower().endswith(f"_{name_lower}")
    ]
    if suffix_keys:
        # Deterministic: shortest key, lexicographic tiebreak (not dict order)
        key = min(suffix_keys, key=lambda k: (len(k), k))
        return data[key] if isinstance(data[key], list) else [data[key]]

    matches = [(k, v) for k, v in data.items() if name_lower in k.lower()]
    if matches:
        matches.sort(key=lambda x: (len(x[0]), x[0]))
        v = matches[0][1]
        return v if isinstance(v, list) else [v]

    if len(name) >= 3:
        candidates = []
        for key in data:
            if abs(len(key) - len(name)) <= 3:
                dist = levenshtein_distance(name_lower, key.lower())
                if dist <= fuzzy_distance_limit(name):
                    candidates.append((dist, key, data[key]))

        if candidates:
            candidates.sort(key=lambda x: (x[0], len(x[1]), x[1]))
            v = candidates[0][2]
            return v if isinstance(v, list) else [v]

    return None


def _build_index(
    source: str, manifest: ConventionManifest | None = None
) -> dict[str, Any]:
    """Build binding index from source."""
    from .analysis.binding_detector import BindingDetector
    from .analysis.extractors import extract_registrations

    manifest = manifest or active_manifest()
    log.info(f"Building index from {source}...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    functions, derivatives = _parse_native_functions(source, manifest)

    implementations, symbol_to_file = _find_implementations(source, functions, manifest)

    detector = BindingDetector(
        macro_aliases=manifest.cpp_macro_aliases,
        token_map=manifest.cpp_token_map,
        search_dirs=manifest.cpp_search_dirs,
        exclude_patterns=manifest.exclude_patterns,
        registration_macros=manifest.registration_macros,
    )
    graph = detector.detect_bindings_in_directory(source)

    registrations = extract_registrations(source, manifest)
    log.info(
        f"Registrations: {len(registrations['records'])} records, "
        f"{len(registrations['candidate_edges'])} candidate edges"
    )

    data = {
        "bindings": [b.to_dict() for b in graph.bindings],
        "cuda_kernels": [asdict(k) for k in graph.cuda_kernels],
        "native_functions": functions,
        "derivatives": derivatives,
        "native_implementations": implementations,
        "symbol_to_file": symbol_to_file,
        "registrations": registrations,
        "metadata": _cache_metadata(source, manifest),
    }

    cache = _cache_path(source)
    with open(cache, "w") as f:
        json.dump(data, f)
    log.info(f"Cached index to {cache}")

    return data


def _build_indexes(state: ServerState):
    """Build lookup indexes. Clears prior contents so reloads don't duplicate."""
    state.by_python_name.clear()
    state.by_cpp_name.clear()
    state.by_dispatch_key.clear()
    state.bindings_by_file.clear()
    state.dispatch_to_op.clear()
    for binding in state.bindings:
        if py_name := binding.get("python_name"):
            state.by_python_name.setdefault(py_name, []).append(binding)
        if cpp_name := binding.get("cpp_name"):
            state.by_cpp_name.setdefault(cpp_name, []).append(binding)
        if dispatch := binding.get("dispatch_key"):
            state.by_dispatch_key.setdefault(dispatch, []).append(binding)
        if file_path := binding.get("file_path"):
            state.bindings_by_file.setdefault(file_path, []).append(binding)
    # Reverse index: vendor-backend impl symbol → parent ATen op. Captures the
    # `dispatch: { CUDA: cudnn_convolution_forward }` shape so a walk that lands
    # on `cudnn_convolution_forward` resolves to `cudnn_convolution`.
    for op_name, entry in state.native_functions.items():
        for impl in entry.get("dispatch", {}).values():
            if impl and impl != op_name:
                state.dispatch_to_op.setdefault(impl, op_name)
    # File → ops defined in it, derived from `native_implementations`. Used to
    # bridge inner vendor-backend helpers (`raw_cudnn_convolution_forward_out`)
    # to the registered op family in the same file (`cudnn_convolution`,
    # `cudnn_convolution_transpose`, ...) when the call graph lacks edges.
    state.ops_by_file.clear()
    for op_name, impls in state.native_implementations.items():
        for impl in impls:
            if fp := impl.get("file_path"):
                state.ops_by_file.setdefault(fp, set()).add(op_name)


def _load_from_json(path: str):
    """Load bindings from JSON file."""
    global _state

    log.info(f"Loading bindings from {path}...")
    with open(path) as f:
        data = json.load(f)

    _state.bindings = data.get("bindings", [])
    _state.cuda_kernels = data.get("cuda_kernels", [])
    _state.native_functions = data.get("native_functions", {})
    _state.derivatives = data.get("derivatives", {})
    _state.native_implementations = data.get("native_implementations", {})
    _state.symbol_to_file = data.get("symbol_to_file", {})
    _state.registrations = data.get("registrations", {})

    _build_indexes(_state)

    log.info(
        f"Loaded {len(_state.bindings)} bindings, "
        f"{len(_state.cuda_kernels)} CUDA kernels"
    )


def _init_cpp_call_graph(source: str):
    """Initialize C++ call graph (background thread on first run)."""
    global _state

    try:
        from .analysis.cpp_call_graph import LIBCLANG_AVAILABLE, CppCallGraphExtractor

        if not LIBCLANG_AVAILABLE:
            log.info("libclang not available - C++ call graph disabled")
            return

        cg_cache_dir = CACHE_DIR / "call_graph"
        cache_key = cache_paths(source)["callgraph"].stem

        extractor = CppCallGraphExtractor(cache_dir=cg_cache_dir)
        if extractor.load_cache(
            cache_key, expect_fingerprint=_source_fingerprint(source)
        ):
            _state.cpp_extractor = extractor
            log.info(
                f"Loaded C++ call graph from cache "
                f"({len(extractor.function_locations)} functions)"
            )
            return

        # Build in background
        import threading

        manifest = active_manifest()

        def build():
            global _state
            try:
                ext = CppCallGraphExtractor(cache_dir=cg_cache_dir)
                ext.extract_from_pytorch_parallel(
                    source,
                    include_dirs=list(manifest.cpp_search_dirs),
                    exclude_patterns=manifest.exclude_patterns,
                )
                ext.save_cache(cache_key, fingerprint=_source_fingerprint(source))
                _state.cpp_extractor = ext
                log.info(
                    f"C++ call graph ready: {len(ext.function_locations)} functions"
                )
            except Exception as e:
                log.warning(f"C++ call graph build failed: {e}")
            finally:
                _state.cpp_building = False

        _state.cpp_building = True
        log.info("Building C++ call graph in background (~1-2 min)...")
        _state.cpp_thread = threading.Thread(target=build, daemon=True)
        _state.cpp_thread.start()

    except Exception as e:
        log.warning(f"Failed to init C++ call graph: {e}")


def _cpp_status() -> str:
    """Get C++ call graph status. Empty string if ready."""
    if _state.cpp_building:
        return "C++ call graph building in background (~1-2 min). Try again shortly."

    if not _state.cpp_extractor:
        if _state.pytorch_source:
            src = Path(_state.pytorch_source)
            if (
                not (src / "compile_commands.json").exists()
                and not (src / "build/compile_commands.json").exists()
            ):
                return (
                    "C++ call graph unavailable - "
                    "`compile_commands.json` not found.\n\n"
                    "**To enable:** Build PyTorch once:\n"
                    f"```\ncd {_state.pytorch_source}\npython setup.py develop\n```"
                )
        return "C++ call graph unavailable. Install libclang or build PyTorch."

    return ""


def _ensure_loaded(component: str = "bindings"):
    """Ensure required data is loaded.

    Args:
        component: What to check - "bindings" (default), "test", "python", "cpp"
    """
    checks = {
        "bindings": (_state.bindings or _state.native_functions, "No data loaded"),
        "test": (_state.test_files, "Test infrastructure not loaded"),
        "python": (_state.py_modules, "Python module analysis not available"),
        "cpp": (_state.cpp_extractor, "C++ call graph not available"),
    }
    loaded, msg = checks.get(component, (True, ""))
    if not loaded:
        raise RuntimeError(f"{msg}. Start server with --pytorch-source.")


def _init_python_modules(source: str):
    """Initialize Python module analysis using configured search directories."""
    global _state

    try:
        from .analysis.alias_map import build_function_alias_map
        from .analysis.python_analyzer import PythonAnalyzer, build_module_index

        _state.alias_map = build_function_alias_map(_state.native_functions)
        log.info(f"Alias map: {len(_state.alias_map)} torch.<op> aliases")

        manifest = active_manifest()
        analyzer = PythonAnalyzer(
            alias_map=_state.alias_map,
            package_roots=manifest.python_package_roots or None,
        )
        src = Path(source)

        dirs_to_analyze = [src / d for d in manifest.python_search_dirs]

        all_modules: dict[str, Any] = {}
        analyzed_count = 0
        for dir_path in dirs_to_analyze:
            if dir_path.exists():
                modules = analyzer.analyze_directory(str(dir_path), skip_tests=True)
                all_modules.update(modules)
                analyzed_count += 1

        log.info(f"Analyzed {analyzed_count}/{len(dirs_to_analyze)} Python directories")

        if all_modules:
            index = build_module_index(all_modules)
            _state.py_modules = all_modules
            _state.py_classes = index["by_class"]
            _state.py_functions = index["by_function"]
            _state.nn_modules = index["nn_modules"]
            log.info(
                f"Loaded {len(all_modules)} Python modules, "
                f"{len(_state.nn_modules)} nn.Module classes"
            )
            _init_py_cpp_edges(source)
    except Exception as e:
        log.warning(f"Failed to analyze Python modules: {e}")


# v3: edges now include import-aware resolution (`linalg.cross(...)` →
# `aten::linalg_cross` after `from torch import linalg`) and namespaced
# aliases. Bump invalidates v2 caches built before import tracking.
_PY_CPP_EDGES_CACHE_VERSION = 3


def _build_py_to_cpp_edges(modules: dict[str, Any]) -> dict[str, list[dict]]:
    """Reverse-index PyFunction.cpp_bindings as cpp_symbol → caller sites."""
    edges: dict[str, list[dict]] = {}

    def record(func, cpp_bindings):
        for b in cpp_bindings:
            edges.setdefault(b.cpp_symbol, []).append(
                {
                    "caller_qualname": func.qualified_name,
                    "file": func.file_path,
                    "line": b.line,
                }
            )

    for module in modules.values():
        for func in module.functions:
            record(func, func.cpp_bindings)
        for cls in module.classes:
            for method in cls.methods:
                record(method, method.cpp_bindings)
    return edges


def _save_py_cpp_edges_cache(path: Path) -> None:
    """Persist the py→cpp edge index to JSON, versioned and fingerprinted."""
    payload = {
        "version": _PY_CPP_EDGES_CACHE_VERSION,
        "fingerprint": _source_fingerprint(_state.pytorch_source or ""),
        "edges": _state.py_to_cpp_edges,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _load_py_cpp_edges_cache(path: Path, source: str) -> bool:
    """Hydrate edge index from cache. Returns True on success."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("version") != _PY_CPP_EDGES_CACHE_VERSION:
        return False
    if payload.get("fingerprint") != _source_fingerprint(source):
        return False
    _state.py_to_cpp_edges = payload.get("edges", {})
    return True


def _init_py_cpp_edges(source: str) -> None:
    """Build (or load from cache) the cpp_symbol → Python-callsite reverse index."""
    cache = cache_paths(source)["py_cpp_edges"]
    if cache.exists() and _load_py_cpp_edges_cache(cache, source):
        log.info(f"Py→Cpp edges (cached): {len(_state.py_to_cpp_edges)} symbols")
        return
    _state.py_to_cpp_edges = _build_py_to_cpp_edges(_state.py_modules)
    log.info(f"Py→Cpp edges: {len(_state.py_to_cpp_edges)} symbols")
    try:
        _save_py_cpp_edges_cache(cache)
    except OSError as e:
        log.debug(f"Failed to save py→cpp edges cache: {e}")


def _init_from_source(source: str):
    """Initialize from PyTorch source."""
    global _state

    src = Path(source).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    cache = _cache_path(str(src))
    if _cache_valid(cache, str(src)):
        log.info(f"Using cached index from {cache}")
        _load_from_json(str(cache))
    else:
        data = _build_index(str(src))
        _state.bindings = data.get("bindings", [])
        _state.cuda_kernels = data.get("cuda_kernels", [])
        _state.native_functions = data.get("native_functions", {})
        _state.derivatives = data.get("derivatives", {})
        _state.native_implementations = data.get("native_implementations", {})
        _state.symbol_to_file = data.get("symbol_to_file", {})
        _state.registrations = data.get("registrations", {})
        _build_indexes(_state)

    _state.pytorch_source = str(src)
    _state.package = detect_package_identity(str(src), active_manifest().package)
    _init_decomp_aliases(str(src))
    _init_backward_bridge()
    _init_dispatch_stubs(str(src))
    _init_cpp_call_graph(str(src))
    _init_python_modules(str(src))
    _init_test_infrastructure(str(src))


def _init_decomp_aliases(source: str):
    """Build aten ↔ python-fn alias map from decomp/refs decorators."""
    from .analysis.decomp_aliases import extract_decomp_aliases

    _state.decomp_alias_map = extract_decomp_aliases(Path(source))
    log.info(f"Decomp aliases: {len(_state.decomp_alias_map)} entries")


def _init_backward_bridge():
    """Build backward → forward op map by scanning derivatives.yaml formulas."""
    from .analysis.backward_bridge import extract_backward_to_forward

    _state.backward_to_forward = extract_backward_to_forward(_state.derivatives)
    log.info(f"Backward bridge: {len(_state.backward_to_forward)} entries")


def _init_dispatch_stubs(source: str):
    """Build kernel-impl → ATen op map from REGISTER_DISPATCH macros."""
    from .analysis.dispatch_stub_map import extract_kernel_impl_to_op

    _state.kernel_impl_to_op = extract_kernel_impl_to_op(
        Path(source), _state.native_functions
    )
    log.info(f"Dispatch stub map: {len(_state.kernel_impl_to_op)} kernel impls")


def load_python_profiling(path: str) -> int:
    """Load PyTorch's `td_heuristic_profiling.json` into state. Returns entry count.

    Download: https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/td_heuristic_profiling.json
    """
    data = json.loads(Path(path).read_text())
    _state.python_profiling = data
    log.info(f"Python profiling: {len(data)} source files")
    return len(data)


# v2: mention vocabulary reseeded from native_functions + OpInfo names.
# v3: vocabulary gains nn-Module PascalCase variants (`AvgPool2d`).
_TEST_INFRA_CACHE_VERSION = 3


def _save_test_infra_cache(path: Path) -> None:
    """Persist test infrastructure state to JSON. Sets become sorted lists."""
    payload = {
        "version": _TEST_INFRA_CACHE_VERSION,
        "fingerprint": _source_fingerprint(_state.pytorch_source or ""),
        "test_files": _state.test_files,
        "test_classes": _state.test_classes,
        "test_functions": _state.test_functions,
        "test_utilities": _state.test_utilities,
        "opinfo_registry": _state.opinfo_registry,
        "opinfo_alias_map": _state.opinfo_alias_map,
        "opinfo_test_files": sorted(_state.opinfo_test_files),
        "test_attr_index": _state.test_attr_index,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _load_test_infra_cache(path: Path, source: str) -> bool:
    """Hydrate state from a test-infra cache file. Returns True on success."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("version") != _TEST_INFRA_CACHE_VERSION:
        return False
    if payload.get("fingerprint") != _source_fingerprint(source):
        return False
    _state.test_files = payload["test_files"]
    _state.test_classes = payload["test_classes"]
    _state.test_functions = payload["test_functions"]
    _state.test_utilities = payload["test_utilities"]
    _state.opinfo_registry = payload["opinfo_registry"]
    _state.opinfo_alias_map = payload["opinfo_alias_map"]
    _state.opinfo_test_files = set(payload["opinfo_test_files"])
    _state.test_attr_index = payload["test_attr_index"]
    return True


def _init_test_infrastructure(source: str):
    """Initialize test infrastructure analysis (cache-first)."""
    global _state

    cache = cache_paths(source)["test_infra"]
    if cache.exists() and _load_test_infra_cache(cache, source):
        log.info(
            f"Test infrastructure (cached): {len(_state.test_files)} files, "
            f"{len(_state.test_classes)} test classes, "
            f"{len(_state.test_functions)} test functions"
        )
        return

    try:
        import ast

        from .analysis.affected import api_attr_variants, normalize_api

        src = Path(source)

        log.info("Analyzing test infrastructure...")

        # OpInfo registry must exist before the scan: it feeds the mention
        # vocabulary below.
        opinfo_dirs = [
            src / "torch/testing/_internal/opinfo",
            src / "torch/testing/_internal/opinfo/definitions",
        ]
        for opinfo_dir in opinfo_dirs:
            if opinfo_dir.exists():
                for opinfo_file in opinfo_dir.glob("*.py"):
                    _parse_opinfo_registry(str(opinfo_file))
        op_db_main = src / "torch/testing/_internal/common_methods_invocations.py"
        if op_db_main.exists():
            _parse_opinfo_registry(str(op_db_main))

        # Bound the attr index by only recording mentions of names that could
        # plausibly resolve to an API: binding python_names, YAML op names, and
        # OpInfo names/aliases. Seeding from bindings alone missed `cummax`,
        # `gelu`, `avg_pool2d` — the ops CI selection needs most.
        interesting_attrs: set[str] = set()
        for py_name in _state.by_python_name:
            interesting_attrs.update(api_attr_variants(normalize_api(py_name)))
        for name, entry in _state.native_functions.items():
            interesting_attrs.update(api_attr_variants(entry.get("base_name") or name))
        for op_name, entry in _state.opinfo_registry.items():
            interesting_attrs.update(api_attr_variants(op_name))
            for alias in entry.get("aliases", []):
                interesting_attrs.update(api_attr_variants(alias))
            if aten_name := entry.get("aten_name"):
                interesting_attrs.update(api_attr_variants(aten_name))

        # Scan test directories
        manifest = active_manifest()
        for test_dir in manifest.test_search_dirs:
            dir_path = src / test_dir
            if not dir_path.exists():
                continue

            # Standalone test dirs (outside the import package) are indexed
            # inclusively; package-internal dirs (e.g. torch/testing) hold
            # utilities, so only content-matching files are indexed.
            is_main_test_dir = (
                test_dir.split("/")[0] not in manifest.python_package_roots
            )

            for py_file in dir_path.rglob("*.py"):
                file_str = str(py_file)

                # Skip __pycache__
                if "__pycache__" in file_str:
                    continue

                try:
                    content = py_file.read_text(encoding="utf-8", errors="replace")

                    if not is_main_test_dir and not _has_test_patterns(
                        content, manifest.test_content_patterns
                    ):
                        continue

                    # Parse AST to extract test classes and functions
                    try:
                        tree = ast.parse(content, filename=str(py_file))
                    except SyntaxError:
                        continue

                    rel_path = str(py_file.relative_to(src))
                    file_info = {
                        "path": rel_path,
                        "full_path": file_str,
                        "classes": [],
                        "functions": [],
                        "imports": [],
                    }

                    if _has_ops_decorator(tree):
                        _state.opinfo_test_files.add(rel_path)

                    for node in ast.walk(tree):
                        # Extract test classes
                        if isinstance(node, ast.ClassDef):
                            bases = [_get_ast_name(b) for b in node.bases]
                            is_test_class = any(
                                "TestCase" in b or "TestBase" in b for b in bases
                            )
                            class_info = {
                                "name": node.name,
                                "file": rel_path,
                                "line": node.lineno,
                                "bases": bases,
                                "is_test_class": is_test_class,
                            }
                            file_info["classes"].append(class_info)

                            # Index test classes
                            _state.test_classes.setdefault(node.name, []).append(
                                class_info
                            )

                            # Extract test methods
                            for item in node.body:
                                if isinstance(
                                    item, ast.FunctionDef
                                ) and item.name.startswith("test_"):
                                    func_info = {
                                        "name": item.name,
                                        "class": node.name,
                                        "file": rel_path,
                                        "line": item.lineno,
                                    }
                                    file_info["functions"].append(func_info)
                                    _state.test_functions.setdefault(
                                        item.name, []
                                    ).append(func_info)
                                    if interesting_attrs:
                                        _collect_test_attr_hits(
                                            item,
                                            rel_path,
                                            node.name,
                                            interesting_attrs,
                                            _state.test_attr_index,
                                        )

                        # Extract standalone test functions
                        elif isinstance(node, ast.FunctionDef) and node.name.startswith(
                            "test_"
                        ):
                            if node.col_offset == 0:  # Top-level function
                                func_info = {
                                    "name": node.name,
                                    "class": None,
                                    "file": rel_path,
                                    "line": node.lineno,
                                }
                                file_info["functions"].append(func_info)
                                _state.test_functions.setdefault(node.name, []).append(
                                    func_info
                                )
                                if interesting_attrs:
                                    _collect_test_attr_hits(
                                        node,
                                        rel_path,
                                        None,
                                        interesting_attrs,
                                        _state.test_attr_index,
                                    )

                    _state.test_files[rel_path] = file_info

                except Exception as e:
                    log.debug(f"Error parsing {py_file.name}: {e}")

        # Index test utilities
        for util_path in manifest.test_utility_modules:
            full_path = src / util_path
            if full_path.exists():
                _state.test_utilities[util_path] = {
                    "path": util_path,
                    "full_path": str(full_path),
                    "exists": True,
                }

        log.info(
            f"Test infrastructure: {len(_state.test_files)} files, "
            f"{len(_state.test_classes)} test classes, "
            f"{len(_state.test_functions)} test functions"
        )
        try:
            _save_test_infra_cache(cache)
        except OSError as e:
            log.debug(f"Failed to save test infra cache: {e}")

    except Exception as e:
        log.warning(f"Failed to analyze test infrastructure: {e}")


_TORCH_MODULE_NAMES = {"torch", "F"}


def _classify_rhs(node) -> str | None:
    """Best-effort type tag for an assignment RHS."""
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Set):
        return "set"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, str):
            return "str"
        if isinstance(v, (int, float)):
            return "number"
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in _TORCH_MODULE_NAMES
        ):
            return "tensor"
    return None


def _infer_local_types(func_node) -> dict[str, str]:
    """Map local var name → inferred type within `func_node` body."""
    types: dict[str, str] = {}
    for sub in ast.walk(func_node):
        if not isinstance(sub, ast.Assign):
            continue
        rhs_type = _classify_rhs(sub.value)
        if rhs_type is None:
            continue
        for target in sub.targets:
            if isinstance(target, ast.Name):
                types[target.id] = rhs_type
    return types


def _collect_test_attr_hits(
    func_node,
    file: str,
    class_name: str | None,
    interesting_attrs: set[str],
    index: dict[str, list[dict]],
) -> None:
    """Record API-variant Attribute/Name accesses inside a `test_*` method."""
    local_types = _infer_local_types(func_node)
    for sub in ast.walk(func_node):
        name: str | None = None
        receiver_type: str | None = None
        if isinstance(sub, ast.Attribute):
            name = sub.attr
            if isinstance(sub.value, ast.Name):
                receiver_type = local_types.get(sub.value.id)
        elif isinstance(sub, ast.Name):
            name = sub.id
        if name and name in interesting_attrs:
            index.setdefault(name, []).append(
                {
                    "file": file,
                    "class": class_name,
                    "function": func_node.name,
                    "receiver_type": receiver_type,
                }
            )


def _has_ops_decorator(tree) -> bool:
    """True if any class/function in `tree` is decorated with `@ops(...)`.

    Marks files that consume `op_db` via `instantiate_device_type_tests` and
    `@ops`, the standard PyTorch OpInfo-driven test pattern.
    """
    import ast

    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Name) and target.id == "ops":
                return True
            if isinstance(target, ast.Attribute) and target.attr == "ops":
                return True
    return False


def _get_ast_name(node) -> str:
    """Get string name from AST node."""

    try:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{_get_ast_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return _get_ast_name(node.value)
        elif isinstance(node, ast.Call):
            return _get_ast_name(node.func)
        elif isinstance(node, ast.Constant):
            return str(node.value) if node.value else ""
    except Exception:
        pass
    return ""


_OPINFO_CLASSES = {"OpInfo", "BinaryUfuncInfo", "UnaryUfuncInfo", "ShapeFuncInfo"}


def _is_opinfo_call(node) -> bool:
    """True if `node` is an `OpInfo(...)` / `BinaryUfuncInfo(...)` / etc. call."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _OPINFO_CLASSES
    if isinstance(func, ast.Attribute):
        return func.attr in _OPINFO_CLASSES
    return False


def _opinfo_string_kw(node, kwarg: str) -> str | None:
    """Extract a string kwarg from an OpInfo call, e.g. `aten_name='conv2d'`."""
    for kw in node.keywords:
        if kw.arg == kwarg and isinstance(kw.value, ast.Constant):
            v = kw.value.value
            if isinstance(v, str):
                return v
    return None


def _opinfo_string_tuple_kw(node, kwarg: str) -> list[str]:
    """Extract a string tuple/list kwarg, e.g. `aliases=('conv2d', 'conv2d_v2')`."""
    out: list[str] = []
    for kw in node.keywords:
        if kw.arg != kwarg or not isinstance(kw.value, (ast.Tuple, ast.List)):
            continue
        for elt in kw.value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
    return out


def _parse_opinfo_registry(opinfo_path: str):
    """Parse OpInfo definitions for op name + aliases + aten_name."""
    global _state

    try:
        path = Path(opinfo_path)
        content = path.read_text()
        try:
            tree = ast.parse(content, filename=opinfo_path)
        except SyntaxError:
            return

        if _state.pytorch_source:
            try:
                rel_path = str(path.relative_to(_state.pytorch_source))
            except ValueError:
                rel_path = str(path)
        else:
            rel_path = str(path)

        count = 0
        for node in ast.walk(tree):
            if not _is_opinfo_call(node):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            op_name = node.args[0].value
            if not isinstance(op_name, str):
                continue

            aliases = _opinfo_string_tuple_kw(node, "aliases")
            aten_name = _opinfo_string_kw(node, "aten_name")

            entry = {
                "name": op_name,
                "file": rel_path,
                "line": node.lineno,
                "aliases": aliases,
                "aten_name": aten_name,
            }
            _state.opinfo_registry[op_name] = entry
            for alias in aliases:
                _state.opinfo_alias_map.setdefault(alias, []).append(entry)
            if aten_name:
                _state.opinfo_alias_map.setdefault(aten_name, []).append(entry)
            count += 1

        if count > 0:
            log.debug(f"Found {count} OpInfo definitions in {path.name}")

    except OSError as e:
        log.debug(f"Failed to read OpInfo file {opinfo_path}: {e}")


def _auto_detect_source() -> str | None:
    """Auto-detect the active harness's source using 3-level resolution.

    Priority: env var > config.toml > None.
    CLI flag is handled upstream in run_server().
    """
    return resolve_source()


def update_index(source: str, since: str, on_uncovered: str = "warn") -> dict:
    """Incrementally refresh the bindings index using a prior snapshot as baseline.

    Re-detects only C++/CUDA files that changed in git between the snapshot's
    recorded commit and current HEAD. YAML files are re-parsed when they change.
    The C++ call graph is NOT incrementally updated and may be stale; run
    `torchtalk index build` to rebuild it.

    `on_uncovered` selects the policy for changed headers not in the baseline's
    recorded include graph: warn (default), fail (flag in stats), or widen
    (textually grep compile-DB TUs and add to the reparse set).
    """
    import subprocess
    from dataclasses import asdict

    from .analysis.binding_detector import BindingDetector
    from .snapshots import _relpath, _snapshot_dir, read_manifest

    manifest = read_manifest(since)
    active = active_manifest()
    if manifest.package and manifest.package != active.package:
        raise ValueError(
            f"Snapshot '{since}' was built by the '{manifest.package}' harness "
            f"but '{active.package}' is active. Pass --harness {manifest.package}."
        )
    if not manifest.git_commit:
        raise ValueError(
            f"Snapshot '{since}' has no git_commit; cannot diff against HEAD"
        )

    snap_dir = _snapshot_dir(since)
    prior = json.loads((snap_dir / "bindings.json").read_text())

    # Prior rows are carried forward verbatim; an older payload schema would
    # produce a mixed-format cache that _cache_valid then accepts.
    prior_version = prior.get("metadata", {}).get("format_version")
    if prior_version != _BINDINGS_CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Snapshot '{since}' bindings use cache format v{prior_version}; "
            f"current is v{_BINDINGS_CACHE_FORMAT_VERSION}. Run 'torchtalk "
            "index build' and save a fresh snapshot."
        )

    try:
        diff_out = subprocess.run(
            [
                "git",
                "-C",
                source,
                "diff",
                "--name-status",
                f"{manifest.git_commit}..HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"git diff failed: {e}") from e

    cpp_exts = (".cpp", ".cc", ".cxx", ".cu", ".cuh")
    header_exts = (".h", ".hpp", ".hxx", ".hh", ".inc")
    yaml_files = {
        p for p in (active.native_functions_yaml, active.derivatives_yaml) if p
    }

    changed_cpp: set[str] = set()
    removed_cpp: set[str] = set()
    changed_headers: set[str] = set()
    yaml_changed = False
    for line in diff_out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0], parts[-1]
        if path in yaml_files:
            yaml_changed = True
        if path.endswith(header_exts):
            changed_headers.add(path)
            continue
        if not path.endswith(cpp_exts):
            continue
        (removed_cpp if status.startswith("D") else changed_cpp).add(path)

    dirty = changed_cpp | removed_cpp
    prior_source = manifest.pytorch_source
    new_bindings = [
        b
        for b in prior.get("bindings", [])
        if _relpath(b.get("file_path", ""), prior_source) not in dirty
    ]
    new_kernels = [
        k
        for k in prior.get("cuda_kernels", [])
        if _relpath(k.get("file_path", ""), prior_source) not in dirty
    ]

    detector = BindingDetector(
        macro_aliases=active.cpp_macro_aliases,
        token_map=active.cpp_token_map,
    )
    src = Path(source)
    # Same harness boundary as the full build: search dirs, exclusion
    # patterns, and the binding-pattern prefilter must all match
    # detect_bindings_in_directory, or incremental results diverge.
    bounds = tuple(d.rstrip("/") + "/" for d in active.cpp_search_dirs)
    for rel in changed_cpp:
        if bounds and not rel.startswith(bounds):
            continue
        # Exclusion runs on the repo-relative path: patterns describe repo
        # shapes, and the checkout's own directory name must not match them.
        if _should_exclude(rel, active.exclude_patterns):
            continue
        full = src / rel
        if not full.exists():
            continue
        try:
            content = full.read_text(errors="ignore")
        except OSError:
            continue
        if not _has_binding_patterns(content, active.registration_macros):
            continue
        g = detector.detect_bindings(str(full), content)
        new_bindings.extend(b.to_dict() for b in g.bindings)
        new_kernels.extend(asdict(k) for k in g.cuda_kernels)

    if yaml_changed:
        functions, derivatives = _parse_native_functions(source)
        implementations, symbol_to_file = _find_implementations(source, functions)
    else:
        functions = prior.get("native_functions", {})
        derivatives = prior.get("derivatives", {})
        implementations = prior.get("native_implementations", {})
        symbol_to_file = prior.get("symbol_to_file", {})

    data = {
        "bindings": new_bindings,
        "cuda_kernels": new_kernels,
        "native_functions": functions,
        "derivatives": derivatives,
        "native_implementations": implementations,
        "symbol_to_file": symbol_to_file,
        # Python files are not rescanned on incremental update; carry the
        # baseline's registration records forward.
        "registrations": prior.get(
            "registrations", {"records": [], "candidate_edges": []}
        ),
        "metadata": {
            **_cache_metadata(source),
            "updated_since": since,
            "updated_commit": manifest.git_commit,
        },
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(source)
    with open(cache, "w") as f:
        json.dump(data, f)

    cg_stats = _update_call_graph(
        source, since, changed_cpp, removed_cpp, changed_headers, on_uncovered
    )

    return {
        "cpp_files_changed": len(changed_cpp),
        "cpp_files_removed": len(removed_cpp),
        "headers_changed": len(changed_headers),
        "yaml_changed": yaml_changed,
        "bindings_total": len(new_bindings),
        "cuda_kernels_total": len(new_kernels),
        "baseline_snapshot": since,
        "baseline_commit": manifest.git_commit,
        "call_graph": cg_stats,
    }


def _widen_reparse_set(
    source: Path, uncovered: set[str], cc_index: dict[str, dict]
) -> set[str]:
    """Return repo-relative TUs that textually #include any uncovered header.

    Basename match is deliberately conservative: a header `ops.h` matches every
    `#include ".../ops.h"`, including unrelated same-named files. Over-reparsing
    is preferable to missing a real edge under the `widen` policy.
    """
    import os
    import re
    import subprocess

    compiled_rel: set[str] = set()
    for fp in cc_index:
        try:
            rel = os.path.relpath(fp, str(source))
        except ValueError:
            continue
        if not rel.startswith(".."):
            compiled_rel.add(rel)

    extra: set[str] = set()
    for hdr in uncovered:
        basename = os.path.basename(hdr)
        if not basename:
            continue
        pattern = rf'^\s*#include\s*[<"][^<"]*{re.escape(basename)}[>"]'
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "grep",
                    "-l",
                    "-E",
                    pattern,
                    "--",
                    "*.cpp",
                    "*.cu",
                    "*.mm",
                    "*.cc",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and line in compiled_rel:
                extra.add(line)
    return extra


def _update_call_graph(
    source: str,
    since: str,
    changed_cpp: set[str],
    removed_cpp: set[str],
    changed_headers: set[str],
    on_uncovered: str = "warn",
) -> dict[str, Any]:
    """Incrementally refresh the C++ call graph using a snapshot as baseline.

    Loads the baseline call graph, computes which TUs a header change affects
    via recorded per-TU include sets, evicts per-file records for the union of
    changed cpp + header-affected TUs + removed files, re-parses the dirty
    TUs, and writes the result to the active cache.
    """
    import os

    from .analysis.cpp_call_graph import LIBCLANG_AVAILABLE, CppCallGraphExtractor
    from .snapshots import _snapshot_dir

    if not LIBCLANG_AVAILABLE:
        return {"skipped": "libclang not available"}

    snap_cg = _snapshot_dir(since) / "callgraph.json"
    if not snap_cg.exists():
        return {"skipped": "baseline snapshot has no call graph"}

    cg_cache_dir = CACHE_DIR / "call_graph"
    cache_key = cache_paths(source)["callgraph"].stem

    extractor = CppCallGraphExtractor(cache_dir=cg_cache_dir)
    if not extractor.load_from_path(snap_cg):
        return {"skipped": "failed to load baseline call graph"}

    header_affected = extractor.find_affected_tus(changed_headers)
    tus_to_reparse = set(changed_cpp) | header_affected

    uncovered = changed_headers - extractor.known_headers()

    src = Path(source)
    compile_commands = src / "compile_commands.json"
    if not compile_commands.exists():
        compile_commands = src / "build" / "compile_commands.json"

    cc_index: dict[str, dict] = {}
    need_db = tus_to_reparse or (uncovered and on_uncovered == "widen")
    if compile_commands.exists() and need_db:
        with open(compile_commands) as f:
            compile_db = json.load(f)
        for entry in compile_db:
            fp = entry.get("file", "")
            directory = entry.get("directory", "")
            if not os.path.isabs(fp) and directory:
                fp = os.path.join(directory, fp)
            cc_index[fp] = entry

    widened_net: set[str] = set()
    if uncovered and on_uncovered == "widen" and cc_index:
        widened = _widen_reparse_set(src, uncovered, cc_index)
        widened_net = widened - tus_to_reparse
        tus_to_reparse |= widened

    # Same harness boundary as the full call-graph build.
    active = active_manifest()
    entries: list[tuple[str, list[str]]] = []
    for rel in tus_to_reparse:
        full = str(src / rel)
        if not _should_include_dir(full, list(active.cpp_search_dirs)):
            continue
        if _should_exclude(full, active.exclude_patterns):
            continue
        entry = cc_index.get(full)
        if entry is None:
            continue
        command = entry.get("command", "")
        args = command.split()[1:] if command else entry.get("arguments", [])[1:]
        entries.append((full, args))

    removed_abs = [str(src / r) for r in removed_cpp]

    try:
        stats = extractor.update_files(entries, removed=removed_abs, source_root=source)
    except RuntimeError as e:
        return {"skipped": str(e)}

    stats["header_affected_tus"] = len(header_affected)
    stats["uncovered_headers"] = len(uncovered)
    stats["uncovered_sample"] = sorted(uncovered)[:5]
    stats["on_uncovered"] = on_uncovered
    if widened_net:
        stats["widened_tus"] = len(widened_net)
    if uncovered and on_uncovered == "fail":
        stats["uncovered_fail"] = True

    extractor.save_cache(cache_key, fingerprint=_source_fingerprint(source))
    return stats


def build_index(source: str, wait_for_cpp: bool = True) -> dict:
    """Build or refresh the index for a PyTorch source and return stats.

    Headless equivalent of `mcp-serve` startup. Blocks on the C++ call graph
    when wait_for_cpp is True; otherwise returns while it builds in the
    background.
    """
    _init_from_source(source)

    if wait_for_cpp and _state.cpp_thread is not None and _state.cpp_thread.is_alive():
        log.info("Waiting for C++ call graph build to finish...")
        _state.cpp_thread.join()

    cg_functions = 0
    if _state.cpp_extractor is not None:
        cg_functions = len(_state.cpp_extractor.function_locations)

    return {
        "package": str(_state.package),
        "registration_records": len(_state.registrations.get("records", [])),
        "candidate_edges": len(_state.registrations.get("candidate_edges", [])),
        "bindings": len(_state.bindings),
        "cuda_kernels": len(_state.cuda_kernels),
        "native_functions": len(_state.native_functions),
        "derivatives": len(_state.derivatives),
        "call_graph_functions": cg_functions,
        "call_graph_building": _state.cpp_building,
        "python_modules": len(_state.py_modules),
        "nn_modules": len(_state.nn_modules),
        "test_files": len(_state.test_files),
        "test_functions": len(_state.test_functions),
    }
