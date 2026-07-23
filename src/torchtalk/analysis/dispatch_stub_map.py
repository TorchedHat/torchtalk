"""Map kernel-impl C++ symbols to the ATen op they implement.

PyTorch's CPU/CUDA kernels live behind a stub-and-impl indirection:

    DECLARE_DISPATCH(fn_t, hardsigmoid_stub)        // header
    DEFINE_DISPATCH(hardsigmoid_stub)                // .cpp
    REGISTER_DISPATCH(hardsigmoid_stub, &hardsigmoid_kernel)  // <arch>/Kernel.cpp

When `hardsigmoid_kernel` changes, our walker has no binding to land on (the
only TORCH_LIBRARY_IMPL entry uses the stub via dispatch). This module
scrapes `REGISTER_*` macros to build `kernel_impl_name → ATen op name`, so
`_bindings_for` can resolve the kernel directly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .binding_detector import _CPP_CONTROL_KEYWORDS

log = logging.getLogger(__name__)

# Matches REGISTER_DISPATCH, REGISTER_AVX512, REGISTER_CUDA_DISPATCH,
# ALSO_REGISTER_AVX512_DISPATCH, REGISTER_NO_AVX2_DISPATCH, etc.
_REGISTER_RE = re.compile(r"\b(?:ALSO_)?REGISTER_\w+\s*\(\s*(\w+)\s*,\s*&?\s*(\w+)")

_SCAN_DIRS = ("cpu", "cuda", "quantized/cpu", "quantized/cuda")
_SCAN_EXTS = ("*.cpp", "*.cu", "*.h")
_STUB_SUFFIXES = ("_stub", "_kernel_impl", "_kernel")


def _stub_to_op(stub: str, nf_keys: set[str]) -> str | None:
    """Resolve a stub name to an ATen op by trying suffix-stripped candidates.

    Stubs like `softmax_lastdim_kernel` peel multiple `_`-segments before
    matching `softmax`. Tries longest→shortest until one is in
    native_functions.
    """
    candidates: list[str] = [stub]
    for suffix in _STUB_SUFFIXES:
        if stub.endswith(suffix):
            candidates.append(stub[: -len(suffix)])
    parts = stub.split("_")
    for i in range(len(parts) - 1, 0, -1):
        candidates.append("_".join(parts[:i]))
    for c in candidates:
        if c in nf_keys:
            return c
    return None


def extract_kernel_impl_to_op(
    source: Path, native_functions: dict[str, dict] | None
) -> dict[str, str]:
    """Build kernel-impl → ATen op map by scraping REGISTER_* macros."""
    if not native_functions:
        return {}
    nf_keys = set(native_functions)
    native_root = source / "aten" / "src" / "ATen" / "native"
    if not native_root.exists():
        return {}

    impl_to_op: dict[str, str] = {}
    for op_name, entry in native_functions.items():
        for impl in entry.get("dispatch", {}).values():
            if impl and impl != op_name:
                impl_to_op.setdefault(impl, op_name)

    kernel_to_op: dict[str, str] = {}
    unresolved: dict[str, list[str]] = {}
    registers_by_file: dict[Path, list[str]] = {}
    for d in _SCAN_DIRS:
        sub = native_root / d
        if not sub.exists():
            continue
        for ext in _SCAN_EXTS:
            for path in sub.rglob(ext):
                try:
                    content = path.read_text(errors="replace")
                except OSError as e:
                    log.debug(f"Skipping {path}: {e}")
                    continue
                for stub, kernel in _REGISTER_RE.findall(content):
                    registers_by_file.setdefault(path, []).append(kernel)
                    if op := _stub_to_op(stub, nf_keys):
                        kernel_to_op[kernel] = op
                    else:
                        unresolved.setdefault(stub, []).append(kernel)
    for stub, op in _stubs_via_call_sites(
        native_root, set(unresolved), nf_keys, impl_to_op
    ):
        for kernel in unresolved[stub]:
            kernel_to_op.setdefault(kernel, op)
    _map_kernel_tu_helpers(registers_by_file, kernel_to_op)
    return kernel_to_op


_FUNC_DEF_RE = re.compile(r"(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{")
_FILE_OP_CAP = 4


def _map_kernel_tu_helpers(
    registers_by_file: dict[Path, list[str]], kernel_to_op: dict[str, str]
) -> None:
    """Map helper functions in a kernel TU to the TU's registered op(s).

    Kernel TUs are compiled as generated build/ copies the call graph
    excludes, so helpers like `cpu_flash_attention` have no graph presence.
    When a TU's registrations resolve to few ops, its other definitions
    belong to those ops; `backward` helpers pair with `backward` ops.
    """
    for path, kernels in registers_by_file.items():
        ops = {kernel_to_op[k] for k in kernels if k in kernel_to_op}
        if not ops or len(ops) > _FILE_OP_CAP:
            continue
        fwd = sorted(o for o in ops if "backward" not in o)
        bwd = sorted(o for o in ops if "backward" in o)
        try:
            content = path.read_text(errors="replace")
        except OSError:
            continue
        for d in _FUNC_DEF_RE.finditer(content):
            name = d.group(1)
            if name in _CPP_CONTROL_KEYWORDS or name in kernel_to_op:
                continue
            pick = (bwd or fwd) if "backward" in name else (fwd or bwd)
            kernel_to_op[name] = pick[0]


def _stubs_via_call_sites(
    native_root: Path,
    stubs: set[str],
    nf_keys: set[str],
    impl_to_op: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Resolve name-mangled stubs to the YAML op whose body invokes them.

    `_stub_to_op("flash_attention_kernel")` strips to `flash_attention`, which
    is not an op — but the stub is called inside an op (or op dispatch-impl)
    body in attention.cpp. Scan non-arch native sources for `stub(` call
    sites; a direct native_functions enclosing name wins over one resolved
    through the YAML dispatch table.
    """
    if not stubs:
        return []
    call_re = re.compile(
        r"\b(" + "|".join(re.escape(s) for s in sorted(stubs)) + r")\s*\("
    )
    skip_prefixes = tuple(str(native_root / d) for d in _SCAN_DIRS)
    direct: dict[str, str] = {}
    via_impl: dict[str, str] = {}
    for path in native_root.rglob("*.cpp"):
        if str(path).startswith(skip_prefixes):
            continue
        try:
            content = path.read_text(errors="replace")
        except OSError:
            continue
        for m in call_re.finditer(content):
            stub = m.group(1)
            if stub in direct:
                continue
            enclosing = None
            for d in _FUNC_DEF_RE.finditer(content, 0, m.start()):
                if d.group(1) not in _CPP_CONTROL_KEYWORDS:
                    enclosing = d.group(1)
            if enclosing in nf_keys:
                direct.setdefault(stub, enclosing)
            elif impl_to_op and enclosing in impl_to_op:
                via_impl.setdefault(stub, impl_to_op[enclosing])
    return list({**via_impl, **direct}.items())
