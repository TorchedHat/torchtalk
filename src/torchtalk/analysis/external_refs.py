"""Cross-package references: the single edge primitive for the bridge.

An `ExternalRef` records one place where a symbol in the active package
names something that lives in another package (a dependency listed in the
manifest's `depends_on`). `tools/bridge.py` resolves these against the
target package's index; this module only *collects* them.

Kinds (see docs/bridge-design.md):
  import      module-level `import torch.nn` / `from torch import nn`
  op          `torch.ops.aten.X` / `torch.X` op reference (resolver: op_namespaces)
  cpp         `at::X` / `c10::X` C++ symbol (resolver: cpp_namespaces)
  base_class  `class Foo(torch.nn.Module)` (resolver: base_class_namespaces)
  provides    registration flipped: this package *defines* `to_name`
  version_pin package-level pin from requirements/pyproject

`import`, `op` and `cpp` edges are collected here; the remaining kinds are
still roadmap (docs/bridge-design.md).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torchtalk.harness import (
    ConventionManifest,
    ManifestError,
    get_harness,
    load_builtin_manifest,
)

REF_KINDS = ("import", "op", "cpp", "base_class", "provides", "version_pin")


@dataclass(frozen=True)
class ExternalRef:
    """One outgoing reference from this package into a dependency."""

    from_symbol: str  # qualified symbol in this package (module name for imports)
    to_name: str  # name as written, e.g. "torch.nn.Module"
    kind: str  # one of REF_KINDS
    evidence: str  # "path:line"
    confidence: float = 1.0
    to_package: str = ""  # harness name the ref should resolve against, if known

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bridge_manifests(manifest: ConventionManifest) -> dict[str, ConventionManifest]:
    """Map each `depends_on` harness name → its manifest.

    Uses the registered harness when one is active for that name, else the
    shipped TOML profile. Unknown names are skipped; the bridge is
    best-effort by design.
    """
    out: dict[str, ConventionManifest] = {}
    for dep in manifest.depends_on:
        try:
            out[dep] = get_harness(dep).manifest
        except KeyError:
            try:
                out[dep] = load_builtin_manifest(dep)
            except ManifestError:
                continue
    return out


def bridge_targets(manifest: ConventionManifest) -> dict[str, tuple[str, ...]]:
    """Map each `depends_on` harness name → its Python package roots."""
    return {
        dep: tuple(m.python_package_roots)
        for dep, m in bridge_manifests(manifest).items()
        if m.python_package_roots
    }


def op_namespace_targets(manifest: ConventionManifest) -> dict[str, str]:
    """Map each C++ op namespace a dependency defines → that dependency.

    Read from the dependencies' own `[python.op_namespaces]` values, so
    `aten` → "pytorch" without the extension naming it.
    """
    out: dict[str, str] = {}
    for dep, m in bridge_manifests(manifest).items():
        for ns in m.op_namespaces.values():
            out.setdefault(ns, dep)
    return out


def _package_for(name: str, targets: dict[str, tuple[str, ...]]) -> str | None:
    head = name.split(".", 1)[0]
    for pkg, roots in targets.items():
        if head in roots:
            return pkg
    return None


def _import_target(imp: Any) -> str | None:
    """Dotted name an import statement refers to; None for relative imports."""
    module = imp.module or ""
    if not module:
        return None  # relative import (`from . import x`) — package-internal
    if imp.name and imp.name != module and imp.name != "*":
        return f"{module}.{imp.name}"
    return module


def collect_import_refs(
    modules: dict[str, Any],
    manifest: ConventionManifest,
    targets: dict[str, tuple[str, ...]] | None = None,
    source: str | None = None,
) -> list[ExternalRef]:
    """Module-level import edges from `modules` into `depends_on` packages.

    Each `PyImport` whose top-level package is a dependency root becomes one
    `ExternalRef(kind="import")`. Imports of the active package itself and
    of third parties not in `depends_on` are ignored. Deterministic order:
    by module name, then line, then target.
    """
    if targets is None:
        targets = bridge_targets(manifest)
    if not targets:
        return []
    own_roots = set(manifest.python_package_roots)
    root = Path(source).resolve() if source else None
    refs: list[ExternalRef] = []
    for mod_name in sorted(modules):
        mod = modules[mod_name]
        seen: set[tuple[str, int]] = set()
        for imp in getattr(mod, "imports", ()):
            target = _import_target(imp)
            if target is None or target.split(".", 1)[0] in own_roots:
                continue
            pkg = _package_for(target, targets)
            if pkg is None:
                continue
            key = (target, imp.line_number)
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                ExternalRef(
                    from_symbol=mod_name,
                    to_name=target,
                    kind="import",
                    evidence=f"{_rel(imp.file_path, root)}:{imp.line_number}",
                    to_package=pkg,
                )
            )
    refs.sort(key=lambda r: (r.from_symbol, r.evidence, r.to_name))
    return refs


def collect_op_refs(
    edges: dict[str, list[dict]],
    manifest: ConventionManifest,
    targets: dict[str, str] | None = None,
    source: str | None = None,
) -> list[ExternalRef]:
    """Python call sites that reach an op a dependency defines.

    `edges` is the `cpp_symbol → caller sites` index (`aten::silu` →
    `[{caller_qualname, file, line}]`). A symbol whose namespace belongs to
    a dependency (`op_namespace_targets`) becomes one `ExternalRef(kind="op")`
    per call site. Ops in this package's own namespaces and bare pybind
    names are ignored.
    """
    if targets is None:
        targets = op_namespace_targets(manifest)
    if not targets:
        return []
    root = Path(source).resolve() if source else None
    refs: list[ExternalRef] = []
    for symbol in sorted(edges):
        ns, sep, _ = symbol.rpartition("::")
        pkg = targets.get(ns) if sep else None
        if pkg is None:
            continue
        seen: set[tuple[str, int]] = set()
        for site in edges[symbol]:
            key = (site["caller_qualname"], site["line"])
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                ExternalRef(
                    from_symbol=site["caller_qualname"],
                    to_name=symbol,
                    kind="op",
                    evidence=f"{_rel(site['file'], root)}:{site['line']}",
                    to_package=pkg,
                )
            )
    refs.sort(key=lambda r: (r.from_symbol, r.evidence, r.to_name))
    return refs


def collect_cpp_refs(
    callees: dict[str, Iterable[str]],
    locations: dict[str, tuple[str, int]],
    manifest: ConventionManifest,
    source: str | None = None,
) -> list[ExternalRef]:
    """C++ call edges into the namespaces listed in `[bridge] cpp_namespaces`.

    `callees` and `locations` are the call graph's caller → callee sets and
    function → (file, line) map. Only callers defined under `source` count,
    so inline functions pulled in from the dependency's own headers do not
    appear as this package's refs. The evidence is the caller's definition:
    the call graph keeps no call-site lines. Refs resolve against the first
    `depends_on` entry, the package that owns those namespaces.
    """
    namespaces = set(manifest.cpp_namespaces)
    if not namespaces or not manifest.depends_on:
        return []
    pkg = manifest.depends_on[0]
    root = Path(source).resolve() if source else None
    # The call graph may spell the checkout with or without symlinks resolved.
    roots = {root, Path(source).absolute()} if root else set()
    prefixes = tuple(f"{r}/" for r in roots)
    refs: list[ExternalRef] = []
    for caller in sorted(callees):
        loc = locations.get(caller)
        if loc is None or (prefixes and not loc[0].startswith(prefixes)):
            continue
        evidence = f"{_rel(loc[0], root)}:{loc[1]}"
        for callee in sorted(set(callees[caller])):
            ns, sep, _ = callee.partition("::")
            if not sep or ns not in namespaces:
                continue
            refs.append(
                ExternalRef(
                    from_symbol=caller,
                    to_name=callee,
                    kind="cpp",
                    evidence=evidence,
                    to_package=pkg,
                )
            )
    return refs


def _rel(path: str, root: Path | None) -> str:
    """Evidence paths are repo-relative so they survive moving the checkout."""
    if root is None:
        return path
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return path


def refs_by_target(refs: Iterable[ExternalRef]) -> dict[str, list[ExternalRef]]:
    """Group refs by `to_name` — the shape the bridge resolver consumes."""
    out: dict[str, list[ExternalRef]] = {}
    for r in refs:
        out.setdefault(r.to_name, []).append(r)
    return out
