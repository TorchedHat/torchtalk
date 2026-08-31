"""Cross-package references: the single edge primitive for the bridge.

An `ExternalRef` records one place where a symbol in the active package
names something that lives in another package (a dependency listed in the
manifest's `depends_on`). The bridge (phase C) resolves these against the
target package's symbol table; this module only *collects* them.

Kinds (see docs/bridge-design.md):
  import      module-level `import torch.nn` / `from torch import nn`
  op          `torch.ops.aten.X` / `torch.X` op reference (resolver: op_namespaces)
  cpp         `at::X` / `c10::X` C++ symbol (resolver: cpp_namespaces)
  base_class  `class Foo(torch.nn.Module)` (resolver: base_class_namespaces)
  provides    registration flipped: this package *defines* `to_name`
  version_pin package-level pin from requirements/pyproject

PR-4 ships `import` edges only; the other kinds are collected in C2.
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


def bridge_targets(manifest: ConventionManifest) -> dict[str, tuple[str, ...]]:
    """Map each `depends_on` harness name → its Python package roots.

    Uses the registered harness when one is active for that name, else the
    shipped TOML profile. Unknown names are skipped (the manifest check in
    B6 reports them); the bridge is best-effort by design.
    """
    targets: dict[str, tuple[str, ...]] = {}
    for dep in manifest.depends_on:
        try:
            dep_manifest = get_harness(dep).manifest
        except KeyError:
            try:
                dep_manifest = load_builtin_manifest(dep)
            except ManifestError:
                continue
        roots = tuple(dep_manifest.python_package_roots)
        if roots:
            targets[dep] = roots
    return targets


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
