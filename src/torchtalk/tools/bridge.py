"""Bridge tool: cross-package references resolved against a dependency index.

Refs are collected at index time (analysis/external_refs.py). This module
answers both directions: which dependency symbols a function of this
package reaches (`uses`), and which functions of this package reach a
dependency symbol (`used_by`). Targets resolve to file:line through the
dependency's own index when `dependency_index` can load it.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from ..analysis.external_refs import bridge_manifests, op_namespace_targets
from ..formatting import create_formatter, relative_path
from ..harness import active_manifest
from ..indexer import ServerState, _ensure_loaded, _state, dependency_index
from ..indexer import dependency_status as _dependency_status
from .common import _python_symbol, _rel_path, _with_note


def refs_from(symbols: Iterable[str]) -> list[dict]:
    """Refs whose source is one of `symbols`.

    A qualified Python name matches exactly; a bare C++ name matches any
    call graph symbol with that final `::` component.
    """
    wanted = set(symbols)
    return [
        r
        for r in _state.external_refs
        if r["from_symbol"] in wanted or r["from_symbol"].rsplit("::", 1)[-1] in wanted
    ]


def refs_to(name: str) -> list[dict]:
    """Refs targeting `name`.

    Accepts `aten::silu`, `aten.silu`, the Python form `torch.silu` (via the
    alias map), `at::silu` and a bare `silu`, which is tried under every
    dependency op namespace and `[bridge] cpp_namespaces`. A module or C++
    scope prefix matches everything beneath it (`torch.nn`, `at::Tensor`).
    """
    manifest = active_manifest()
    names = {name}
    if "::" not in name and "." in name:
        names.add(name.replace(".", "::"))
    if alias := _state.alias_map.get(name):
        names.add(alias)
    if "::" not in name and "." not in name:
        names.update(f"{ns}::{name}" for ns in op_namespace_targets(manifest))
        names.update(f"{ns}::{name}" for ns in manifest.cpp_namespaces)
    prefixes = (f"{name}.", f"{name}::")
    return [
        r
        for r in _state.external_refs
        if r["to_name"] in names or r["to_name"].startswith(prefixes)
    ]


def resolve(ref: dict, dep: ServerState | None = None) -> list[dict]:
    """Definitions of `ref["to_name"]` in the dependency index.

    Each item is `{symbol, file, line}`. An op resolves through the
    dependency's native functions (then its registrations); a C++ symbol
    through its call graph locations, or as an op when its namespace is in
    `[bridge] cpp_op_namespaces` (`at::silu` → `aten::silu`). Imports are
    not resolved: the dependency's Python modules are not cached. Empty when
    the dependency index is unavailable or does not know the name. Callers
    resolving many refs pass the index in rather than looking it up per ref.
    """
    if dep is None:
        dep = dependency_index(ref["to_package"])
    if dep is None:
        return []
    name = ref["to_name"]
    if ref["kind"] == "op":
        return _op_locations(dep, name, ref["to_package"])
    if ref["kind"] == "cpp":
        if loc := dep.cpp_locations.get(name):
            return [{"symbol": name, "file": loc[0], "line": loc[1]}]
        ns, _, bare = name.rpartition("::")
        if op_ns := active_manifest().cpp_op_namespaces.get(ns):
            return _op_locations(dep, f"{op_ns}::{bare}", ref["to_package"])
    return []


def _op_locations(dep: ServerState, symbol: str, pkg: str) -> list[dict]:
    """Where the dependency implements op `ns::name`.

    Dispatch kernels come first (a structured op's kernels sit on its out=
    delegate, so those lead), then implementations named after the op, then
    registration bindings. Kernels the index has no body for are listed
    together at the YAML that declares them, with no line.
    """
    ns, _, op = symbol.rpartition("::")
    out: list[dict] = []
    seen: set[tuple] = set()

    def add(name: str, file: str, line) -> None:
        if file and (name, file, line) not in seen:
            seen.add((name, file, line))
            out.append({"symbol": name, "file": file, "line": line})

    native = dep.native_functions.get(op)
    kernels: list[str] = []
    if native:
        delegate = dep.native_functions.get(native.get("structured_delegate") or "", {})
        for fn in (
            *delegate.get("dispatch", {}).values(),
            *native.get("dispatch", {}).values(),
        ):
            if fn not in kernels:
                kernels.append(fn)
        for fn in (*kernels, op):
            for impl in dep.native_implementations.get(fn, []):
                add(
                    impl["function_name"],
                    impl.get("file_path"),
                    impl.get("line_number"),
                )
    if not out:
        for b in dep.by_python_name.get(f"{ns}.{op}", []):
            add(b.get("cpp_name") or symbol, b.get("file_path"), b.get("line_number"))
    if not out and kernels:
        yaml = bridge_manifests(active_manifest())[pkg].native_functions_yaml
        add(", ".join(kernels), f"{dep.source}/{yaml}" if yaml else "", None)
    return out


def _unique_targets(refs: list[dict]) -> list[dict]:
    """One ref per (from_symbol, to_name), first evidence wins, in a fixed order."""
    seen: set[tuple[str, str]] = set()
    out = []
    for r in refs:
        key = (r["from_symbol"], r["to_name"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return sorted(out, key=lambda r: (r["from_symbol"], r["evidence"], r["to_name"]))


def uses_section(md, refs: list[dict], limit: int = 20) -> None:
    """Render `refs` grouped by dependency, then by source symbol and evidence."""
    own_locations = getattr(_state.cpp_extractor, "function_locations", {})
    for pkg in sorted({r["to_package"] for r in refs}):
        md.h3(f"Calls into {pkg}")
        dep = dependency_index(pkg)
        if status := _dependency_status(pkg):
            md.text(f"*Targets unresolved: {status}.*")
        rows = _unique_targets([r for r in refs if r["to_package"] == pkg])
        source = None
        for r in rows[:limit]:
            if (r["from_symbol"], r["evidence"]) != source:
                source = (r["from_symbol"], r["evidence"])
                md.item(f"`{r['from_symbol']}` (`{r['evidence']}`)")
            line = f"`{r['to_name']}`"
            locs = resolve(r, dep)
            if locs:
                # Name the kernel only when it is not simply the op's own name.
                symbol = locs[0]["symbol"]
                own = (r["to_name"], r["to_name"].rsplit("::", 1)[-1])
                name = f"`{symbol}` " if symbol not in own else ""
                more = f" (+{len(locs) - 1} more)" if len(locs) > 1 else ""
                line += f" → {name}`{_dep_location(dep, locs[0])}`{more}"
            elif r["kind"] == "cpp" and (loc := own_locations.get(r["to_name"])):
                line += f" (declared `{_installed_path(loc[0])}:{loc[1]}`)"
            elif dep is not None and r["kind"] != "import":
                line += " (not in index)"
            md.item(line, 1)
        if len(rows) > limit:
            md.item(f"*... and {len(rows) - limit} more*")
        md.blank()


def _dep_location(dep: ServerState | None, loc: dict) -> str:
    path = relative_path(loc["file"], dep.source if dep else None)
    return f"{path}:{loc['line']}" if loc.get("line") else path


def _installed_path(path: str) -> str:
    """A header path from the environment, shortened to its package part."""
    parts = Path(path).parts
    for marker in ("site-packages", "dist-packages"):
        if marker in parts:
            return "/".join(parts[parts.index(marker) + 1 :])
    return _rel_path(path)


def refs_for(symbol: str) -> list[dict]:
    """Refs from `symbol`: a C++ or qualified name, else a Python symbol query."""
    return refs_from([symbol]) or refs_from(_python_sources(symbol))


def _python_sources(symbol: str) -> list[str]:
    """Qualified names a Python symbol query stands for.

    A class stands for itself and its methods; a bare name that several
    definitions share stands for all of them.
    """
    found = _python_symbol(symbol, "")
    if found is None and "." in symbol:
        # Methods are indexed under their class, not by name.
        owner, _, method = symbol.rpartition(".")
        methods = getattr(_python_symbol(owner, ""), "methods", [])
        found = next((m for m in methods if m.name == method), None)
    candidates = [found] if found else []
    if not candidates and "." not in symbol:
        # A bare name shared by several definitions: answer for all of them.
        candidates = _state.py_functions.get(symbol, []) + _state.py_classes.get(
            symbol, []
        )
    out = []
    for sym in candidates:
        out.append(sym.qualified_name)
        out.extend(m.qualified_name for m in getattr(sym, "methods", []))
    return out


async def _do_bridge(
    symbol: str, mode: Literal["uses", "used_by"] = "uses", limit: int = 20
) -> str:
    _ensure_loaded()
    if not symbol.strip():
        return "Provide a symbol."
    manifest = active_manifest()
    if not manifest.depends_on:
        return f"`{manifest.package}` declares no `depends_on`; nothing to bridge."
    limit = max(1, limit)
    md = create_formatter()

    if mode == "used_by":
        refs = refs_to(symbol)
        if not refs:
            return f"No references to `{symbol}` from `{manifest.package}`."
        md.h2(f"Used by: `{symbol}`")
        rows = _unique_targets(refs)
        md.text(f"*{manifest.package} symbols referencing it ({len(rows)}):*\n")
        for r in rows[:limit]:
            where = f"`{r['evidence']}`, {r['kind']}"
            md.item(f"`{r['from_symbol']}` → `{r['to_name']}` ({where})")
        if len(rows) > limit:
            md.text(f"\n*Showing {limit} of {len(rows)}.*")
        return _with_note(md.build())

    refs = refs_for(symbol)
    if not refs:
        return f"No references from `{symbol}` into {', '.join(manifest.depends_on)}."
    md.h2(f"Uses: `{symbol}`")
    uses_section(md, refs, limit=limit)
    return _with_note(md.build())
