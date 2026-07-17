"""Call graph tool implementations."""

from __future__ import annotations

import os

from ..analysis.helpers import dedupe_by_key
from ..formatting import create_formatter
from ..indexer import _cpp_status, _ensure_loaded, _state
from .common import _rel_path, _with_note

# Hard ceiling on impact-walk depth. Power users can raise the soft default
# via the TORCHTALK_GRAPH_MAX_DEPTH env var, but never above this cap —
# unbounded walks routinely traverse all of ATen and exhaust MCP timeouts.
_GRAPH_HARD_DEPTH_CAP = 10


def _max_depth() -> int:
    raw = os.environ.get("TORCHTALK_GRAPH_MAX_DEPTH")
    if not raw:
        return 5
    try:
        return max(1, min(int(raw), _GRAPH_HARD_DEPTH_CAP))
    except ValueError:
        return 5


def _clamp_depth(depth: int) -> int:
    return min(max(depth, 1), _max_depth())


def _py_name_to_cpp_symbol(py_name: str) -> str:
    """Convert a binding's dotted python_name to py_to_cpp_edges key form.

    `aten.add` → `aten::add`; `aten.add.Tensor` → `aten::add` (overload tag
    drops). Bare names pass through.
    """
    parts = py_name.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}::{parts[1]}"
    return py_name


def _python_callers_for(cpp_func: str) -> list[dict]:
    """Look up Python source callers of `cpp_func` via the M1 edge index.

    Tries known bindings first (binding's python_name → cpp_symbol form),
    then falls back to bare-name guesses (`aten::<bare>`, `<bare>`).
    """
    bare = cpp_func.rsplit("::", 1)[-1]
    edges = _state.py_to_cpp_edges
    if not edges:
        return []
    seen_keys: set[str] = set()
    out: list[dict] = []
    for binding in _state.by_cpp_name.get(bare, []):
        if py_name := binding.get("python_name"):
            key = _py_name_to_cpp_symbol(py_name)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.extend(edges.get(key, []))
    for key in (f"aten::{bare}", bare):
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.extend(edges.get(key, []))
    return out


def _cuda_gap_note() -> str:
    cu = _state.cpp_extractor.coverage_summary().get("cu_unindexed", 0)
    return f"{cu:,} .cu TUs unindexed — CUDA callers unknown." if cu else ""


def _cuda_adjacent(function_name: str, items: list[dict], file_key: str) -> bool:
    if "cuda" in function_name.lower():
        return True
    return any(
        (it.get(file_key) or "").lower().endswith((".cu", ".cuh"))
        or "cuda" in (it.get(file_key) or "").lower()
        for it in items
    )


def _resolve_function(
    function_name: str, relation: str = "callers"
) -> tuple[str | None, str]:
    """Resolve a query to ONE call-graph symbol, disclosing fuzzy resolution.

    Ranked candidates come from match_functions; the first one with any
    relations in the requested direction wins (deterministic). Only that
    symbol is queried — candidates are listed in the note instead of
    silently merging every match's neighborhood.
    """
    ext = _state.cpp_extractor
    matches = ext.match_functions(function_name, fuzzy=True)
    if not matches:
        return None, ""
    getter = ext.get_callees if relation == "callees" else ext.get_callers
    resolved = next((m for m in matches if getter(m, fuzzy=False)), matches[0])
    if resolved == function_name:
        return resolved, ""
    note = f"Resolved '{function_name}' → '{resolved}' (fuzzy)."
    others = [m for m in matches if m != resolved]
    if others:
        shown = ", ".join(f"`{m}`" for m in others[:8])
        more = f", +{len(others) - 8} more" if len(others) > 8 else ""
        note += f" Other matches: {shown}{more}."
    return resolved, note


def _format_call_item(md, item: dict, name_key: str, file_key: str, line_key: str):
    name = item[name_key]
    if file_path := item.get(file_key):
        line = f":{item[line_key]}" if item.get(line_key) else ""
        md.item(f"`{name}` \u2192 `{_rel_path(file_path)}{line}`")
    else:
        md.item(f"`{name}`")


async def _do_calls(function_name: str) -> str:
    _ensure_loaded()
    if not function_name.strip():
        return "Provide a function name."
    if status := _cpp_status():
        return status

    resolved, note = _resolve_function(function_name, relation="callees")
    callees = (
        _state.cpp_extractor.get_callees(resolved, fuzzy=False) if resolved else []
    )
    if not callees:
        msg = f"No outbound calls found for '{function_name}'."
        if note:
            msg = f"{msg}\n\n{note}"
        return _with_note(msg)

    results = dedupe_by_key(callees, "callee")

    md = create_formatter()
    md.h2(f"Calls: `{function_name}`")
    if note:
        md.text(note)
    md.text("*Functions this calls (outbound dependencies):*\n")

    for item in results[:15]:
        _format_call_item(md, item, "callee", "callee_file", "callee_line")

    if len(results) > 15:
        md.text(f"\n*Showing 15 of {len(results)} calls.*")

    return _with_note(md.build())


async def _do_called_by(function_name: str) -> str:
    _ensure_loaded()
    if not function_name.strip():
        return "Provide a function name."
    if status := _cpp_status():
        return status

    resolved, note = _resolve_function(function_name)
    callers = (
        _state.cpp_extractor.get_callers(resolved, fuzzy=False) if resolved else []
    )
    if not callers:
        msg = f"No inbound callers found for '{function_name}'."
        if note:
            msg = f"{msg}\n\n{note}"
        if gap := _cuda_gap_note():
            msg = f"{msg}\n\n{gap}"
        return _with_note(msg)

    results = dedupe_by_key(callers, "caller")

    md = create_formatter()
    md.h2(f"Called by: `{function_name}`")
    if note:
        md.text(note)
    md.text("*Functions that call this (inbound dependents):*\n")

    for item in results[:15]:
        _format_call_item(md, item, "caller", "caller_file", "caller_line")

    if len(results) > 15:
        md.text(f"\n*Showing 15 of {len(results)} callers.*")

    if _cuda_adjacent(function_name, results, "caller_file") and (
        gap := _cuda_gap_note()
    ):
        md.blank()
        md.text(gap)

    return _with_note(md.build())


async def _do_impact(
    function_name: str,
    depth: int = 2,
    focus: str = "callers",
    fuzzy_all_levels: bool = False,
    walk_python: bool = False,
) -> str:
    _ensure_loaded()
    if not function_name.strip():
        return "Provide a function name."
    if status := _cpp_status():
        return status

    depth = _clamp_depth(depth)

    resolved, note = _resolve_function(function_name)
    if resolved is None:
        return _with_note(f"No callers found for '{function_name}'.")

    visited = set()
    current_level = {resolved}
    callers_by_depth: dict[int, list[dict]] = {}

    for level in range(1, depth + 1):
        next_level = set()
        level_callers = []

        for func in current_level:
            # Level 1 fuzziness is handled by the up-front resolution.
            fuzzy = fuzzy_all_levels and level > 1
            for item in _state.cpp_extractor.get_callers(func, fuzzy=fuzzy):
                caller = item["caller"]
                if caller not in visited and caller not in (function_name, resolved):
                    visited.add(caller)
                    next_level.add(caller)
                    level_callers.append(item)

        if level_callers:
            callers_by_depth[level] = level_callers
        current_level = next_level
        if not current_level:
            break

    if not callers_by_depth:
        msg = f"No callers found for '{function_name}'."
        if note:
            msg = f"{msg}\n\n{note}"
        return _with_note(msg)

    md = create_formatter()
    md.h2(f"Impact Analysis: `{function_name}`")
    if note:
        md.text(note)
    md.text(f"*Tracing callers up to {depth} levels deep*\n")

    total = 0
    for level, callers in callers_by_depth.items():
        unique = dedupe_by_key(callers, "caller")
        total += len(unique)
        md.h3(f"Depth {level} ({len(unique)} callers)")

        for item in unique[:15]:
            _format_call_item(md, item, "caller", "caller_file", "caller_line")

        if len(unique) > 15:
            md.item(f"*... and {len(unique) - 15} more*")
        md.blank()

    # Both Python-facing sections must include the seed itself — a query on
    # `at::native::gelu` has entry points/py-callers even with zero C++ callers.
    impact_set = visited | {function_name, resolved}

    if focus == "full":
        python_entries = [
            {
                "python": b.get("python_name", c),
                "cpp": c,
                "dispatch": b.get("dispatch_key", ""),
            }
            for c in impact_set
            if c in _state.by_cpp_name
            for b in _state.by_cpp_name[c][:1]
        ]

        if python_entries:
            md.h3(f"Python Entry Points ({len(python_entries)} found)")
            for entry in python_entries[:10]:
                dispatch = f" [{entry['dispatch']}]" if entry["dispatch"] else ""
                md.item(f"`{entry['python']}`{dispatch} → `{entry['cpp']}`")
            if len(python_entries) > 10:
                md.item(f"*... and {len(python_entries) - 10} more*")

    if walk_python:
        # Source-level Python callers via the M1 edge index — catches pure-
        # Python wrappers that don't go through a registered binding.
        seen_callers: set[tuple[str, str, int]] = set()
        py_callers: list[dict] = []
        for cpp in impact_set:
            for hit in _python_callers_for(cpp):
                key = (hit["caller_qualname"], hit["file"], hit["line"])
                if key in seen_callers:
                    continue
                seen_callers.add(key)
                py_callers.append({**hit, "via": cpp})
        if py_callers:
            md.h3(f"Python Source Callers ({len(py_callers)} found)")
            for hit in py_callers[:15]:
                path = _rel_path(hit["file"])
                md.item(
                    f"`{hit['caller_qualname']}` → "
                    f"`{path}:{hit['line']}` (via `{hit['via']}`)"
                )
            if len(py_callers) > 15:
                md.item(f"*... and {len(py_callers) - 15} more*")
            md.blank()

    md.text(f"Total impact: {total} functions across {len(callers_by_depth)} levels")

    return _with_note(md.build())
