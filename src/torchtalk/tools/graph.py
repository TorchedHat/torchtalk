"""Call graph tool implementations."""

from __future__ import annotations

import os

from ..analysis.helpers import dedupe_by_key
from ..formatting import coverage_note, create_formatter, relative_path
from ..indexer import _cpp_status, _ensure_loaded, _state

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


def _rel_path(path: str) -> str:
    return relative_path(path, _state.pytorch_source)


def _with_note(text: str) -> str:
    note = coverage_note(_state.cpp_extractor)
    return f"{text}\n\n{note}" if note else text


def _format_call_item(md, item: dict, name_key: str, file_key: str, line_key: str):
    name = item[name_key]
    if file_path := item.get(file_key):
        line = f":{item[line_key]}" if item.get(line_key) else ""
        md.item(f"`{name}` \u2192 `{_rel_path(file_path)}{line}`")
    else:
        md.item(f"`{name}`")


async def _do_calls(function_name: str) -> str:
    _ensure_loaded()
    if status := _cpp_status():
        return status

    callees = _state.cpp_extractor.get_callees(function_name, fuzzy=True)
    if not callees:
        return _with_note(f"No outbound calls found for '{function_name}'.")

    results = dedupe_by_key(callees, "callee")

    md = create_formatter()
    md.h2(f"Calls: `{function_name}`")
    md.text("*Functions this calls (outbound dependencies):*\n")

    for item in results[:15]:
        _format_call_item(md, item, "callee", "callee_file", "callee_line")

    if len(results) > 15:
        md.text(f"\n*Showing 15 of {len(results)} calls.*")

    return _with_note(md.build())


async def _do_called_by(function_name: str) -> str:
    _ensure_loaded()
    if status := _cpp_status():
        return status

    callers = _state.cpp_extractor.get_callers(function_name, fuzzy=True)
    if not callers:
        return _with_note(f"No inbound callers found for '{function_name}'.")

    results = dedupe_by_key(callers, "caller")

    md = create_formatter()
    md.h2(f"Called by: `{function_name}`")
    md.text("*Functions that call this (inbound dependents):*\n")

    for item in results[:15]:
        _format_call_item(md, item, "caller", "caller_file", "caller_line")

    if len(results) > 15:
        md.text(f"\n*Showing 15 of {len(results)} callers.*")

    return _with_note(md.build())


async def _do_impact(
    function_name: str,
    depth: int = 2,
    focus: str = "callers",
    fuzzy_all_levels: bool = False,
    walk_python: bool = False,
) -> str:
    _ensure_loaded()
    if status := _cpp_status():
        return status

    depth = min(max(depth, 1), _max_depth())

    visited = set()
    current_level = {function_name}
    callers_by_depth: dict[int, list[dict]] = {}

    for level in range(1, depth + 1):
        next_level = set()
        level_callers = []

        for func in current_level:
            fuzzy = fuzzy_all_levels or level == 1
            for item in _state.cpp_extractor.get_callers(func, fuzzy=fuzzy):
                caller = item["caller"]
                if caller not in visited and caller != function_name:
                    visited.add(caller)
                    next_level.add(caller)
                    level_callers.append(item)

        if level_callers:
            callers_by_depth[level] = level_callers
        current_level = next_level
        if not current_level:
            break

    if not callers_by_depth:
        return _with_note(f"No callers found for '{function_name}'.")

    md = create_formatter()
    md.h2(f"Impact Analysis: `{function_name}`")
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

    if focus == "full":
        python_entries = [
            {
                "python": b.get("python_name", c),
                "cpp": c,
                "dispatch": b.get("dispatch_key", ""),
            }
            for c in visited
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
        for cpp in visited:
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
