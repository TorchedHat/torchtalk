"""Affected-tests tool implementation."""

from __future__ import annotations

from ..analysis.affected import affected_tests
from ..formatting import create_formatter
from ..indexer import _cpp_status, _ensure_loaded, _state
from .graph import _clamp_depth


def _split_funcs(funcs: str) -> list[str]:
    """Split on commas outside angle brackets so template args stay intact."""
    out: list[str] = []
    buf: list[str] = []
    nesting = 0
    for ch in funcs:
        if ch == "<":
            nesting += 1
        elif ch == ">":
            nesting = max(0, nesting - 1)
        if ch == "," and nesting == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf).strip())
    return [f for f in out if f]


async def _do_affected(funcs: str, depth: int = 3) -> str:
    _ensure_loaded()
    if status := _cpp_status():
        return status

    func_list = _split_funcs(funcs)
    if not func_list:
        return "No functions provided."

    depth = _clamp_depth(depth)

    result = affected_tests(
        funcs=func_list,
        cpp_extractor=_state.cpp_extractor,
        by_cpp_name=_state.by_cpp_name,
        test_classes=_state.test_classes,
        test_files=_state.test_files,
        opinfo_registry=_state.opinfo_registry,
        opinfo_alias_map=_state.opinfo_alias_map,
        opinfo_test_files=_state.opinfo_test_files,
        test_attr_index=_state.test_attr_index,
        python_profiling=_state.python_profiling or None,
        decomp_alias_map=_state.decomp_alias_map or None,
        backward_to_forward=_state.backward_to_forward or None,
        native_functions=_state.native_functions or None,
        native_implementations=_state.native_implementations or None,
        kernel_impl_to_op=_state.kernel_impl_to_op or None,
        dispatch_to_op=_state.dispatch_to_op or None,
        bindings_by_file=_state.bindings_by_file or None,
        ops_by_file=_state.ops_by_file or None,
        symbol_to_file=_state.symbol_to_file or None,
        test_functions=_state.test_functions or None,
        depth=depth,
    )

    md = create_formatter()
    md.h2(f"Affected tests for: `{', '.join(func_list)}`")
    md.item(f"Callers walked: {result['callers_walked']}")
    md.item(f"Bindings matched: {len(result['bindings_matched'])}")

    apis = result["python_apis"]
    tiers = result.get("api_tier", {})
    sources = result.get("api_sources", {})
    if not apis:
        md.item("Python APIs: (none)")
    else:
        precise = sorted(a for a in apis if tiers.get(a) == "precise")
        fuzzy = sorted(a for a in apis if tiers.get(a) == "fuzzy")
        md.item(
            f"Python APIs: {len(apis)} total "
            f"({len(precise)} precise, {len(fuzzy)} fuzzy)"
        )
        if precise:
            # Inline source tags — diagnoses *why* each precise api is trusted
            # (call_graph vs dispatch vs alias bridge).
            tagged = [f"{a} [{','.join(sources.get(a, []))}]" for a in precise[:10]]
            preview = ", ".join(tagged)
            suffix = f" *+{len(precise) - 10} more*" if len(precise) > 10 else ""
            md.item(f"Precise: {preview}{suffix}", 1)
        if fuzzy:
            # Fuzzy tags are usually `cohort`/`mention`/`vendor` and add noise
            # at scale; keep the list compact and skip per-api tags here.
            preview = ", ".join(fuzzy[:10])
            suffix = f" *+{len(fuzzy) - 10} more*" if len(fuzzy) > 10 else ""
            md.item(f"Fuzzy: {preview}{suffix}", 1)
    md.blank()

    runs = result["test_runs"]
    opinfo_runs = result.get("opinfo_runs", [])
    if not runs and not opinfo_runs:
        md.text("*No matching test runs found.*")
        return md.build()

    # Pytest node IDs: file::Class, refined to ::Class::function where the
    # test_functions join found exact matches.
    function_hits = result.get("function_hits", {})
    node_ids: list[str] = []
    for tr in runs:
        file = tr["file"]
        classes = tr["included_classes"]
        if not classes:
            node_ids.append(file)
            continue
        for cls in classes:
            funcs_in_cls = function_hits.get(file, {}).get(cls, [])
            if funcs_in_cls:
                node_ids.extend(f"{file}::{cls}::{fn}" for fn in funcs_in_cls)
            else:
                node_ids.append(f"{file}::{cls}")

    md.h3(f"Test runs ({len(runs) + len(opinfo_runs)} selections)")
    for nid in node_ids:
        md.item(f"`{nid}`")
    for orun in opinfo_runs:
        md.item(f'`{" ".join(orun["files"])} -k "{orun["k"]}"`')

    lines = []
    if node_ids:
        lines.append("pytest " + " ".join(node_ids))
    lines.extend(
        f'pytest {" ".join(orun["files"])} -k "{orun["k"]}"' for orun in opinfo_runs
    )
    md.blank()
    md.codeblock("\n".join(lines))

    return md.build()
