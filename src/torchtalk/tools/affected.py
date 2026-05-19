"""Affected-tests tool implementation."""

from __future__ import annotations

from ..analysis.affected import affected_tests
from ..formatting import create_formatter
from ..indexer import _cpp_status, _ensure_loaded, _state


async def _do_affected(funcs: str, depth: int = 3) -> str:
    _ensure_loaded()
    if status := _cpp_status():
        return status

    func_list = [f.strip() for f in funcs.split(",") if f.strip()]
    if not func_list:
        return "No functions provided."

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
    if not runs:
        md.text("*No matching test runs found.*")
        return md.build()

    md.h3(f"Test runs ({len(runs)} files)")
    for tr in runs[:30]:
        classes = tr["included_classes"]
        if classes:
            md.item(f"`{tr['file']}` — {', '.join(classes[:5])}")
            if len(classes) > 5:
                md.item(f"...and {len(classes) - 5} more", 1)
        else:
            md.item(f"`{tr['file']}` *(whole file)*")

    if len(runs) > 30:
        md.text(f"\n*Showing 30 of {len(runs)} files.*")

    return md.build()
