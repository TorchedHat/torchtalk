"""Test infrastructure tool implementations."""

from __future__ import annotations

from pathlib import Path

from ..analysis.helpers import word_match as _word_match
from ..formatting import create_formatter
from ..harness import active_manifest
from ..indexer import _ensure_loaded, _state


async def _do_find_similar_tests(
    query: str, limit: int = 10, focus: str = "all"
) -> str:
    _ensure_loaded("test")

    # Without a query, the file branch's naked-substring match (`q in path`)
    # would match every indexed file — refuse rather than dump 1k+ paths.
    if not query.strip():
        return "Provide a query string to search tests."
    limit = max(1, limit)

    query_lower = query.lower()
    md = create_formatter()
    md.h2(f"Tests matching: `{query}`")

    matching_funcs = []
    if focus in ("all", "functions"):
        for func_name, locations in _state.test_functions.items():
            if _word_match(query_lower, func_name.lower()):
                for loc in locations:
                    matching_funcs.append(
                        {
                            "name": func_name,
                            "class": loc.get("class"),
                            "file": loc["file"],
                            "line": loc["line"],
                        }
                    )

    matching_classes = []
    if focus in ("all", "classes"):
        for class_name, locations in _state.test_classes.items():
            if _word_match(query_lower, class_name.lower()):
                for loc in locations:
                    matching_classes.append(
                        {
                            "name": class_name,
                            "file": loc["file"],
                            "line": loc["line"],
                            "bases": loc.get("bases", []),
                        }
                    )

    matching_files = []
    if focus in ("all", "files"):
        for file_path, info in _state.test_files.items():
            if query_lower in file_path.lower():
                matching_files.append(
                    {
                        "path": file_path,
                        "classes": len(info.get("classes", [])),
                        "functions": len(info.get("functions", [])),
                    }
                )

    matching_opinfo = []
    if focus in ("all", "functions"):
        for op_name, info in _state.opinfo_registry.items():
            if _word_match(query_lower, op_name.lower()):
                matching_opinfo.append(info)

    total = (
        len(matching_funcs)
        + len(matching_classes)
        + len(matching_files)
        + len(matching_opinfo)
    )

    if total == 0:
        return f"No tests found matching `{query}`."

    md.text(f"Found {total} matches\n")

    if matching_opinfo:
        md.h3(f"OpInfo Definitions ({len(matching_opinfo)})")
        md.text("*Operators with official test metadata:*\n")
        for info in matching_opinfo[:5]:
            md.item(f"`{info['name']}` → `{info['file']}:{info['line']}`")
        if len(matching_opinfo) > 5:
            md.item(f"*... and {len(matching_opinfo) - 5} more*")
        md.blank()

    if matching_funcs:
        md.h3(f"Test Functions ({len(matching_funcs)})")
        for func in matching_funcs[:limit]:
            class_prefix = f"{func['class']}." if func.get("class") else ""
            md.item(f"`{class_prefix}{func['name']}` → `{func['file']}:{func['line']}`")
        if len(matching_funcs) > limit:
            md.item(f"*... and {len(matching_funcs) - limit} more*")
        md.blank()

    if matching_classes:
        md.h3(f"Test Classes ({len(matching_classes)})")
        for cls in matching_classes[:5]:
            bases = f" ({', '.join(cls['bases'][:2])})" if cls.get("bases") else ""
            md.item(f"`{cls['name']}`{bases} → `{cls['file']}:{cls['line']}`")
        if len(matching_classes) > 5:
            md.item(f"*... and {len(matching_classes) - 5} more*")
        md.blank()

    if matching_files:
        md.h3(f"Test Files ({len(matching_files)})")
        for f in matching_files[:5]:
            md.item(f"`{f['path']}` ({f['classes']} classes, {f['functions']} tests)")
        if len(matching_files) > 5:
            md.item(f"*... and {len(matching_files) - 5} more*")

    return md.build()


async def _do_list_test_utils() -> str:
    _ensure_loaded("test")

    md = create_formatter()
    md.h2("Test Utilities")

    manifest = active_manifest()
    md.h3("Core Utilities")
    if not manifest.test_utility_modules:
        md.text("*No test utility modules are configured for this harness*")
    for path in manifest.test_utility_modules:
        if path in _state.test_utilities:
            exists = True
        elif _state.source:
            exists = (Path(_state.source) / path).exists()
        else:
            exists = False
        status = "[ok]" if exists else "[missing]"
        md.item(f"**{path}** {status}")
        if note := manifest.test_utility_notes.get(path):
            md.item(f"*{note}*", 1)
        md.blank()

    md.h3("Test Infrastructure Stats")
    if _state.test_files:
        md.item(f"Test files indexed: {len(_state.test_files)}")
        md.item(f"Test classes: {len(_state.test_classes)}")
        md.item(f"Test functions: {len(_state.test_functions)}")
        md.item(f"OpInfo definitions: {len(_state.opinfo_registry)}")
    else:
        md.text("*Test infrastructure not yet indexed*")

    if manifest.test_patterns:
        md.h3("Common Test Patterns")
        for code, desc in manifest.test_patterns.items():
            md.item(f"`{code}`: *{desc}*")

    return md.build()


async def _do_test_file_info(query: str) -> str:
    _ensure_loaded("test")
    if not query.strip():
        return "Provide a test file name."

    needle = query.lower()
    matches = []
    for path, info in _state.test_files.items():
        if needle in path.lower():
            matches.append((path, info))

    if not matches:
        return f"No test file found matching `{query}`."

    md = create_formatter()

    for path, info in matches[:3]:
        md.h2(f"Test File: `{path}`")

        if info.get("classes"):
            md.h3(f"Test Classes ({len(info['classes'])})")
            for cls in info["classes"][:10]:
                bases = (
                    f" extends {', '.join(cls['bases'][:2])}"
                    if cls.get("bases")
                    else ""
                )
                md.item(f"`{cls['name']}`{bases} (line {cls['line']})")

        if info.get("functions"):
            md.h3(f"Test Functions ({len(info['functions'])})")
            for func in info["functions"][:20]:
                class_prefix = f"{func['class']}." if func.get("class") else ""
                md.item(f"`{class_prefix}{func['name']}` (line {func['line']})")
            if len(info["functions"]) > 20:
                md.item(f"*... and {len(info['functions']) - 20} more*")

        md.blank()

    if len(matches) > 3:
        md.text(f"*Showing 3 of {len(matches)} matching files*")

    return md.build()
