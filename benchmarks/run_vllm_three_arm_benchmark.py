"""Benchmark TorchTalk MCP vs raw rg vs a source-guided navigator baseline."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import run_vllm_mcp_vs_rg_benchmark as base
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

SOURCE_ROOT_HINTS = ("vllm/", "csrc/")
STOP_SYMBOLS = {
    "append",
    "apply",
    "args",
    "astype",
    "build",
    "call",
    "check",
    "close",
    "copy",
    "create",
    "data",
    "decode",
    "dict",
    "encode",
    "extend",
    "format",
    "from_pretrained",
    "get",
    "items",
    "kwargs",
    "list",
    "load",
    "loads",
    "map",
    "model",
    "name",
    "open",
    "output",
    "outputs",
    "params",
    "path",
    "paths",
    "pop",
    "print",
    "process",
    "result",
    "results",
    "run",
    "save",
    "self",
    "shape",
    "size",
    "split",
    "step",
    "str",
    "text",
    "tokenize",
    "type",
    "update",
    "value",
    "values",
    "write",
}


@dataclass
class RankedHit:
    query: str
    variant: str
    path: Path
    rel_path: str
    line_number: int
    line_text: str
    block_text: str
    enclosing_class: str | None
    enclosing_function: str | None
    normalized_label: str
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark TorchTalk MCP vs raw rg/file reads vs a source-guided "
            "navigator baseline on vLLM."
        )
    )
    parser.add_argument(
        "--source",
        default="/data/vllm",
        help="Path to the vLLM source tree",
    )
    parser.add_argument(
        "--tasks",
        default=str(Path(__file__).with_name("vllm_navigation_tasks_strict.json")),
        help="Path to the benchmark task spec JSON",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("vllm_three_arm_results.json")),
        help="Path to write machine-readable results JSON",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of warm repeated measurements per task and method",
    )
    parser.add_argument(
        "--raw-rg-max-matches-per-query",
        type=int,
        default=6,
        help="Maximum raw rg matches to keep per query for the raw baseline",
    )
    parser.add_argument(
        "--baseline-max-snippets",
        type=int,
        default=18,
        help="Maximum snippets to include in the raw rg baseline output",
    )
    parser.add_argument(
        "--context-lines",
        type=int,
        default=2,
        help="Context lines to include around a raw rg baseline hit",
    )
    parser.add_argument(
        "--navigator-candidate-matches",
        type=int,
        default=40,
        help="Maximum candidate matches per query variant for the navigator arm",
    )
    parser.add_argument(
        "--navigator-expansion-limit",
        type=int,
        default=8,
        help="Maximum expansion symbols to follow in the navigator arm",
    )
    parser.add_argument(
        "--navigator-primary-limit",
        type=int,
        default=10,
        help="Maximum primary resolved hits to include in the navigator output",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_source_file(rel_path: str) -> bool:
    return rel_path.startswith(SOURCE_ROOT_HINTS)


def _path_penalty(rel_path: str) -> int:
    lowered = rel_path.lower()
    penalty = 0
    if lowered.startswith("tests/") or "/tests/" in lowered:
        penalty += 120
    if lowered.startswith("examples/") or "/examples/" in lowered:
        penalty += 90
    if lowered.startswith("docs/") or "/docs/" in lowered:
        penalty += 80
    if lowered.startswith("benchmarks/") or "/benchmarks/" in lowered:
        penalty += 60
    if lowered.startswith("scripts/") or "/scripts/" in lowered:
        penalty += 50
    if lowered.startswith("experimental/") or "/experimental/" in lowered:
        penalty += 30
    if not _is_source_file(rel_path):
        penalty += 20
    return penalty


def _line_bonus(line_text: str) -> int:
    stripped = line_text.strip()
    if stripped.startswith(("async def ", "def ", "class ")):
        return -40
    if "@register_backend" in stripped or "register_impl" in stripped:
        return -25
    if "ops.impl(" in stripped:
        return -25
    if stripped.startswith("#"):
        return 15
    return 0


def _query_variants(query: str) -> list[str]:
    variants = [query]
    if "." in query:
        parts = query.split(".")
        tail = parts[-1]
        if tail:
            variants.append(tail)
            variants.append(f"def {tail}(")
            variants.append(f"async def {tail}(")
        class_hint = parts[-2] if len(parts) >= 2 else ""
        if class_hint and class_hint not in {"ops", "_C"}:
            variants.append(class_hint)
    if query.startswith("torch.ops._C."):
        tail = query.split(".")[-1]
        variants.extend([tail, f'"{tail}(', f'ops.impl("{tail}"'])
    if query.isupper():
        variants.append(query.replace("_", " "))

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        cleaned = variant.strip()
        if not cleaned or cleaned in seen:
            continue
        deduped.append(cleaned)
        seen.add(cleaned)
    return deduped


def _read_file_lines(path: Path, cache: dict[Path, list[str]]) -> list[str]:
    if path not in cache:
        cache[path] = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return cache[path]


def _build_python_payload(
    path: Path,
    lines_cache: dict[Path, list[str]],
    ast_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any] | None:
    if path in ast_cache:
        return ast_cache[path]
    try:
        text = "\n".join(_read_file_lines(path, lines_cache))
        tree = ast.parse(text)
    except SyntaxError:
        ast_cache[path] = None
        return None

    parent_map: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[id(child)] = node

    functions: list[ast.AST] = []
    classes: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)
        elif isinstance(node, ast.ClassDef):
            classes.append(node)

    ast_cache[path] = {
        "tree": tree,
        "parent_map": parent_map,
        "functions": functions,
        "classes": classes,
    }
    return ast_cache[path]


def _best_enclosing_node(
    nodes: list[ast.AST],
    line_number: int,
) -> ast.AST | None:
    best: ast.AST | None = None
    best_span: int | None = None
    for node in nodes:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None or not (start <= line_number <= end):
            continue
        span = end - start
        if best is None or best_span is None or span < best_span:
            best = node
            best_span = span
    return best


def _parent_class_for_node(node: ast.AST, parent_map: dict[int, ast.AST]) -> str | None:
    current = parent_map.get(id(node))
    while current is not None:
        if isinstance(current, ast.ClassDef):
            return current.name
        current = parent_map.get(id(current))
    return None


def _context_for_hit(
    path: Path,
    line_number: int,
    lines_cache: dict[Path, list[str]],
    ast_cache: dict[Path, dict[str, Any] | None],
) -> tuple[str, str | None, str | None]:
    lines = _read_file_lines(path, lines_cache)
    if path.suffix == ".py":
        payload = _build_python_payload(path, lines_cache, ast_cache)
        if payload:
            function_node = _best_enclosing_node(payload["functions"], line_number)
            class_node = _best_enclosing_node(payload["classes"], line_number)
            enclosing_function = getattr(function_node, "name", None)
            enclosing_class = None
            if function_node is not None:
                enclosing_class = _parent_class_for_node(
                    function_node,
                    payload["parent_map"],
                )
            elif class_node is not None:
                enclosing_class = getattr(class_node, "name", None)

            block_node = function_node or class_node
            if block_node is not None:
                start = max(1, getattr(block_node, "lineno", 1))
                end = min(len(lines), getattr(block_node, "end_lineno", start))
                block = "\n".join(lines[start - 1 : end])
                return block, enclosing_class, enclosing_function

    start = max(1, line_number - 6)
    end = min(len(lines), line_number + 8)
    block = "\n".join(lines[start - 1 : end])
    return block, None, None


def _normalized_label(
    query: str,
    path: Path,
    line_text: str,
    block_text: str,
    enclosing_class: str | None,
    enclosing_function: str | None,
) -> str:
    stripped = line_text.strip()
    if query.startswith("torch.ops._C."):
        return query

    if stripped.startswith("class "):
        match = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
        if match:
            return match.group(1)

    if enclosing_class and enclosing_function:
        return f"{enclosing_class}.{enclosing_function}"
    if enclosing_function:
        return enclosing_function
    if enclosing_class:
        return enclosing_class

    if query.isupper():
        match = re.search(r"\b([A-Z][A-Z0-9_]{2,})\b", stripped)
        if match:
            return match.group(1)

    if "torch_bindings.cpp" in path.name:
        tail = query.split(".")[-1]
        if tail and (
            f'"{tail}(' in block_text
            or f'ops.impl("{tail}"' in block_text
            or f"torch.ops._C.{tail}" in block_text
        ):
            return f"torch.ops._C.{tail}"

    if "kernels/" in path.as_posix():
        stem = path.stem
        provider = stem[:-4] if stem.endswith("_ops") else stem
        tail = query.split(".")[-1]
        if tail and tail in block_text and "register_impl" in block_text:
            return f"{tail}::{provider}"

    dotted = re.search(
        r"\b([A-Z][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)\b",
        block_text,
    )
    if dotted:
        return dotted.group(1)

    return query


def _ranked_hits_for_query(
    source_root: Path,
    query: str,
    candidate_matches: int,
    lines_cache: dict[Path, list[str]],
    ast_cache: dict[Path, dict[str, Any] | None],
    search_cache: dict[tuple[str, int], list[dict[str, Any]]],
) -> list[RankedHit]:
    candidates: list[RankedHit] = []
    seen: set[tuple[str, int]] = set()
    variants = _query_variants(query)

    query_name = Path(query).name
    if Path(query_name).suffix in base.CODE_SUFFIXES:
        for path in source_root.rglob(query_name):
            if not path.is_file():
                continue
            key = (str(path), 1)
            if key in seen:
                continue
            seen.add(key)
            rel_path = str(path.relative_to(source_root))
            block_text, enclosing_class, enclosing_function = _context_for_hit(
                path,
                1,
                lines_cache,
                ast_cache,
            )
            candidates.append(
                RankedHit(
                    query=query,
                    variant=query_name,
                    path=path,
                    rel_path=rel_path,
                    line_number=1,
                    line_text=rel_path,
                    block_text=block_text,
                    enclosing_class=enclosing_class,
                    enclosing_function=enclosing_function,
                    normalized_label=_normalized_label(
                        query,
                        path,
                        rel_path,
                        block_text,
                        enclosing_class,
                        enclosing_function,
                    ),
                    score=float(_path_penalty(rel_path) - 35),
                )
            )

    for variant in variants:
        cache_key = (variant, candidate_matches)
        if cache_key not in search_cache:
            search_cache[cache_key] = base._run_rg_query(
                source_root,
                variant,
                max_matches=candidate_matches,
            )
        raw_hits = search_cache[cache_key]
        for hit in raw_hits:
            key = (str(hit["path"]), hit["line_number"])
            if key in seen:
                continue
            seen.add(key)
            rel_path = str(hit["path"].relative_to(source_root))
            block_text, enclosing_class, enclosing_function = _context_for_hit(
                hit["path"],
                hit["line_number"],
                lines_cache,
                ast_cache,
            )

            score = float(_path_penalty(rel_path) + _line_bonus(hit["line_text"]))
            lowered_query = query.lower()
            lowered_variant = variant.lower()
            lowered_line = hit["line_text"].lower()
            lowered_block = block_text.lower()
            query_tail = query.split(".")[-1].lower()

            if lowered_query in lowered_line:
                score -= 35
            elif lowered_query in lowered_block:
                score -= 25
            elif lowered_variant in lowered_line:
                score -= 18

            if hit["path"].name.lower() in (
                lowered_query,
                lowered_variant,
            ):
                score -= 28
            if hit["path"].stem.lower() in (
                lowered_query,
                lowered_variant,
            ):
                score -= 24

            if rel_path.startswith("vllm/ir/ops/"):
                score -= 14
            if rel_path.startswith("vllm/kernels/"):
                score -= 10
            if rel_path.startswith("csrc/libtorch_stable/"):
                score -= 12

            if "." in query:
                class_hint, tail = query.rsplit(".", 1)
                class_hint = class_hint.split(".")[-1]
                if enclosing_class and enclosing_class.lower() == class_hint.lower():
                    score -= 18
                if enclosing_function and enclosing_function.lower() == tail.lower():
                    score -= 18
            elif (
                enclosing_function and enclosing_function.lower() == query.lower()
            ) or (enclosing_class and enclosing_class.lower() == query.lower()):
                score -= 20
            elif enclosing_function and enclosing_function.lower() == query_tail:
                score -= 12

            if query.startswith("torch.ops._C."):
                tail = query.split(".")[-1]
                if "torch_bindings.cpp" in rel_path:
                    score -= 25
                if (
                    f'"{tail}(' in lowered_block
                    or f'ops.impl("{tail}"' in lowered_block
                ):
                    score -= 18

            if query.isupper() and "=" in hit["line_text"]:
                score -= 12
            if "register_impl" in lowered_block:
                score -= 8
            if (
                query_tail
                and f"def {query_tail}(" in lowered_block
                and rel_path.startswith(("vllm/ir/ops/", "vllm/kernels/", "csrc/"))
            ):
                score -= 12

            candidates.append(
                RankedHit(
                    query=query,
                    variant=variant,
                    path=hit["path"],
                    rel_path=rel_path,
                    line_number=hit["line_number"],
                    line_text=hit["line_text"],
                    block_text=block_text,
                    enclosing_class=enclosing_class,
                    enclosing_function=enclosing_function,
                    normalized_label=_normalized_label(
                        query,
                        hit["path"],
                        hit["line_text"],
                        block_text,
                        enclosing_class,
                        enclosing_function,
                    ),
                    score=score,
                )
            )

    candidates.sort(key=lambda item: (item.score, item.rel_path, item.line_number))
    return candidates


def _extract_expansion_queries(hit: RankedHit) -> list[str]:
    text = hit.block_text
    expansions: list[str] = []

    for match in re.findall(r"torch\.ops\._C\.[A-Za-z_][A-Za-z0-9_]*", text):
        expansions.append(match)

    for match in re.findall(r"\b[A-Z][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b", text):
        expansions.append(match)

    for match in re.findall(
        r"\b(?:self|cls)\.[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b",
        text,
    ):
        parts = [part for part in match.split(".") if part not in {"self", "cls"}]
        if len(parts) >= 2:
            expansions.append(f"{parts[-2]}.{parts[-1]}")
        expansions.append(parts[-1])

    for match in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text):
        expansions.append(match)

    if hit.enclosing_function:
        expansions.append(hit.enclosing_function)

    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for symbol in expansions:
        cleaned = symbol.strip()
        tail = cleaned.split(".")[-1]
        if (
            not cleaned
            or cleaned in seen
            or len(tail) < 3
            or tail.lower() in STOP_SYMBOLS
        ):
            continue
        seen.add(cleaned)
        priority = 0
        if cleaned.startswith("torch.ops._C."):
            priority += 8
        if cleaned.startswith("_"):
            priority += 6
        if "." in cleaned:
            priority += 5
        if any(char.isupper() for char in cleaned) and any(
            char.islower() for char in cleaned
        ):
            priority += 4
        if "_" in cleaned:
            priority += 3
        if cleaned.isupper():
            priority += 1
        scored.append((priority, cleaned))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [symbol for _, symbol in scored]


def source_guided_bundle(
    source_root: Path,
    task: base.BenchmarkTask,
    candidate_matches: int,
    expansion_limit: int,
    primary_limit: int,
    lines_cache: dict[Path, list[str]],
    ast_cache: dict[Path, dict[str, Any] | None],
    search_cache: dict[tuple[str, int], list[dict[str, Any]]],
) -> tuple[str, dict[str, Any]]:
    primary_hits: list[RankedHit] = []
    primary_seen_labels: set[str] = set()
    primary_seen_keys: set[tuple[str, int]] = set()

    for query in task.baseline_queries:
        ranked = _ranked_hits_for_query(
            source_root,
            query,
            candidate_matches=candidate_matches,
            lines_cache=lines_cache,
            ast_cache=ast_cache,
            search_cache=search_cache,
        )
        for hit in ranked:
            key = (hit.rel_path, hit.line_number)
            if key in primary_seen_keys:
                continue
            primary_hits.append(hit)
            primary_seen_keys.add(key)
            primary_seen_labels.add(hit.normalized_label)
            break
        if len(primary_hits) >= primary_limit:
            break

    expansion_symbols: list[str] = []
    expansion_seen: set[str] = set()
    for hit in primary_hits:
        for symbol in _extract_expansion_queries(hit):
            if symbol in expansion_seen or symbol in task.baseline_queries:
                continue
            expansion_seen.add(symbol)
            expansion_symbols.append(symbol)
            if len(expansion_symbols) >= expansion_limit:
                break
        if len(expansion_symbols) >= expansion_limit:
            break

    expansion_hits: list[RankedHit] = []
    expansion_seen_keys = set(primary_seen_keys)
    for symbol in expansion_symbols:
        ranked = _ranked_hits_for_query(
            source_root,
            symbol,
            candidate_matches=max(12, candidate_matches // 2),
            lines_cache=lines_cache,
            ast_cache=ast_cache,
            search_cache=search_cache,
        )
        for hit in ranked:
            key = (hit.rel_path, hit.line_number)
            if key in expansion_seen_keys:
                continue
            expansion_hits.append(hit)
            expansion_seen_keys.add(key)
            break

    lines = [f"[source-guided navigator baseline: `{task.task_id}`]"]
    lines.append("Seed resolution")
    for index, hit in enumerate(primary_hits, start=1):
        lines.append(
            f"- {index}. `{hit.normalized_label}` → `{hit.rel_path}:{hit.line_number}` "
            f"[query={hit.query}]"
        )

    if expansion_hits:
        lines.append("")
        lines.append("Agent follow-ups")
        for index, hit in enumerate(expansion_hits[:expansion_limit], start=1):
            lines.append(
                f"- {index}. `{hit.normalized_label}` → "
                f"`{hit.rel_path}:{hit.line_number}` "
                f"[from={hit.query}]"
            )

    graph_hits = primary_hits + expansion_hits
    has_flash_attn = any("FLASH_ATTN" in hit.normalized_label for hit in graph_hits)
    has_backend_cls = any(
        "get_attn_backend_cls" in hit.normalized_label for hit in graph_hits
    )
    if task.category == "graph" and (has_flash_attn or has_backend_cls):
        lines.append("")
        lines.append("Reasoning notes")
        lines.append("- confidence=conditional")
        lines.append("- source implementations were preferred over tests and examples")

    metadata = {
        "unique_files": len({hit.rel_path for hit in primary_hits + expansion_hits}),
        "seed_hits": len(primary_hits),
        "expansion_hits": len(expansion_hits),
        "expansion_symbols": expansion_symbols,
    }
    return "\n".join(lines), metadata


def benchmark_source_guided(
    source_root: Path,
    task: base.BenchmarkTask,
    repeats: int,
    candidate_matches: int,
    expansion_limit: int,
    primary_limit: int,
    lines_cache: dict[Path, list[str]],
    ast_cache: dict[Path, dict[str, Any] | None],
    search_cache: dict[tuple[str, int], list[dict[str, Any]]],
) -> tuple[str, float, dict[str, Any]]:
    durations: list[float] = []
    output = ""
    metadata: dict[str, Any] = {}
    for _ in range(repeats):
        start = time.perf_counter()
        output, metadata = source_guided_bundle(
            source_root,
            task,
            candidate_matches=candidate_matches,
            expansion_limit=expansion_limit,
            primary_limit=primary_limit,
            lines_cache=lines_cache,
            ast_cache=ast_cache,
            search_cache=search_cache,
        )
        durations.append((time.perf_counter() - start) * 1000)
    return output, statistics.mean(durations), metadata


def _arm_summary(results: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    return {
        "avg_score": base._average([item[arm]["score"] for item in results]),
        "avg_tokens": base._average([item[arm]["tokens"] for item in results]),
        "avg_time_ms": base._average([item[arm]["time_ms"] for item in results]),
    }


def _category_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({item["category"] for item in results})
    summary: dict[str, Any] = {}
    for category in categories:
        rows = [item for item in results if item["category"] == category]
        summary[category] = {
            "task_count": len(rows),
            "torchtalk": _arm_summary(rows, "torchtalk"),
            "source_guided": _arm_summary(rows, "source_guided"),
            "rg_baseline": _arm_summary(rows, "rg_baseline"),
        }
    return summary


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_count": len(results),
        "torchtalk": _arm_summary(results, "torchtalk"),
        "source_guided": _arm_summary(results, "source_guided"),
        "rg_baseline": _arm_summary(results, "rg_baseline"),
        "torchtalk_best_or_tied_on_score": sum(
            1
            for item in results
            if item["torchtalk"]["score"]
            >= max(item["source_guided"]["score"], item["rg_baseline"]["score"])
        ),
        "source_guided_best_or_tied_on_score": sum(
            1
            for item in results
            if item["source_guided"]["score"]
            >= max(item["torchtalk"]["score"], item["rg_baseline"]["score"])
        ),
        "rg_baseline_best_or_tied_on_score": sum(
            1
            for item in results
            if item["rg_baseline"]["score"]
            >= max(item["torchtalk"]["score"], item["source_guided"]["score"])
        ),
        "avg_source_guided_unique_files": base._average(
            [item["source_guided"]["unique_files"] for item in results]
        ),
        "avg_source_guided_seed_hits": base._average(
            [item["source_guided"]["seed_hits"] for item in results]
        ),
        "avg_source_guided_expansion_hits": base._average(
            [item["source_guided"]["expansion_hits"] for item in results]
        ),
        "avg_rg_unique_files": base._average(
            [item["rg_baseline"]["unique_files"] for item in results]
        ),
        "avg_rg_snippets": base._average(
            [item["rg_baseline"]["snippet_count"] for item in results]
        ),
        "by_category": _category_summary(results),
    }


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source).resolve()
    tasks = base.load_tasks(args.tasks)
    navigator_lines_cache: dict[Path, list[str]] = {}
    navigator_ast_cache: dict[Path, dict[str, Any] | None] = {}
    navigator_search_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}

    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "torchtalk.cli",
            "mcp-serve",
            "--framework",
            "vllm",
            "--source",
            str(source_root),
        ],
        cwd=str(repo_root()),
        env=base._server_env(source_root),
    )

    results: list[dict[str, Any]] = []
    initialize_ms = 0.0
    list_tools_ms = 0.0
    tools: list[str] = []

    async with (
        stdio_client(params) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        start = time.perf_counter()
        await session.initialize()
        initialize_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        tool_listing = await session.list_tools()
        list_tools_ms = (time.perf_counter() - start) * 1000
        tools = sorted(tool.name for tool in tool_listing.tools)

        for task in tasks:
            torchtalk_output, torchtalk_time_ms = await base._time_mcp_tool(
                session,
                task,
                args.repeats,
            )
            source_guided_output, source_guided_time_ms, source_guided_metadata = (
                benchmark_source_guided(
                    source_root,
                    task,
                    repeats=args.repeats,
                    candidate_matches=args.navigator_candidate_matches,
                    expansion_limit=args.navigator_expansion_limit,
                    primary_limit=args.navigator_primary_limit,
                    lines_cache=navigator_lines_cache,
                    ast_cache=navigator_ast_cache,
                    search_cache=navigator_search_cache,
                )
            )
            rg_output, rg_time_ms, rg_metadata = base.benchmark_baseline(
                source_root,
                task,
                repeats=args.repeats,
                max_matches_per_query=args.raw_rg_max_matches_per_query,
                max_snippets=args.baseline_max_snippets,
                context_lines=args.context_lines,
            )

            torchtalk_metrics = base.evaluate_output(
                torchtalk_output,
                task.required_groups,
                task.ordered_groups,
            )
            source_guided_metrics = base.evaluate_output(
                source_guided_output,
                task.required_groups,
                task.ordered_groups,
            )
            rg_metrics = base.evaluate_output(
                rg_output,
                task.required_groups,
                task.ordered_groups,
            )

            results.append(
                {
                    "id": task.task_id,
                    "category": task.category,
                    "description": task.description,
                    "tool_name": task.tool_name,
                    "torchtalk": {
                        **torchtalk_metrics,
                        "tokens": base.count_tokens(torchtalk_output),
                        "time_ms": round(torchtalk_time_ms, 2),
                        "preview": torchtalk_output.splitlines()[:12],
                    },
                    "source_guided": {
                        **source_guided_metrics,
                        "tokens": base.count_tokens(source_guided_output),
                        "time_ms": round(source_guided_time_ms, 2),
                        "preview": source_guided_output.splitlines()[:12],
                        **source_guided_metadata,
                    },
                    "rg_baseline": {
                        **rg_metrics,
                        "tokens": base.count_tokens(rg_output),
                        "time_ms": round(rg_time_ms, 2),
                        "preview": rg_output.splitlines()[:12],
                        **rg_metadata,
                    },
                }
            )

    return {
        "source_root": str(source_root),
        "repeats": args.repeats,
        "limitations": {
            "true_cursor_sdk_agent_unavailable": True,
            "reason": (
                "cursor_sdk was not installed and CURSOR_API_KEY was not available, "
                "so the third arm uses a deterministic no-TorchTalk source-guided "
                "navigator instead of a live Cursor SDK agent."
            ),
        },
        "server": {
            "initialize_time_ms": round(initialize_ms, 2),
            "list_tools_time_ms": round(list_tools_ms, 2),
            "tool_names": tools,
            "command": [params.command, *params.args],
        },
        "tasks": results,
        "summary": summarize_results(results),
    }


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run_benchmark(args))
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
