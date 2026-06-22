"""Benchmark real TorchTalk MCP calls against rg + file-read navigation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

try:
    import tiktoken
except ImportError:  # pragma: no cover - benchmark fallback only
    tiktoken = None


CODE_GLOBS = ("*.py", "*.cpp", "*.cc", "*.cu", "*.cuh", "*.h", "*.hpp")
CODE_SUFFIXES = {".py", ".cpp", ".cc", ".cu", ".cuh", ".h", ".hpp"}
RG_BINARY = shutil.which("rg")


@dataclass
class BenchmarkTask:
    task_id: str
    category: str
    description: str
    tool_name: str
    tool_args: dict[str, Any]
    baseline_queries: list[str]
    required_groups: list[list[str]]
    ordered_groups: list[list[str]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TorchTalk MCP vs rg + file reads on vLLM."
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
        default=str(Path(__file__).with_name("vllm_mcp_vs_rg_results.json")),
        help="Path to write machine-readable results JSON",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of warm repeated measurements per task and method",
    )
    parser.add_argument(
        "--rg-max-matches-per-query",
        type=int,
        default=6,
        help="Maximum rg matches to keep per query",
    )
    parser.add_argument(
        "--baseline-max-snippets",
        type=int,
        default=18,
        help="Maximum snippets to include in the rg baseline output",
    )
    parser.add_argument(
        "--context-lines",
        type=int,
        default=2,
        help="Context lines to include around a baseline hit",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_tasks(path: str) -> list[BenchmarkTask]:
    payload = json.loads(Path(path).read_text())
    return [
        BenchmarkTask(
            task_id=item["id"],
            category=item["category"],
            description=item["description"],
            tool_name=item["tool_name"],
            tool_args=item.get("tool_args", {}),
            baseline_queries=item["baseline_queries"],
            required_groups=item["required_groups"],
            ordered_groups=item.get("ordered_groups"),
        )
        for item in payload
    ]


def count_tokens(text: str) -> int:
    if tiktoken is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def match_groups(text: str, groups: list[list[str]]) -> tuple[int, list[int]]:
    lowered = text.lower()
    positions: list[int] = []
    matched = 0
    for group in groups:
        found_positions = [
            lowered.find(marker.lower())
            for marker in group
            if lowered.find(marker.lower()) != -1
        ]
        if not found_positions:
            positions.append(-1)
            continue
        matched += 1
        positions.append(min(found_positions))
    return matched, positions


def evaluate_output(
    text: str,
    required_groups: list[list[str]],
    ordered_groups: list[list[str]] | None = None,
) -> dict[str, Any]:
    matched_required, required_positions = match_groups(text, required_groups)
    total_required = len(required_groups)
    coverage = matched_required / total_required if total_required else 1.0

    ordered_score = 1.0
    if ordered_groups:
        _, ordered_positions = match_groups(text, ordered_groups)
        if any(position < 0 for position in ordered_positions):
            ordered_score = 0.0
        else:
            ordered_score = (
                1.0 if ordered_positions == sorted(ordered_positions) else 0.0
            )

    score = round(100 * ((0.7 * coverage) + (0.3 * ordered_score)))
    return {
        "matched_required": matched_required,
        "total_required": total_required,
        "coverage": round(coverage, 4),
        "ordered_score": ordered_score,
        "score": score,
        "required_positions": required_positions,
    }


def _server_env(source_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    src_path = str(repo_root() / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}:{existing}"
    env["VLLM_SOURCE"] = str(source_root)
    return env


def _tool_result_text(result: Any) -> str:
    chunks: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
            continue
        chunks.append(str(block))
    return "\n".join(chunks)


async def _time_mcp_tool(
    session: ClientSession,
    task: BenchmarkTask,
    repeats: int,
) -> tuple[str, float]:
    durations: list[float] = []
    output = ""
    for _ in range(repeats):
        start = time.perf_counter()
        result = await session.call_tool(task.tool_name, task.tool_args)
        output = _tool_result_text(result)
        durations.append((time.perf_counter() - start) * 1000)
    return output, statistics.mean(durations)


def _run_rg_query(
    source_root: Path,
    query: str,
    max_matches: int,
) -> list[dict[str, Any]]:
    if RG_BINARY:
        command = [
            RG_BINARY,
            "-n",
            "-i",
            "-F",
            "--color",
            "never",
            "--max-count",
            str(max_matches),
        ]
        for glob_pattern in CODE_GLOBS:
            command.extend(["-g", glob_pattern])
        command.extend([query, str(source_root)])
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(
                f"rg failed for query {query!r}: {result.stderr.strip()}"
            )

        matches: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            path, line_number, text = line.split(":", 2)
            matches.append(
                {
                    "query": query,
                    "path": Path(path),
                    "line_number": int(line_number),
                    "line_text": text,
                }
            )
        return matches

    lowered_query = query.lower()
    matches = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
            continue
        try:
            file_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:  # pragma: no cover - filesystem race or permission edge case
            continue
        for line_number, line_text in enumerate(file_lines, start=1):
            if lowered_query not in line_text.lower():
                continue
            matches.append(
                {
                    "query": query,
                    "path": path,
                    "line_number": line_number,
                    "line_text": line_text,
                }
            )
            if len(matches) >= max_matches:
                return matches
    return matches


def _extract_snippet(path: Path, line_number: int, context_lines: int) -> str:
    file_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, line_number - context_lines)
    end = min(len(file_lines), line_number + context_lines)
    snippet_lines = [
        f"{path}:{current}: {file_lines[current - 1]}"
        for current in range(start, end + 1)
    ]
    return "\n".join(snippet_lines)


def baseline_bundle(
    source_root: Path,
    task: BenchmarkTask,
    max_matches_per_query: int,
    max_snippets: int,
    context_lines: int,
) -> tuple[str, dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for query in task.baseline_queries:
        matches.extend(
            _run_rg_query(
                source_root,
                query,
                max_matches=max_matches_per_query,
            )
        )

    lines = [f"[rg+read baseline: `{task.task_id}`]"]
    seen_snippets: set[tuple[str, int]] = set()
    used_files: set[str] = set()
    snippet_count = 0
    query_hit_counts = {query: 0 for query in task.baseline_queries}

    for match in matches:
        query_hit_counts[match["query"]] += 1

    for match in matches:
        key = (str(match["path"]), match["line_number"])
        if key in seen_snippets:
            continue
        seen_snippets.add(key)
        used_files.add(str(match["path"]))
        lines.append("")
        lines.append(f"## query={match['query']}")
        lines.append(
            _extract_snippet(
                match["path"],
                match["line_number"],
                context_lines,
            )
        )
        snippet_count += 1
        if snippet_count >= max_snippets:
            break

    return "\n".join(lines), {
        "query_hit_counts": query_hit_counts,
        "unique_files": len(used_files),
        "snippet_count": snippet_count,
    }


def benchmark_baseline(
    source_root: Path,
    task: BenchmarkTask,
    repeats: int,
    max_matches_per_query: int,
    max_snippets: int,
    context_lines: int,
) -> tuple[str, float, dict[str, Any]]:
    durations: list[float] = []
    output = ""
    metadata: dict[str, Any] = {}
    for _ in range(repeats):
        start = time.perf_counter()
        output, metadata = baseline_bundle(
            source_root,
            task,
            max_matches_per_query=max_matches_per_query,
            max_snippets=max_snippets,
            context_lines=context_lines,
        )
        durations.append((time.perf_counter() - start) * 1000)
    return output, statistics.mean(durations), metadata


def _average(items: list[float | int]) -> float:
    return round(statistics.mean(items), 2) if items else 0.0


def _category_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({item["category"] for item in results})
    summary: dict[str, Any] = {}
    for category in categories:
        rows = [item for item in results if item["category"] == category]
        torchtalk_scores = [row["torchtalk"]["score"] for row in rows]
        baseline_scores = [row["baseline"]["score"] for row in rows]
        torchtalk_tokens = [row["torchtalk"]["tokens"] for row in rows]
        baseline_tokens = [row["baseline"]["tokens"] for row in rows]
        torchtalk_times = [row["torchtalk"]["time_ms"] for row in rows]
        baseline_times = [row["baseline"]["time_ms"] for row in rows]
        summary[category] = {
            "task_count": len(rows),
            "avg_torchtalk_score": _average(torchtalk_scores),
            "avg_baseline_score": _average(baseline_scores),
            "avg_torchtalk_tokens": _average(torchtalk_tokens),
            "avg_baseline_tokens": _average(baseline_tokens),
            "avg_torchtalk_time_ms": _average(torchtalk_times),
            "avg_baseline_time_ms": _average(baseline_times),
        }
    return summary


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    torchtalk_scores = [item["torchtalk"]["score"] for item in results]
    baseline_scores = [item["baseline"]["score"] for item in results]
    torchtalk_tokens = [item["torchtalk"]["tokens"] for item in results]
    baseline_tokens = [item["baseline"]["tokens"] for item in results]
    torchtalk_times = [item["torchtalk"]["time_ms"] for item in results]
    baseline_times = [item["baseline"]["time_ms"] for item in results]
    baseline_unique_files = [item["baseline"]["unique_files"] for item in results]
    baseline_snippets = [item["baseline"]["snippet_count"] for item in results]
    return {
        "task_count": len(results),
        "avg_torchtalk_score": _average(torchtalk_scores),
        "avg_baseline_score": _average(baseline_scores),
        "avg_torchtalk_tokens": _average(torchtalk_tokens),
        "avg_baseline_tokens": _average(baseline_tokens),
        "avg_torchtalk_time_ms": _average(torchtalk_times),
        "avg_baseline_time_ms": _average(baseline_times),
        "avg_baseline_unique_files": _average(baseline_unique_files),
        "avg_baseline_snippets": _average(baseline_snippets),
        "torchtalk_wins_on_score": sum(
            1
            for item in results
            if item["torchtalk"]["score"] > item["baseline"]["score"]
        ),
        "baseline_wins_on_score": sum(
            1
            for item in results
            if item["baseline"]["score"] > item["torchtalk"]["score"]
        ),
        "ties_on_score": sum(
            1
            for item in results
            if item["baseline"]["score"] == item["torchtalk"]["score"]
        ),
        "by_category": _category_summary(results),
    }


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source).resolve()
    tasks = load_tasks(args.tasks)

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
        env=_server_env(source_root),
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
            torchtalk_output, torchtalk_time_ms = await _time_mcp_tool(
                session,
                task,
                args.repeats,
            )
            baseline_output, baseline_time_ms, baseline_metadata = benchmark_baseline(
                source_root,
                task,
                repeats=args.repeats,
                max_matches_per_query=args.rg_max_matches_per_query,
                max_snippets=args.baseline_max_snippets,
                context_lines=args.context_lines,
            )

            torchtalk_metrics = evaluate_output(
                torchtalk_output,
                task.required_groups,
                task.ordered_groups,
            )
            baseline_metrics = evaluate_output(
                baseline_output,
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
                        "tokens": count_tokens(torchtalk_output),
                        "time_ms": round(torchtalk_time_ms, 2),
                        "preview": torchtalk_output.splitlines()[:12],
                    },
                    "baseline": {
                        **baseline_metrics,
                        "tokens": count_tokens(baseline_output),
                        "time_ms": round(baseline_time_ms, 2),
                        "preview": baseline_output.splitlines()[:12],
                        **baseline_metadata,
                    },
                }
            )

    return {
        "source_root": str(source_root),
        "repeats": args.repeats,
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
