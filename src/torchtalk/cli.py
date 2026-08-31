#!/usr/bin/env python3
"""
TorchTalk CLI - Cross-language binding analysis for PyTorch codebases.

Quick start:
    torchtalk init --source /path/to/pytorch
    claude mcp add torchtalk -s user -- torchtalk mcp-serve

The MCP server automatically builds and caches the binding index on first run.
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _activate_harness(name: str | None) -> bool:
    """Activate the named harness, or the configured default when name is None.

    Returns False (after logging) when the harness is not registered.
    """
    from torchtalk.config import default_harness
    from torchtalk.harness import set_active_harness

    target = name or default_harness()
    if not target:
        return True
    try:
        set_active_harness(target)
    except KeyError as e:
        log.error(str(e))
        return False
    return True


def cmd_init(args):
    """Configure TorchTalk with a source path for a harness."""
    from torchtalk.config import set_source, validate_source_path
    from torchtalk.harness import get_harness

    try:
        manifest = get_harness(args.harness).manifest
    except KeyError as e:
        log.error(str(e))
        sys.exit(1)

    source_path = str(Path(args.source).resolve())

    valid, msg = validate_source_path(source_path, manifest)
    if not valid:
        log.error(msg)
        sys.exit(1)

    path = set_source(args.harness, source_path, make_default=args.set_default)

    default_note = f"  default harness = {args.harness}\n" if args.set_default else ""
    log.info(
        "Config saved to %s\n"
        "  sources.%s = %s\n%s\n"
        "Next steps:\n"
        "  claude mcp add torchtalk -s user -- torchtalk mcp-serve\n\n"
        "Verify with:\n"
        "  torchtalk status",
        path,
        args.harness,
        source_path,
        default_note,
    )


def _read_cache_stats(cache_path: Path) -> dict | None:
    """Return the top-level 'stats' dict from a call-graph cache, or None."""
    try:
        with open(cache_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("stats") or None


def _read_coverage_from_cache(cache_path: Path) -> dict[str, int] | None:
    """Return the coverage summary from a call-graph cache, or None on failure."""
    stats = _read_cache_stats(cache_path)
    return (stats or {}).get("coverage") or None


def _format_coverage(cov: dict[str, int]) -> str:
    """Render coverage as 'N ok / M parse_failed / ...' in a stable order."""
    order = ["ok", "parse_failed", "unsupported_language", "filtered"]
    parts = [f"{cov[k]:,} {k}" for k in order if k in cov]
    extras = [f"{v:,} {k}" for k, v in cov.items() if k not in order]
    return " / ".join(parts + extras) if (parts or extras) else "unknown"


def _configured_sources(config: dict) -> dict[str, str]:
    """Harness → source path from config, folding in the legacy pytorch key."""
    sources = dict(config.get("sources", {}))
    legacy = config.get("source", {}).get("pytorch_source")
    if legacy and "pytorch" not in sources:
        sources["pytorch"] = legacy
    return sources


def _harness_status_lines(harness: str, config_source: str) -> tuple[list[str], bool]:
    """Status lines for one configured harness; returns (lines, is_valid)."""
    from datetime import datetime

    from torchtalk.config import cache_paths, validate_source_path
    from torchtalk.harness import get_harness

    lines = [f"Harness: {harness}"]
    try:
        manifest = get_harness(harness).manifest
    except KeyError:
        lines.append(f"  Source:          {config_source} (UNREGISTERED harness)")
        return lines, False

    valid, msg = validate_source_path(config_source, manifest)
    lines.append(
        f"  Source:          {config_source} ({'valid' if valid else 'INVALID'})"
    )
    if not valid:
        lines.append(f"                   {msg}")
        return lines, False

    paths = cache_paths(config_source, package=manifest.package)
    for key, label in [("bindings", "Bindings index"), ("callgraph", "Call graph")]:
        p = paths[key]
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            lines.append(
                f"  {label + ':':18s} {p.name} "
                f"({size_mb:.1f} MB, {mtime:%Y-%m-%d %H:%M})"
            )
            if key == "callgraph":
                stats = _read_cache_stats(p) or {}
                cov = stats.get("coverage")
                if cov:
                    lines.append(f"  {'TU coverage:':18s} " + _format_coverage(cov))
                idirs = stats.get("include_dirs_count")
                if isinstance(idirs, int) and idirs > 0:
                    lines.append(f"  {'-I dirs tracked:':18s} {idirs}")
        else:
            lines.append(f"  {label + ':':18s} not built")

    compile_cmds = Path(config_source) / "build" / "compile_commands.json"
    if not compile_cmds.exists():
        compile_cmds = Path(config_source) / "compile_commands.json"
    found = "found" if compile_cmds.exists() else "not found (call graph unavailable)"
    lines.append(f"  compile_commands.json: {found}")
    return lines, True


def cmd_status(args):
    """Show TorchTalk configuration and cache status."""
    from torchtalk import __version__
    from torchtalk.config import CACHE_DIR, CONFIG_FILE, default_harness, load_config

    lines = [f"TorchTalk v{__version__}", ""]

    lines.append("Configuration:")
    if not CONFIG_FILE.exists():
        lines.append(f"  Config file:     {CONFIG_FILE} (not found)")
        lines.append("")
        lines.append("Run 'torchtalk init --source /path/to/pytorch' to configure.")
        print("\n".join(lines))
        sys.exit(1)

    lines.append(f"  Config file:     {CONFIG_FILE}")
    lines.append(f"  Cache directory: {CACHE_DIR}")
    default = default_harness()
    if default:
        lines.append(f"  Default harness: {default}")

    # Show raw config values for diagnostics, even if stale
    sources = _configured_sources(load_config())
    if not sources:
        lines.append("  Sources:         not configured")
        lines.append("")
        lines.append("Run 'torchtalk init --source /path/to/pytorch' to configure.")
        print("\n".join(lines))
        sys.exit(1)

    all_valid = True
    for harness in sorted(sources):
        lines.append("")
        harness_lines, valid = _harness_status_lines(harness, sources[harness])
        lines.extend(harness_lines)
        all_valid = all_valid and valid

    lines.append("")
    if not all_valid:
        lines.append("Status: NOT READY (fix the entries above or re-run init)")
        print("\n".join(lines))
        sys.exit(1)
    lines.append("Status: Ready")
    print("\n".join(lines))


def cmd_mcp_serve(args):
    """Start MCP server for Claude Code integration."""
    from torchtalk.formatting import set_formatter_mode
    from torchtalk.server import run_server

    if not _activate_harness(args.harness):
        return 1
    set_formatter_mode(args.format)
    run_server(
        pytorch_source=args.source,
        index_path=args.index,
        transport=args.transport,
    )
    return 0


def _apply_repo_manifest(source: str, explicit_harness: str | None) -> None:
    """Prefer a `.torchtalk.toml` shipped in the checkout unless --harness was given."""
    from torchtalk.harness import ManifestError, activate_repo_manifest

    if explicit_harness:
        return
    try:
        name = activate_repo_manifest(source)
    except ManifestError as e:
        log.warning("Ignoring repo manifest: %s", e)
        return
    if name:
        log.info("Using repo-local manifest .torchtalk.toml (harness %r)", name)


def cmd_index_build(args):
    """Build or refresh the index for a source checkout and exit."""
    from torchtalk.config import resolve_source
    from torchtalk.indexer import build_index

    if not _activate_harness(args.harness):
        return 1
    source = resolve_source(cli_flag=args.source)
    if not source:
        log.error("No source configured. Run 'torchtalk init' first.")
        return 1
    _apply_repo_manifest(source, args.harness)

    stats = build_index(source, wait_for_cpp=not args.no_wait)

    print(f"Index built for {source}")
    print(f"  Bindings:         {stats['bindings']:,}")
    print(f"  CUDA kernels:     {stats['cuda_kernels']:,}")
    print(f"  Native functions: {stats['native_functions']:,}")
    print(f"  Python modules:   {stats['python_modules']:,}")
    print(f"  nn.Module classes:{stats['nn_modules']:>8,}")
    print(f"  Test files:       {stats['test_files']:,}")
    if stats["call_graph_building"]:
        print("  C++ call graph:   building in background")
    else:
        print(f"  C++ call graph:   {stats['call_graph_functions']:,} functions")
    return 0


def cmd_index_update(args):
    """Incrementally refresh the bindings index using a snapshot as baseline."""
    from torchtalk.config import resolve_source
    from torchtalk.indexer import update_index
    from torchtalk.snapshots import SnapshotError

    if not _activate_harness(args.harness):
        return 1
    source = resolve_source(cli_flag=args.source)
    if not source:
        log.error("No source configured. Run 'torchtalk init' first.")
        return 1
    _apply_repo_manifest(source, args.harness)

    try:
        stats = update_index(source, since=args.since, on_uncovered=args.on_uncovered)
    except (SnapshotError, RuntimeError, ValueError) as e:
        log.error(str(e))
        return 1

    print(f"Index updated for {source}")
    print(f"  Baseline snapshot: {stats['baseline_snapshot']}")
    print(f"  Baseline commit:   {stats['baseline_commit']}")
    print(f"  C++ files changed: {stats['cpp_files_changed']}")
    print(f"  C++ files removed: {stats['cpp_files_removed']}")
    print(f"  Headers changed:   {stats['headers_changed']}")
    print(f"  YAML re-parsed:    {stats['yaml_changed']}")
    print(f"  Bindings total:    {stats['bindings_total']:,}")
    print(f"  CUDA kernels:      {stats['cuda_kernels_total']:,}")

    cg = stats.get("call_graph") or {}
    if "skipped" in cg:
        print(f"  C++ call graph:    skipped ({cg['skipped']})")
    else:
        affected = cg.get("header_affected_tus", 0)
        widened = cg.get("widened_tus", 0)
        extra = f", {widened} from widen" if widened else ""
        print(
            f"  C++ call graph:    "
            f"{cg.get('files_updated', 0)} updated "
            f"({affected} from headers{extra}), "
            f"{cg.get('files_removed', 0)} removed, "
            f"{cg.get('total_functions', 0):,} functions total"
        )

    uncovered = cg.get("uncovered_headers", 0)
    if uncovered:
        sample = cg.get("uncovered_sample", [])
        policy = cg.get("on_uncovered", "warn")
        if policy == "widen":
            print(
                f"\nNOTE: {uncovered} changed header(s) not in baseline coverage; "
                f"widen policy added {cg.get('widened_tus', 0)} TU(s) to reparse."
            )
        else:
            print(
                f"\nWARNING: {uncovered} changed header(s) not in baseline coverage.\n"
                "  Their affected TUs can't be resolved from the recorded include "
                "graph. Likely causes:\n"
                "  - baseline had parse failures on TUs that include them\n"
                "  - generated headers added since baseline\n"
                "  - truly unused headers (safe to ignore)\n"
                "  Run 'torchtalk index build' for a clean rebuild if unsure."
            )
        for h in sample:
            print(f"    {h}")
        if uncovered > len(sample):
            print(f"    ... and {uncovered - len(sample)} more")
    if cg.get("uncovered_fail"):
        return 1
    return 0


def cmd_snapshot_save(args):
    """Save current cache as a named snapshot."""
    from torchtalk.snapshots import SnapshotError, save_snapshot

    if not _activate_harness(args.harness):
        return 1
    try:
        manifest = save_snapshot(args.name)
    except SnapshotError as e:
        log.error(str(e))
        return 1

    total_mb = (manifest.bindings_size + manifest.callgraph_size) / (1024 * 1024)
    print(f"Saved snapshot '{manifest.name}' ({total_mb:.1f} MB)")
    print(f"  PyTorch source: {manifest.pytorch_source}")
    if manifest.git_commit:
        print(f"  Git commit:     {manifest.git_commit}")
    return 0


def cmd_snapshot_load(args):
    """Load a named snapshot into the active cache."""
    from torchtalk.snapshots import SnapshotError, find_nearest_snapshot, load_snapshot

    if not _activate_harness(args.harness):
        return 1
    name = args.name
    force = args.force

    if args.nearest:
        if name:
            log.error("Cannot combine --nearest with a snapshot name")
            return 1
        picked = find_nearest_snapshot()
        if picked is None:
            log.error(
                "No snapshot matches the current source's commit. "
                "Run 'torchtalk snapshot list' to see available snapshots."
            )
            return 1
        print(f"Selected '{picked.name}' (commit {picked.git_commit})")
        name = picked.name
        force = True

    if not name:
        log.error("Provide a snapshot name or use --nearest")
        return 1

    try:
        manifest = load_snapshot(name, force=force)
    except SnapshotError as e:
        log.error(str(e))
        return 1

    print(f"Loaded snapshot '{manifest.name}' into active cache")
    print(f"  Created:        {manifest.created}")
    print(f"  PyTorch source: {manifest.pytorch_source}")
    if manifest.git_commit:
        print(f"  Git commit:     {manifest.git_commit}")
    return 0


def cmd_snapshot_list(args):
    """List available snapshots."""
    from torchtalk.snapshots import list_snapshots

    snapshots = list_snapshots()
    if not snapshots:
        print("No snapshots found.")
        return 0

    name_w = max(20, max(len(m.name) for m in snapshots) + 2)
    print(f"{'Name':<{name_w}} {'Created':<12} {'Size':>8}  {'Commit':<12}")
    print("-" * (name_w + 36))
    for m in snapshots:
        total_mb = (m.bindings_size + m.callgraph_size) / (1024 * 1024)
        created = m.created.split("T")[0]
        commit = m.git_commit or "-"
        print(f"{m.name:<{name_w}} {created:<12} {total_mb:>6.1f}M  {commit:<12}")
    return 0


def cmd_snapshot_delete(args):
    """Delete a named snapshot."""
    from torchtalk.snapshots import SnapshotError, delete_snapshot

    try:
        delete_snapshot(args.name)
    except SnapshotError as e:
        log.error(str(e))
        return 1

    print(f"Deleted snapshot '{args.name}'")
    return 0


def cmd_snapshot_export(args):
    """Package a snapshot into a gzipped tarball for CI artifact upload."""
    from torchtalk.snapshots import SnapshotError, export_snapshot

    output = Path(args.output) if args.output else Path(f"{args.name}.tar.gz")
    try:
        written = export_snapshot(args.name, output)
    except SnapshotError as e:
        log.error(str(e))
        return 1

    size_mb = written.stat().st_size / (1024 * 1024)
    print(f"Exported '{args.name}' to {written} ({size_mb:.1f} MB)")
    return 0


def cmd_snapshot_import(args):
    """Extract a snapshot tarball into the snapshots directory."""
    from torchtalk.snapshots import SnapshotError, import_snapshot

    try:
        manifest = import_snapshot(Path(args.archive), name=args.name)
    except SnapshotError as e:
        log.error(str(e))
        return 1

    print(f"Imported snapshot '{manifest.name}' from {args.archive}")
    print(f"  PyTorch source: {manifest.pytorch_source}")
    if manifest.git_commit:
        print(f"  Git commit:     {manifest.git_commit}")
    return 0


def _print_file_section(label: str, paths: list[str], sigil: str, limit: int = 20):
    """Print a labeled file-path section if non-empty, with truncation."""
    if not paths:
        return
    print(f"\n{label} ({len(paths)}):")
    for p in paths[:limit]:
        print(f"  {sigil} {p}")
    if len(paths) > limit:
        print(f"  ... and {len(paths) - limit} more")


def cmd_snapshot_diff(args):
    """Show structural differences between two snapshots."""
    from dataclasses import asdict

    from torchtalk.snapshots import SnapshotError, diff_snapshots

    try:
        d = diff_snapshots(args.left, args.right)
    except SnapshotError as e:
        log.error(str(e))
        return 1

    if args.json:
        print(json.dumps(asdict(d), indent=2))
        return 0

    if d.is_empty():
        print(f"No differences between '{d.left}' and '{d.right}'.")
        return 0

    print(f"Diff: '{d.left}' -> '{d.right}'")
    _print_file_section("Files added", d.files_added, "+")
    _print_file_section("Files removed", d.files_removed, "-")
    _print_file_section("Files modified", d.files_modified, "~")

    if d.bindings_added or d.bindings_removed:
        print(f"\nBindings: +{d.bindings_added} -{d.bindings_removed}")

    if d.dispatch_keys_added or d.dispatch_keys_removed:
        added = ", ".join(d.dispatch_keys_added) if d.dispatch_keys_added else "-"
        removed = ", ".join(d.dispatch_keys_removed) if d.dispatch_keys_removed else "-"
        print(f"Dispatch keys added:   {added}")
        print(f"Dispatch keys removed: {removed}")
    return 0


def cmd_cursor_add(args):
    """Add torchtalk MCP server to Cursor and copy .claude/ to project .cursor/."""
    MCP_CONFIG_NAME = "mcp.json"
    project_root = Path(args.project_dir).resolve()
    if not project_root.is_dir():
        log.error("Project directory is not a directory: %s", project_root)
        return 1

    cursor_dir = project_root / ".cursor"
    cursor_dir.mkdir(parents=True, exist_ok=True)

    # Copy torchtalk's .claude/ contents into project_dir/.cursor/
    pkg_root = Path(__file__).resolve().parent.parent.parent
    claude_src = pkg_root / ".claude"
    if claude_src.is_dir():
        for item in claude_src.iterdir():
            dest = cursor_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        print(f"Copied .claude/ to {cursor_dir}")
    else:
        log.error("Setup failed: no .claude/ directory found at %s", claude_src)
        return 1

    # MCP config: project or user
    if args.global_config:
        config_dir = Path.home() / ".cursor"
        config_path = config_dir / MCP_CONFIG_NAME
    else:
        config_path = cursor_dir / MCP_CONFIG_NAME

    source = Path(args.source).resolve()
    if not source.is_dir():
        log.error("Source path is not a directory: %s", source)
        return 1

    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.error("Could not read %s: %s", config_path, e)
            return 1
    else:
        data = {}

    if "mcpServers" not in data:
        data["mcpServers"] = {}

    data["mcpServers"]["torchtalk"] = {
        "command": "torchtalk",
        "args": ["mcp-serve", "--source", str(source)],
    }

    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    try:
        config_path.write_text(json.dumps(data, indent=2) + "\n")
    except OSError as e:
        log.error("Could not write %s: %s", config_path, e)
        return 1

    scope = "user config" if args.global_config else "project"
    print(f"Added torchtalk to {scope} at {config_path}")
    print("Restart Cursor (or reload the window) to load the MCP server.")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="torchtalk",
        description="TorchTalk - Cross-language binding analysis for PyTorch codebases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
  torchtalk init --source /path/to/pytorch
  claude mcp add torchtalk -s user -- torchtalk mcp-serve

Verify setup:
  torchtalk status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init command
    parser_init = subparsers.add_parser(
        "init",
        help="Configure TorchTalk with a source path for a harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Saves configuration to ~/.config/torchtalk/config.toml.
Safe to re-run -- updates existing config without data loss.
Run once per harness to configure multiple repos side by side.

Examples:
  torchtalk init --source /myworkspace/pytorch
  torchtalk init --harness vllm --source /myworkspace/vllm --set-default
        """,
    )
    parser_init.add_argument(
        "--source",
        "--pytorch-source",
        "-p",
        dest="source",
        required=True,
        help="Path to the source checkout (--pytorch-source is a deprecated alias)",
    )
    parser_init.add_argument(
        "--harness",
        default="pytorch",
        help="Harness the source belongs to (default: pytorch)",
    )
    parser_init.add_argument(
        "--set-default",
        action="store_true",
        help="Make this harness the default when --harness is omitted",
    )
    parser_init.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    parser_init.set_defaults(func=cmd_init)

    # status command
    subparsers.add_parser(
        "status",
        help="Show configuration and cache status",
    ).set_defaults(func=cmd_status)

    # index command: headless build-and-exit
    parser_index = subparsers.add_parser(
        "index",
        help="Build or refresh the index without starting the MCP server",
    )
    index_sub = parser_index.add_subparsers(dest="index_cmd", required=True)
    p_build = index_sub.add_parser("build", help="Build or refresh the index and exit")
    p_build.add_argument(
        "--source",
        "--pytorch-source",
        "-p",
        dest="source",
        help="Override source path (default: from config)",
    )
    p_build.add_argument(
        "--no-wait",
        action="store_true",
        help="Return immediately instead of waiting for the C++ call graph",
    )
    p_build.add_argument(
        "--harness",
        help="Harness to index with (default: from config, else pytorch)",
    )
    p_build.set_defaults(func=cmd_index_build)

    p_update = index_sub.add_parser(
        "update",
        help="Incrementally refresh bindings and call graph "
        "using a snapshot as baseline",
    )
    p_update.add_argument(
        "--since", required=True, help="Baseline snapshot name (e.g. 'nightly')"
    )
    p_update.add_argument(
        "--source",
        "--pytorch-source",
        "-p",
        dest="source",
        help="Override source path (default: from config)",
    )
    p_update.add_argument(
        "--on-uncovered",
        choices=["warn", "fail", "widen"],
        default="warn",
        help=(
            "How to handle changed headers not in baseline's include graph: "
            "warn (default), fail (non-zero exit), "
            "widen (textually grep compile-DB TUs and add to reparse set)"
        ),
    )
    p_update.add_argument(
        "--harness",
        help="Harness to update with (default: from config, else pytorch)",
    )
    p_update.set_defaults(func=cmd_index_update)

    # mcp-serve command
    parser_mcp = subparsers.add_parser(
        "mcp-serve",
        help="Start MCP server for Claude Code integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Reads the harness's source from config. Override with --source flag.
First run builds the index (a few minutes); subsequent runs use cache.
        """,
    )
    parser_mcp.add_argument(
        "--source",
        "--pytorch-source",
        "-p",
        dest="source",
        help="Override source path (default: from config)",
    )
    parser_mcp.add_argument(
        "--index", "-i", help="Path to existing bindings.json or index directory"
    )
    parser_mcp.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "streamable-http"],
        help="MCP transport type (default: stdio for Claude Code)",
    )
    parser_mcp.add_argument(
        "--harness",
        help="Harness to index with (default: from config, else pytorch)",
    )
    parser_mcp.add_argument(
        "--format",
        default="compact",
        choices=["compact", "markdown"],
        help="Tool response format (default: compact)",
    )
    parser_mcp.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser_mcp.set_defaults(func=cmd_mcp_serve)

    # snapshot: manage named cache snapshots
    parser_snap = subparsers.add_parser(
        "snapshot",
        help="Save, load, list, delete, or diff named snapshots of the cache",
    )
    snap_sub = parser_snap.add_subparsers(dest="snapshot_cmd", required=True)

    p_save = snap_sub.add_parser("save", help="Save current cache as a snapshot")
    p_save.add_argument(
        "name",
        help="Name (letters, digits, . - _ ; max 64 chars)",
    )
    p_save.add_argument(
        "--harness",
        help="Harness whose cache to snapshot (default: from config, else pytorch)",
    )
    p_save.set_defaults(func=cmd_snapshot_save)

    p_load = snap_sub.add_parser("load", help="Restore a snapshot into active cache")
    p_load.add_argument(
        "name", nargs="?", help="Snapshot name to load (omit when using --nearest)"
    )
    p_load.add_argument(
        "--force",
        action="store_true",
        help="Load even if the snapshot's source fingerprint does not match",
    )
    p_load.add_argument(
        "--nearest",
        action="store_true",
        help="Auto-pick the snapshot whose git commit best matches current HEAD",
    )
    p_load.add_argument(
        "--harness",
        help="Harness whose cache to restore into (default: from config, else pytorch)",
    )
    p_load.set_defaults(func=cmd_snapshot_load)

    snap_sub.add_parser("list", help="List available snapshots").set_defaults(
        func=cmd_snapshot_list
    )

    p_del = snap_sub.add_parser("delete", help="Delete a snapshot")
    p_del.add_argument("name", help="Snapshot name to delete")
    p_del.set_defaults(func=cmd_snapshot_delete)

    p_diff = snap_sub.add_parser("diff", help="Structural diff between two snapshots")
    p_diff.add_argument("left", help="Baseline snapshot name")
    p_diff.add_argument("right", help="Comparison snapshot name")
    p_diff.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable output",
    )
    p_diff.set_defaults(func=cmd_snapshot_diff)

    p_export = snap_sub.add_parser(
        "export", help="Package a snapshot into a .tar.gz (for CI artifacts)"
    )
    p_export.add_argument("name", help="Snapshot name to export")
    p_export.add_argument(
        "--output",
        "-o",
        help="Output tarball path (default: <name>.tar.gz in cwd)",
    )
    p_export.set_defaults(func=cmd_snapshot_export)

    p_import = snap_sub.add_parser(
        "import", help="Extract a snapshot tarball into the snapshots directory"
    )
    p_import.add_argument("archive", help="Path to the .tar.gz archive")
    p_import.add_argument(
        "--name",
        help="Rename the snapshot on import (default: use the name from the archive)",
    )
    p_import.set_defaults(func=cmd_snapshot_import)

    # Cursor: add MCP to .cursor/mcp.json
    parser_cursor = subparsers.add_parser(
        "cursor-add",
        help="Add torchtalk MCP to Cursor's .cursor/mcp.json",
    )
    parser_cursor.add_argument(
        "--project-dir",
        "-C",
        required=True,
        help="Project root: .cursor/mcp.json and .claude contents are written here",
    )
    parser_cursor.add_argument(
        "--source",
        "--pytorch-source",
        "-p",
        dest="source",
        required=True,
        help="Path to the source checkout (--pytorch-source is a deprecated alias)",
    )
    parser_cursor.add_argument(
        "--global",
        dest="global_config",
        action="store_true",
        help="Write to ~/.cursor/mcp.json (user-wide)",
    )
    parser_cursor.set_defaults(func=cmd_cursor_add)

    args = parser.parse_args()

    log_level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s"
        if log_level == logging.INFO
        else "%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
