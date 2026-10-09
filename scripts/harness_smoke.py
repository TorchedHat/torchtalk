"""Harness smoke test: validates that a harness config indexes a real repo.

Reads repo/ref/sparse_paths from tests/integration/<target>.yml (single
source of truth). Thresholds come from `expected_minimums` in the harness
manifest (src/torchtalk/manifests/<target>.toml).

Usage:
    python scripts/harness_smoke.py --harness pytorch --source /path/to/pytorch
    python scripts/harness_smoke.py --harness pytorch --clone
    python scripts/harness_smoke.py --list
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from torchtalk.harness import get_harness
from torchtalk.integration_manifest import (
    IntegrationManifestError,
    integration_manifest_paths,
    load_integration_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = REPO_ROOT / "tests" / "integration"


def _load_manifest(target: str) -> dict:
    """Load the integration manifest for a target."""
    path = MANIFESTS_DIR / f"{target}.yml"
    try:
        return load_integration_manifest(path)
    except IntegrationManifestError as exc:
        sys.exit(str(exc))


def _clone_url(repo: str) -> str:
    """Convert owner/repo to a clone URL."""
    return f"https://github.com/{repo}.git"


def sparse_clone(repo_url: str, ref: str, paths: list[str], dest: Path) -> None:
    subprocess.run(
        [
            "git",
            "clone",
            "--depth=1",
            "--filter=blob:none",
            "--no-checkout",
            repo_url,
            str(dest),
        ],
        check=True,
    )
    is_commit_sha = len(ref) in {40, 64} and all(
        char in "0123456789abcdefABCDEF" for char in ref
    )
    refspec = ref if is_commit_sha else f"refs/tags/{ref}"
    subprocess.run(
        ["git", "fetch", "--depth=1", "origin", refspec], cwd=dest, check=True
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", *paths],
        cwd=dest,
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=dest, check=True)


def smoke_targets() -> list[str]:
    """Integration targets whose harness declares `[expected_minimums]`.

    CI builds the smoke matrix from this, so a target needs both an anchor
    file and thresholds before it is smoke-tested.
    """
    return [
        p.stem
        for p in integration_manifest_paths(MANIFESTS_DIR)
        if get_harness(p.stem).manifest.expected_minimums
    ]


def main():
    targets = smoke_targets()
    parser = argparse.ArgumentParser(description="Harness smoke test")
    parser.add_argument("--harness", choices=targets)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--source", type=Path)
    group.add_argument(
        "--clone",
        action="store_true",
        help="clone the manifest's pinned version tag or commit SHA",
    )
    group.add_argument(
        "--list", action="store_true", help="print the smoke targets as JSON and exit"
    )
    args = parser.parse_args()
    if args.list:
        print(json.dumps(targets))
        return
    if not args.harness or not (args.source or args.clone):
        parser.error("--harness and one of --source/--clone are required")

    manifest = _load_manifest(args.harness)
    repo_url = _clone_url(manifest["repo"])
    ref = manifest["ref"]
    sparse_paths = manifest["sparse_paths"]

    if args.clone:
        print(f"Using ref: {ref}")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "repo"
            sparse_clone(repo_url, ref, sparse_paths, source)
            stats = _run_index(args.harness, source)
    else:
        stats = _run_index(args.harness, args.source)

    thresholds = get_harness(args.harness).manifest.expected_minimums
    failures = []
    for field, threshold in thresholds.items():
        if field not in stats:
            failures.append(f"{field}: not a stat key (have {sorted(stats)})")
            continue
        actual = stats[field]
        print(f"  {field}: {actual} (threshold: >= {threshold})")
        if actual < threshold:
            failures.append(f"{field}: {actual} < {threshold}")

    if failures:
        print(f"\nFAILED: {len(failures)} assertion(s)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\nPassed.")


def _run_index(harness_name: str, source: Path) -> dict:
    from torchtalk.harness import set_active_harness
    from torchtalk.indexer import build_index

    set_active_harness(harness_name)
    return build_index(str(source), wait_for_cpp=False)


if __name__ == "__main__":
    main()
