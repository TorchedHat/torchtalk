"""Harness smoke test: validates that a harness config indexes a real repo.

Reads repo/ref/sparse_paths from tests/integration/<target>.yml (single
source of truth). Thresholds come from `expected_minimums` in the harness
manifest (src/torchtalk/manifests/<target>.toml).

Usage:
    python scripts/harness_smoke.py --harness pytorch --source /path/to/pytorch
    python scripts/harness_smoke.py --harness pytorch --clone
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from torchtalk.harness import get_harness, list_harnesses

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = REPO_ROOT / "tests" / "integration"


_STABLE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _load_manifest(target: str) -> dict:
    """Load the integration manifest for a target."""
    path = MANIFESTS_DIR / f"{target}.yml"
    if not path.exists():
        sys.exit(f"Manifest not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _clone_url(repo: str) -> str:
    """Convert owner/repo to a clone URL."""
    return f"https://github.com/{repo}.git"


def latest_stable_tag(repo_url: str) -> str:
    """Resolve the latest stable release tag (excludes -rc, -beta, etc.)."""
    result = subprocess.run(
        ["git", "ls-remote", "--tags", repo_url],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    tags = []
    for line in result.stdout.splitlines():
        ref = line.split("\t")[1].removeprefix("refs/tags/")
        m = _STABLE_TAG.match(ref)
        if m:
            tags.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), ref))
    if not tags:
        raise RuntimeError(f"No stable tags found in {repo_url}")
    tags.sort()
    return tags[-1][3]


def sparse_clone(repo_url: str, ref: str, paths: list[str], dest: Path) -> None:
    subprocess.run(
        [
            "git",
            "clone",
            "--depth=1",
            "--branch",
            ref,
            "--sparse",
            "--no-checkout",
            repo_url,
            str(dest),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", *paths],
        cwd=dest,
        check=True,
    )
    subprocess.run(["git", "checkout"], cwd=dest, check=True)


def main():
    targets = sorted(
        n for n in list_harnesses() if get_harness(n).manifest.expected_minimums
    )
    parser = argparse.ArgumentParser(description="Harness smoke test")
    parser.add_argument("--harness", required=True, choices=targets)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path)
    group.add_argument("--clone", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(args.harness)
    repo_url = _clone_url(manifest["repo"])
    ref = manifest.get("ref") or latest_stable_tag(repo_url)
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
