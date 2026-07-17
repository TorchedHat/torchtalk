"""Package-qualified symbol identity for cross-repo graph merging.

Symbol IDs follow ``<package>@<revision>/<symbol>``, e.g.
``pytorch@a1b2c3d4e5f6/at::native::add``. Revision is the short git HEAD
sha of the source checkout, falling back to a version string.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

UNKNOWN_REVISION = "unknown"

_HEX_SHA_RE = re.compile(r"[0-9a-f]{40,64}")
_VERSION_RE = re.compile(r"__version__\s*=\s*['\"]([^'\"]+)['\"]")


@dataclass(frozen=True)
class PackageIdentity:
    """A source package pinned to a revision (git sha or version string)."""

    name: str
    revision: str

    def __str__(self) -> str:
        return f"{self.name}@{self.revision}"


def make_symbol_id(package: PackageIdentity, symbol: str) -> str:
    """Build a globally unique symbol ID: ``<package>@<revision>/<symbol>``."""
    return f"{package}/{symbol}"


def parse_symbol_id(symbol_id: str) -> tuple[PackageIdentity, str]:
    """Split a symbol ID into (PackageIdentity, symbol); ValueError if malformed."""
    head, slash, symbol = symbol_id.partition("/")
    name, at, revision = head.partition("@")
    if not (slash and at and name and revision and symbol):
        raise ValueError(f"Malformed symbol ID: {symbol_id!r}")
    return PackageIdentity(name, revision), symbol


def detect_package_identity(source: str | Path, name: str) -> PackageIdentity:
    """Derive package identity from a source checkout."""
    src = Path(source)
    revision = _git_head_sha(src) or _version_string(src) or UNKNOWN_REVISION
    return PackageIdentity(name, revision)


def content_fingerprint(source: str | Path) -> str | None:
    """Merkle-style hash over HEAD tree + any uncommitted diff.

    Uses git's own tree hash (a content-addressed Merkle over all tracked files)
    combined with a hash of `git diff HEAD` to cover dirty trees. Two checkouts
    with identical content produce the same fingerprint regardless of path.
    Returns None when source is not a git working tree.
    """
    try:
        tree = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "-C", str(source), "diff", "HEAD"],
            capture_output=True,
            check=True,
            timeout=15,
        ).stdout
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None

    h = hashlib.blake2b(tree.encode(), digest_size=16)
    h.update(diff)
    return h.hexdigest()


def _git_head_sha(src: Path) -> str | None:
    """Short HEAD sha via git, falling back to reading .git/HEAD directly."""
    try:
        result = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    head = src / ".git" / "HEAD"
    if head.is_file():
        text = head.read_text(errors="ignore").strip()
        if _HEX_SHA_RE.fullmatch(text):
            return text[:12]
    return None


def _version_string(src: Path) -> str | None:
    """Version from version.txt or a top-level package's version.py."""
    version_txt = src / "version.txt"
    if version_txt.is_file():
        version = version_txt.read_text(errors="ignore").strip()
        if version:
            return version
    for version_py in sorted(src.glob("*/version.py")):
        match = _VERSION_RE.search(version_py.read_text(errors="ignore"))
        if match:
            return match.group(1)
    return None
