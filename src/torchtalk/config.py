"""Persistent configuration for TorchTalk.

Manages user config at ~/.config/torchtalk/config.toml (XDG-compliant)
and cache at ~/.cache/torchtalk/.

Source paths are configured per harness under [sources] (keyed by harness
name), with [defaults] harness selecting the harness used when --harness is
omitted. Resolution order for a harness's source:
  1. --source CLI flag (highest priority)
  2. TORCHTALK_SOURCE_<HARNESS> environment variable
     (plus PYTORCH_SOURCE / PYTORCH_PATH for the pytorch harness)
  3. [sources].<harness> in config.toml
     (plus the legacy [source] pytorch_source key for the pytorch harness)
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .harness import ConventionManifest

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore[assignment]

try:
    import tomli_w
except ImportError:
    tomli_w = None  # type: ignore[assignment]

from platformdirs import user_cache_path, user_config_path

log = logging.getLogger(__name__)

CONFIG_DIR = user_config_path("torchtalk")
CONFIG_FILE = CONFIG_DIR / "config.toml"
CACHE_DIR = user_cache_path("torchtalk")


def load_config() -> dict:
    """Load config from ~/.config/torchtalk/config.toml.

    Returns empty dict if file doesn't exist or can't be parsed.
    """
    if not CONFIG_FILE.exists():
        return {}

    if tomllib is None:
        log.warning("Cannot read config: tomllib/tomli not available")
        return {}

    try:
        with open(CONFIG_FILE, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        log.warning("Failed to read %s: %s", CONFIG_FILE, e)
        return {}


def save_config(config: dict) -> Path:
    """Write config to ~/.config/torchtalk/config.toml.

    Returns the path written to.
    """
    if tomli_w is None:
        raise RuntimeError(
            "Cannot write config: tomli-w not installed. "
            "Install with: pip install tomli-w"
        )

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "wb") as f:
        tomli_w.dump(config, f)
    return CONFIG_FILE


def source_env_var(harness: str) -> str:
    """Environment variable naming a harness's source override."""
    return "TORCHTALK_SOURCE_" + re.sub(r"[^A-Za-z0-9]", "_", harness).upper()


def resolve_source(
    harness: str | None = None, cli_flag: str | None = None
) -> str | None:
    """Resolve a harness's source path using 3-level priority.

    1. cli_flag (--source)
    2. TORCHTALK_SOURCE_<HARNESS> env var (pytorch also honors the legacy
       PYTORCH_SOURCE / PYTORCH_PATH vars)
    3. config.toml [sources].<harness> (pytorch also honors the legacy
       [source] pytorch_source key)

    `harness` defaults to the active harness.
    """
    if cli_flag:
        return cli_flag

    if harness is None:
        from .harness import active_harness_name

        harness = active_harness_name()

    env_vars = [source_env_var(harness)]
    if harness == "pytorch":
        env_vars += ["PYTORCH_SOURCE", "PYTORCH_PATH"]
    for var in env_vars:
        val = os.environ.get(var)
        if val and Path(val).exists():
            return val

    config = load_config()
    config_val = config.get("sources", {}).get(harness)
    if not config_val and harness == "pytorch":
        config_val = config.get("source", {}).get("pytorch_source")
    if config_val and Path(config_val).exists():
        return config_val

    return None


def default_harness() -> str | None:
    """The configured default harness name, or None when unset."""
    return load_config().get("defaults", {}).get("harness")


def set_source(harness: str, path: str, make_default: bool = False) -> Path:
    """Record a harness's source path in config.toml; returns the path written.

    Mirrors the pytorch entry to the legacy [source] pytorch_source key,
    which older readers (e.g. plugin-setup.sh) grep for.
    """
    config = load_config()
    config.setdefault("sources", {})[harness] = path
    if harness == "pytorch":
        config.setdefault("source", {})["pytorch_source"] = path
    if make_default:
        config.setdefault("defaults", {})["harness"] = harness
    return save_config(config)


def source_hash(source: str | Path) -> str:
    """Compute a stable hash for a PyTorch source directory.

    Used as a cache key suffix to distinguish indexes built from
    different source checkouts.
    """
    import hashlib

    return hashlib.md5(str(Path(source).resolve()).encode()).hexdigest()[:12]


def cache_paths(source: str | Path, package: str | None = None) -> dict[str, Path]:
    """Return the canonical cache file paths for a source directory and package.

    Paths are qualified by both source hash and harness package so indexing
    the same checkout under different harnesses never shares cache files.

    Keys:
        bindings  - Binding index JSON
        callgraph - C++ call graph JSON
    """
    if package is None:
        from .harness import active_manifest

        package = active_manifest().package
    h = source_hash(source)
    callgraph_dir = CACHE_DIR / "call_graph"
    return {
        "bindings": CACHE_DIR / f"bindings_{package}_{h}.json",
        "callgraph": callgraph_dir / f"{package}_callgraph_parallel_{h}.json",
        "test_infra": CACHE_DIR / f"test_infra_{package}_{h}.json",
        "py_cpp_edges": CACHE_DIR / f"py_cpp_edges_{package}_{h}.json",
    }


def validate_source_path(
    path: str | Path, manifest: ConventionManifest
) -> tuple[bool, str]:
    """Validate that a path looks like a checkout of the manifest's package.

    Returns (is_valid, message).
    """
    p = Path(path)
    if not p.exists():
        return False, f"Path does not exist: {p}"
    if not p.is_dir():
        return False, f"Path is not a directory: {p}"
    roots = (*manifest.python_package_roots, *manifest.cpp_search_dirs)
    if roots and not any((p / d).exists() for d in roots):
        return False, (
            f"No {manifest.package} directories ({', '.join(sorted(set(roots)))}) "
            f"found in {p} (not a {manifest.package} checkout?)"
        )
    if manifest.native_functions_yaml:
        nf = p / manifest.native_functions_yaml
        if not nf.exists():
            return (
                False,
                f"{manifest.native_functions_yaml} not found in {p} "
                "(required for operator indexing)",
            )
    return True, f"Valid {manifest.package} source: {p}"
