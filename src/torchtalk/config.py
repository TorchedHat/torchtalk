"""Persistent configuration for TorchTalk.

Manages user config at ~/.config/torchtalk/config.toml (XDG-compliant)
and cache at ~/.cache/torchtalk/.

Resolution order for pytorch_source:
  1. --pytorch-source CLI flag (highest priority)
  2. PYTORCH_SOURCE environment variable
  3. ~/.config/torchtalk/config.toml
"""

import logging
import os
import sys
from pathlib import Path

from .adapters.base import DEFAULT_FRAMEWORK, FrameworkId

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

_FRAMEWORK_CONFIG_KEYS = {
    "pytorch": "pytorch_source",
    "vllm": "vllm_source",
}
_FRAMEWORK_ENV_VARS = {
    "pytorch": ("PYTORCH_SOURCE", "PYTORCH_PATH"),
    "vllm": ("VLLM_SOURCE",),
}


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


def _normalize_framework(framework: FrameworkId | None) -> FrameworkId:
    return (framework or DEFAULT_FRAMEWORK).lower()


def _resolve_source_from_config(
    cli_flag: str | None,
    *,
    env_vars: tuple[str, ...],
    config_key: str,
) -> str | None:
    """Resolve a source path with CLI, env, then config precedence."""

    if cli_flag:
        return cli_flag

    for env_var in env_vars:
        env_val = os.environ.get(env_var)
        if env_val and Path(env_val).exists():
            return env_val

    config = load_config()
    config_val = config.get("source", {}).get(config_key)
    if config_val and Path(config_val).exists():
        return config_val

    return None


def resolve_framework_source(
    framework: FrameworkId = DEFAULT_FRAMEWORK,
    cli_flag: str | None = None,
) -> str | None:
    """Resolve a source path for the requested framework."""

    normalized = _normalize_framework(framework)
    config_key = _FRAMEWORK_CONFIG_KEYS.get(normalized)
    env_vars = _FRAMEWORK_ENV_VARS.get(normalized)
    if config_key is None or env_vars is None:
        raise ValueError(f"Unsupported framework: {framework}")
    return _resolve_source_from_config(
        cli_flag,
        env_vars=env_vars,
        config_key=config_key,
    )


def resolve_pytorch_source(cli_flag: str | None = None) -> str | None:
    """Resolve PyTorch source path using 3-level priority.

    1. cli_flag (--pytorch-source)
    2. PYTORCH_SOURCE env var
    3. config.toml [source] pytorch_source
    """

    return resolve_framework_source("pytorch", cli_flag)


def source_hash(source: str | Path) -> str:
    """Compute a stable hash for a PyTorch source directory.

    Used as a cache key suffix to distinguish indexes built from
    different source checkouts.
    """
    import hashlib

    return hashlib.md5(str(Path(source).resolve()).encode()).hexdigest()[:12]


def cache_paths(source: str | Path) -> dict[str, Path]:
    """Return the canonical cache file paths for a given source directory.

    Keys:
        bindings  - Binding index JSON
        callgraph - C++ call graph JSON
    """
    h = source_hash(source)
    return {
        "bindings": CACHE_DIR / f"bindings_{h}.json",
        "callgraph": CACHE_DIR / "call_graph" / f"pytorch_callgraph_parallel_{h}.json",
        "test_infra": CACHE_DIR / f"test_infra_{h}.json",
        "py_cpp_edges": CACHE_DIR / f"py_cpp_edges_{h}.json",
    }


def framework_cache_path(
    source: str | Path,
    framework: FrameworkId,
    artifact: str = "index",
) -> Path:
    """Return a framework-specific cache artifact path.

    PyTorch keeps its legacy cache layout via `cache_paths`; this helper is for
    non-PyTorch adapters and new generic artifacts.
    """

    h = source_hash(source)
    normalized = _normalize_framework(framework)
    return CACHE_DIR / f"{normalized}_{artifact}_{h}.json"


def _validate_pytorch_path(path: str | Path) -> tuple[bool, str]:
    """Validate that a path looks like a PyTorch source checkout."""

    p = Path(path)
    if not p.exists():
        return False, f"Path does not exist: {p}"
    if not p.is_dir():
        return False, f"Path is not a directory: {p}"
    if not (p / "torch").exists():
        return False, f"No 'torch/' directory found in {p} (not a PyTorch checkout?)"
    nf = p / "aten" / "src" / "ATen" / "native" / "native_functions.yaml"
    if not nf.exists():
        return (
            False,
            f"native_functions.yaml not found in {p} (required for operator indexing)",
        )
    return True, f"Valid PyTorch source: {p}"


def _validate_vllm_path(path: str | Path) -> tuple[bool, str]:
    """Validate that a path looks like a vLLM source checkout."""

    p = Path(path)
    if not p.exists():
        return False, f"Path does not exist: {p}"
    if not p.is_dir():
        return False, f"Path is not a directory: {p}"
    if not (p / "vllm").exists():
        return False, f"No 'vllm/' package found in {p} (not a vLLM checkout?)"

    required_paths = [
        p / "pyproject.toml",
        p / "vllm" / "entrypoints" / "llm.py",
        p / "vllm" / "v1" / "engine" / "llm_engine.py",
        p / "vllm" / "model_executor" / "models" / "registry.py",
    ]
    missing = [
        path.relative_to(p).as_posix()
        for path in required_paths
        if not path.exists()
    ]
    if missing:
        return (
            False,
            f"Missing required vLLM markers in {p}: {', '.join(missing)}",
        )
    return True, f"Valid vLLM source: {p}"


def validate_framework_path(
    framework: FrameworkId = DEFAULT_FRAMEWORK,
    path: str | Path = "",
) -> tuple[bool, str]:
    """Validate that a path looks usable for the requested framework."""

    normalized = _normalize_framework(framework)
    if normalized == "pytorch":
        return _validate_pytorch_path(path)
    if normalized == "vllm":
        return _validate_vllm_path(path)
    raise ValueError(f"Unsupported framework: {framework}")


def validate_pytorch_path(path: str | Path) -> tuple[bool, str]:
    """Validate that a path looks like a PyTorch source checkout.

    Returns (is_valid, message).
    """

    return validate_framework_path("pytorch", path)
