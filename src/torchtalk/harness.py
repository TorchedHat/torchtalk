"""Repo harnesses: convention manifests describing how to index a package.

A manifest is data, not code: search directories, exclusion patterns,
data-source paths, and cache-identity strategy for one package. Graphs built
from different manifests merge via package-qualified symbol IDs (symbols.py).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass(frozen=True)
class CallRegistration:
    """A registration call: which arg holds the registry key, which the target."""

    call: str
    key_arg: int | str
    target_arg: int | str
    registry: str = ""


@dataclass(frozen=True)
class ConventionManifest:
    """Indexing conventions for one package."""

    package: str
    cpp_search_dirs: tuple[str, ...]
    python_search_dirs: tuple[str, ...] = ()
    # top-level import-package dirs used to derive module names from paths
    python_package_roots: tuple[str, ...] = ()
    test_search_dirs: tuple[str, ...] = ()
    # content markers identifying test files in package-internal test dirs
    test_content_patterns: tuple[str, ...] = ("TestCase", "pytest", "unittest")
    test_utility_modules: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    registration_macros: tuple[str, ...] = ()
    native_functions_yaml: str = ""
    derivatives_yaml: str = ""
    # C++ extractor config: alias macro → canonical macro it expands to
    # (e.g. {"TORCH_LIBRARY_EXPAND": "TORCH_LIBRARY"}), and token → literal
    # substitutions (e.g. {"TORCH_EXTENSION_NAME": "_C"}).
    cpp_macro_aliases: dict[str, str] = field(default_factory=dict)
    cpp_token_map: dict[str, str] = field(default_factory=dict)
    # Python extractor config (analysis/extractors.py):
    # decorator qualname → registry it populates
    decorator_registries: dict[str, str] = field(default_factory=dict)
    # module-level dict variables mapping string keys to import targets
    string_registries: tuple[str, ...] = ()
    registration_calls: tuple[CallRegistration, ...] = ()
    # dispatcher call → arg (position or kwarg) naming the invoked method
    string_dispatchers: dict[str, int | str] = field(default_factory=dict)
    # Python op prefix → C++ op namespace, e.g. `torch.add` ↔ `aten::add`.
    op_namespaces: dict[str, str] = field(default_factory=dict)
    # Repo-relative files scanned for `@register_decomposition(<ns>.X)`.
    decomp_alias_paths: tuple[str, ...] = ()
    # Repo-relative dir whose cpu/cuda subdirs hold REGISTER_*_DISPATCH macros.
    dispatch_stub_root: str = ""
    # Wrappers stripped from `m.impl("op", WRAPPER(fn))` targets.
    cpp_call_wrappers: tuple[str, ...] = ()
    # Harness names this package's symbols resolve against (bridge targets).
    depends_on: tuple[str, ...] = ()
    # [bridge] resolver lists: C++ namespaces owned by a dependency (`at::`),
    # and base-class prefixes marking cross-package subclassing (`torch.nn`).
    cpp_namespaces: tuple[str, ...] = ()
    base_class_namespaces: tuple[str, ...] = ()
    # Smoke-test floors: index stat name → minimum count (scripts/harness_smoke.py).
    expected_minimums: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class Harness(Protocol):
    """A package integration: its manifest (extractor hooks arrive later)."""

    manifest: ConventionManifest


MANIFESTS_DIR = Path(__file__).parent / "manifests"
REPO_MANIFEST_NAME = ".torchtalk.toml"

# Each TOML section key maps to one ConventionManifest field; `cpp.*` keys
# are prefixed to their field names below.
_SECTION_FIELDS: dict[str, dict[str, str]] = {
    "paths": {
        f: f
        for f in (
            "cpp_search_dirs",
            "python_search_dirs",
            "python_package_roots",
            "test_search_dirs",
            "test_content_patterns",
            "test_utility_modules",
            "exclude_patterns",
            "native_functions_yaml",
            "derivatives_yaml",
            "decomp_alias_paths",
            "dispatch_stub_root",
        )
    },
    "cpp": {
        "registration_macros": "registration_macros",
        "macro_aliases": "cpp_macro_aliases",
        "token_map": "cpp_token_map",
        "call_wrappers": "cpp_call_wrappers",
    },
    "python": {
        f: f
        for f in (
            "decorator_registries",
            "string_registries",
            "registration_calls",
            "string_dispatchers",
            "op_namespaces",
        )
    },
    "bridge": {
        "cpp_namespaces": "cpp_namespaces",
        "base_class_namespaces": "base_class_namespaces",
    },
}
_FIELD_TYPES = {f.name: f.type for f in fields(ConventionManifest)}
_PACKAGE_KEYS = {"name", "extends", "depends_on"}
_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ManifestError(ValueError):
    """A manifest TOML file is malformed or references an unknown profile."""


def _flatten(data: dict[str, Any], origin: str) -> dict[str, Any]:
    """Map TOML sections onto ConventionManifest field names."""
    out: dict[str, Any] = {}
    pkg = data.get("package", {})
    if not isinstance(pkg, dict):
        raise ManifestError(f"{origin}: [package] must be a table")
    for key in pkg:
        if key not in _PACKAGE_KEYS:
            raise ManifestError(f"{origin}: unknown key [package] {key}")
    if "name" in pkg:
        name = pkg["name"]
        if not isinstance(name, str) or not _PACKAGE_NAME_RE.match(name):
            raise ManifestError(
                f"{origin}: [package] name must match [A-Za-z0-9_.-]+, got {name!r}"
            )
        out["package"] = name
    if "depends_on" in pkg:
        out["depends_on"] = pkg["depends_on"]
    for section, mapping in _SECTION_FIELDS.items():
        table = data.get(section, {})
        if not isinstance(table, dict):
            raise ManifestError(f"{origin}: [{section}] must be a table")
        for key, value in table.items():
            if key not in mapping:
                raise ManifestError(f"{origin}: unknown key [{section}] {key}")
            out[mapping[key]] = value
    if "expected_minimums" in data:
        out["expected_minimums"] = data["expected_minimums"]
    known = {"package", "paths", "cpp", "python", "bridge", "expected_minimums"}
    for section in data:
        if section not in known:
            raise ManifestError(f"{origin}: unknown section [{section}]")
    return out


def _coerce(name: str, value: Any, origin: str = "<dict>") -> Any:
    """Convert TOML values to the field's dataclass type (lists → tuples).

    Raises ManifestError when the TOML value has the wrong shape, so a typo
    like `cpp_search_dirs = "csrc"` fails at load time rather than iterating
    over characters deep inside the indexer.
    """
    ftype = str(_FIELD_TYPES[name])
    if name == "registration_calls":
        if not isinstance(value, list) or not all(isinstance(r, dict) for r in value):
            raise ManifestError(
                f"{origin}: registration_calls must be a list of tables"
            )
        out = []
        for r in value:
            missing = [k for k in ("call", "key_arg", "target_arg") if k not in r]
            if missing:
                raise ManifestError(
                    f"{origin}: registration_calls entry missing {missing}: {r}"
                )
            out.append(
                CallRegistration(
                    r["call"], r["key_arg"], r["target_arg"], r.get("registry", "")
                )
            )
        return tuple(out)
    if name == "expected_minimums":
        if not isinstance(value, dict) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in value.values()
        ):
            raise ManifestError(
                f"{origin}: [expected_minimums] values must be integers"
            )
        return dict(value)
    if ftype.startswith("tuple"):
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ManifestError(f"{origin}: {name} must be a list of strings")
        return tuple(value)
    if ftype == "str":
        if not isinstance(value, str):
            raise ManifestError(f"{origin}: {name} must be a string")
        return value
    if ftype.startswith("dict"):
        if not isinstance(value, dict) or not all(
            isinstance(v, (str, int)) and not isinstance(v, bool)
            for v in value.values()
        ):
            raise ManifestError(
                f"{origin}: {name} must be a table of strings or integers"
            )
        return dict(value)
    return value


def manifest_from_dict(
    data: dict[str, Any],
    *,
    base: ConventionManifest | None = None,
    origin: str = "<dict>",
) -> ConventionManifest:
    """Build a manifest from parsed TOML, layered over `base` when given.

    Every key set in `data` replaces the base value outright (lists and
    tables are not merged), so a profile can drop a base default by
    setting it to an empty value.
    """
    values: dict[str, Any] = {}
    if base is not None:
        values.update({f.name: getattr(base, f.name) for f in fields(base)})
    for name, raw in _flatten(data, origin).items():
        values[name] = _coerce(name, raw, origin)
    if "package" not in values:
        raise ManifestError(f"{origin}: [package] name is required")
    if "cpp_search_dirs" not in values:
        raise ManifestError(f"{origin}: [paths] cpp_search_dirs is required")
    return ConventionManifest(**values)


def _read_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        try:
            return tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ManifestError(f"{path}: {e}") from e


def load_manifest(path: str | Path, _seen: tuple[str, ...] = ()) -> ConventionManifest:
    """Load a manifest TOML file, resolving `[package] extends` recursively.

    `extends` names a built-in profile in `torchtalk/manifests/` (e.g.
    "torch-extension" or "pytorch") or a path relative to the file itself.
    """
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"Manifest not found: {path}")
    data = _read_toml(path)
    origin = str(path)
    if origin in _seen:
        raise ManifestError(f"{origin}: circular extends chain {(*_seen, origin)}")
    base = None
    parent = data.get("package", {}).get("extends")
    if parent:
        builtin = MANIFESTS_DIR / f"{parent}.toml"
        candidate = builtin if builtin.exists() else path.parent / parent
        if not candidate.exists():
            raise ManifestError(
                f"{origin}: extends {parent!r} not found "
                f"(built-ins: {builtin_manifest_names()})"
            )
        base = load_manifest(candidate, (*_seen, origin))
    return manifest_from_dict(data, base=base, origin=origin)


def load_builtin_manifest(name: str) -> ConventionManifest:
    """Load `torchtalk/manifests/<name>.toml`."""
    return load_manifest(MANIFESTS_DIR / f"{name}.toml")


def builtin_manifest_names() -> list[str]:
    """Names of shipped manifest profiles (including abstract bases)."""
    return sorted(p.stem for p in MANIFESTS_DIR.glob("*.toml"))


def find_repo_manifest(source: str | Path) -> Path | None:
    """Return `<source>/.torchtalk.toml` when the checkout ships one."""
    path = Path(source) / REPO_MANIFEST_NAME
    return path if path.is_file() else None


PYTORCH_MANIFEST = load_builtin_manifest("pytorch")


@dataclass(frozen=True)
class ManifestHarness:
    """Harness defined entirely by manifest data."""

    manifest: ConventionManifest


PYTORCH_HARNESS = ManifestHarness(PYTORCH_MANIFEST)


VLLM_MANIFEST = load_builtin_manifest("vllm")
TORCHVISION_MANIFEST = load_builtin_manifest("torchvision")

_REGISTRY: dict[str, Harness] = {}
_ACTIVE = "pytorch"


def register_harness(name: str, harness: Harness) -> None:
    """Register a harness under a package name."""
    _REGISTRY[name] = harness


def list_harnesses() -> list[str]:
    """Names of all registered harnesses."""
    return sorted(_REGISTRY)


def _ensure_registered(name: str) -> None:
    """Lazily register a shipped `manifests/<name>.toml` on first use.

    Lets `--harness myfw` work for a profile that exists only as a TOML file
    (e.g. one dropped into `manifests/` by a contributor) without touching
    the hardcoded registrations below. Raises KeyError when no such profile
    exists; a malformed profile raises ManifestError.
    """
    if name in _REGISTRY:
        return
    path = MANIFESTS_DIR / f"{name}.toml"
    if not path.is_file():
        raise KeyError(
            f"Unknown harness {name!r}. Registered: {sorted(_REGISTRY)}; "
            f"built-in manifests: {builtin_manifest_names()}"
        )
    register_harness(name, ManifestHarness(load_manifest(path)))


def get_harness(name: str | None = None) -> Harness:
    """Return the named harness, or the active one when no name is given."""
    key = name or _ACTIVE
    _ensure_registered(key)
    return _REGISTRY[key]


def set_active_harness(name: str) -> None:
    """Select the harness used when get_harness() is called without a name."""
    global _ACTIVE
    _ensure_registered(name)
    _ACTIVE = name


def active_harness_name() -> str:
    """Name of the harness used when get_harness() is called without a name."""
    return _ACTIVE


def active_manifest() -> ConventionManifest:
    return get_harness().manifest


def activate_repo_manifest(source: str | Path) -> str | None:
    """Register and activate `<source>/.torchtalk.toml` if present.

    Returns the harness name activated, or None when the checkout has no
    repo-local manifest. Re-registering an existing name replaces it, so a
    checkout can override a shipped profile of the same package.
    """
    path = find_repo_manifest(source)
    if path is None:
        return None
    manifest = load_manifest(path)
    register_harness(manifest.package, ManifestHarness(manifest))
    set_active_harness(manifest.package)
    return manifest.package


register_harness("pytorch", PYTORCH_HARNESS)
register_harness("vllm", ManifestHarness(VLLM_MANIFEST))
register_harness("torchvision", ManifestHarness(TORCHVISION_MANIFEST))
