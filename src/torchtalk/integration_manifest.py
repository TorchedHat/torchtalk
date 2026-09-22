"""Strict loader for manifest-driven integration anchor YAML files.

Integration manifests live under ``tests/integration/<target>.yml``. Their
filename selects the harness; YAML does not carry a second, redundant harness
field. This module is shared by the test runner, CI workflow, and harness smoke
script so all consumers accept the same contract.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .analysis.binding_detector import BindingType
from .harness import get_harness


class IntegrationManifestError(ValueError):
    """An integration-anchor YAML file violates its contract."""


class _DuplicateKeyError(ValueError):
    """Internal YAML-construction error for repeated mapping keys."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping depth."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise _DuplicateKeyError(
                f"mapping key on line {key_node.start_mark.line + 1} must be a string"
            )
        if key in mapping:
            raise _DuplicateKeyError(
                f"duplicate key {key!r} on line {key_node.start_mark.line + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_ENV_VAR_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_TOP_LEVEL_FIELDS = {"repo", "ref", "env_var", "sparse_paths", "anchors"}
_BINDING_TYPES = frozenset(item.value for item in BindingType)
_ABSTRACT_TARGETS = {"torch-extension"}

_CHECK_FIELDS: dict[str, tuple[set[str], set[str]]] = {
    "pybind_name": ({"file", "value"}, set()),
    "torch_library_cpp_name": ({"file", "value"}, set()),
    "has_cuda_kernel": ({"dir"}, {"glob", "content_filter"}),
    "has_at_dispatch": ({"dir"}, {"glob", "content_filter"}),
    "has_binding_types": ({"dir", "value"}, set()),
}


def supported_anchor_checks() -> frozenset[str]:
    """Return the check names accepted by integration manifests."""
    return frozenset(_CHECK_FIELDS)


def _fail(origin: str, location: str, message: str) -> None:
    raise IntegrationManifestError(f"{origin}: {location}: {message}")


def _require_string(value: Any, *, origin: str, location: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(origin, location, f"must be a non-empty string, got {value!r}")
    return value


def _validate_relative_path(value: Any, *, origin: str, location: str) -> str:
    value = _require_string(value, origin=origin, location=location)
    if (
        "\\" in value
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        _fail(
            origin,
            location,
            f"must be a normalized relative POSIX path, got {value!r}",
        )
    if PurePosixPath(value).is_absolute():
        _fail(origin, location, f"must be a relative path, got {value!r}")
    return value


def _validate_repo(value: Any, *, origin: str) -> str:
    value = _require_string(value, origin=origin, location="repo")
    if not _REPO_RE.fullmatch(value):
        _fail(origin, "repo", "must be a GitHub owner/repo string")
    return value


def _validate_ref(value: Any, *, origin: str) -> str:
    value = _require_string(value, origin=origin, location="ref")
    if value != value.strip() or "\n" in value or "\r" in value:
        _fail(origin, "ref", "must be a single-line, trimmed pinned ref")
    if value.upper() == "HEAD":
        _fail(origin, "ref", "must be a pinned tag or commit, not HEAD")
    return value


def _validate_env_var(value: Any, *, origin: str) -> str:
    value = _require_string(value, origin=origin, location="env_var")
    if not _ENV_VAR_RE.fullmatch(value):
        _fail(origin, "env_var", "must match [A-Z_][A-Z0-9_]*")
    return value


def _validate_sparse_paths(value: Any, *, origin: str) -> list[str]:
    if not isinstance(value, list) or not value:
        _fail(origin, "sparse_paths", "must be a non-empty list of relative paths")
    paths = [
        _validate_relative_path(item, origin=origin, location=f"sparse_paths[{i}]")
        for i, item in enumerate(value)
    ]
    if len(paths) != len(set(paths)):
        _fail(origin, "sparse_paths", "must not contain duplicate paths")
    return paths


def _validate_anchor(anchor: Any, *, origin: str, index: int) -> dict[str, Any]:
    location = f"anchors[{index}]"
    if not isinstance(anchor, dict):
        _fail(origin, location, f"must be a mapping, got {type(anchor).__name__}")
    check = _require_string(
        anchor.get("check"), origin=origin, location=f"{location}.check"
    )
    if check not in _CHECK_FIELDS:
        _fail(
            origin,
            f"{location}.check",
            f"unsupported value {check!r}; expected one of {sorted(_CHECK_FIELDS)}",
        )

    required, optional = _CHECK_FIELDS[check]
    allowed = {"check", *required, *optional}
    unknown = sorted(set(anchor) - allowed)
    if unknown:
        _fail(
            origin,
            location,
            f"unknown field(s) {unknown}; allowed fields: {sorted(allowed)}",
        )
    missing = sorted(required - set(anchor))
    if missing:
        _fail(origin, location, f"missing required field(s) {missing}")

    normalized = dict(anchor)
    if "file" in required:
        normalized["file"] = _validate_relative_path(
            anchor["file"], origin=origin, location=f"{location}.file"
        )
        normalized["value"] = _require_string(
            anchor["value"], origin=origin, location=f"{location}.value"
        )
    if "dir" in required:
        normalized["dir"] = _validate_relative_path(
            anchor["dir"], origin=origin, location=f"{location}.dir"
        )
    if "glob" in optional and "glob" in anchor:
        normalized["glob"] = _require_string(
            anchor["glob"], origin=origin, location=f"{location}.glob"
        )
    if "content_filter" in optional and "content_filter" in anchor:
        normalized["content_filter"] = _require_string(
            anchor["content_filter"],
            origin=origin,
            location=f"{location}.content_filter",
        )
    if check == "has_binding_types":
        value = anchor["value"]
        if not isinstance(value, list) or not value:
            _fail(
                origin,
                f"{location}.value",
                "must be a non-empty list of BindingType values",
            )
        if not all(isinstance(item, str) and item for item in value):
            _fail(origin, f"{location}.value", "must contain only non-empty strings")
        if len(value) != len(set(value)):
            _fail(origin, f"{location}.value", "must not contain duplicates")
        invalid = sorted(set(value) - _BINDING_TYPES)
        if invalid:
            _fail(
                origin,
                f"{location}.value",
                "unknown BindingType value(s) "
                f"{invalid}; expected one of {sorted(_BINDING_TYPES)}",
            )
    return normalized


def validate_integration_manifest(
    data: Any, *, target: str, origin: str
) -> dict[str, Any]:
    """Validate parsed integration YAML and return a normalized mapping."""
    if not isinstance(data, dict):
        _fail(origin, "YAML root", f"must be a mapping, got {type(data).__name__}")
    if target.startswith("_"):
        _fail(origin, "filename", f"target {target!r} is reserved for templates")
    unknown = sorted(set(data) - _TOP_LEVEL_FIELDS)
    if unknown:
        _fail(
            origin,
            "top level",
            f"unknown field(s) {unknown}; allowed fields: {sorted(_TOP_LEVEL_FIELDS)}",
        )
    missing = sorted(_TOP_LEVEL_FIELDS - set(data))
    if missing:
        _fail(origin, "top level", f"missing required field(s) {missing}")
    if target in _ABSTRACT_TARGETS:
        _fail(origin, "filename", f"target {target!r} is an abstract harness")
    try:
        harness = get_harness(target)
    except KeyError:
        _fail(
            origin,
            "filename",
            f"filename-derived harness {target!r} is not registered",
        )
    if harness.manifest.package != target:
        _fail(
            origin,
            "filename",
            f"filename-derived harness {target!r} resolves to package "
            f"{harness.manifest.package!r}",
        )

    anchors = data["anchors"]
    if not isinstance(anchors, list) or not anchors:
        _fail(origin, "anchors", "must be a non-empty list of anchor mappings")
    return {
        "repo": _validate_repo(data["repo"], origin=origin),
        "ref": _validate_ref(data["ref"], origin=origin),
        "env_var": _validate_env_var(data["env_var"], origin=origin),
        "sparse_paths": _validate_sparse_paths(data["sparse_paths"], origin=origin),
        "anchors": [
            _validate_anchor(anchor, origin=origin, index=index)
            for index, anchor in enumerate(anchors)
        ],
        "_name": target,
    }


def load_integration_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate ``tests/integration/<target>.yml`` style YAML."""
    path = Path(path)
    origin = str(path)
    if not path.is_file():
        raise IntegrationManifestError(f"{origin}: manifest not found")
    try:
        with open(path) as f:
            data = yaml.load(f, Loader=_UniqueKeyLoader)
    except _DuplicateKeyError as exc:
        raise IntegrationManifestError(f"{origin}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise IntegrationManifestError(f"{origin}: invalid YAML: {exc}") from exc
    return validate_integration_manifest(data, target=path.stem, origin=origin)
