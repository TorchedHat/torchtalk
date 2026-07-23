"""Repo harnesses: convention manifests describing how to index a package.

A manifest is data, not code: search directories, exclusion patterns,
data-source paths, and cache-identity strategy for one package. Graphs built
from different manifests merge via package-qualified symbol IDs (symbols.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .analysis.patterns import (
    CPP_BINDING_PATTERNS,
    CPP_SEARCH_DIRS,
    EXCLUDE_PATTERNS,
    PYTHON_SEARCH_DIRS,
    TEST_CONTENT_PATTERNS,
    TEST_SEARCH_DIRS,
    TEST_UTILITY_MODULES,
)


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


@runtime_checkable
class Harness(Protocol):
    """A package integration: its manifest (extractor hooks arrive later)."""

    manifest: ConventionManifest


PYTORCH_MANIFEST = ConventionManifest(
    package="pytorch",
    cpp_search_dirs=tuple(CPP_SEARCH_DIRS),
    python_search_dirs=tuple(PYTHON_SEARCH_DIRS),
    python_package_roots=("torch",),
    test_search_dirs=tuple(TEST_SEARCH_DIRS),
    test_content_patterns=tuple(TEST_CONTENT_PATTERNS),
    test_utility_modules=tuple(TEST_UTILITY_MODULES),
    exclude_patterns=tuple(EXCLUDE_PATTERNS),
    registration_macros=tuple(CPP_BINDING_PATTERNS),
    native_functions_yaml="aten/src/ATen/native/native_functions.yaml",
    derivatives_yaml="tools/autograd/derivatives.yaml",
    decorator_registries={"register_decomposition": "decompositions"},
)


@dataclass(frozen=True)
class ManifestHarness:
    """Harness defined entirely by manifest data."""

    manifest: ConventionManifest


PYTORCH_HARNESS = ManifestHarness(PYTORCH_MANIFEST)


VLLM_MANIFEST = ConventionManifest(
    package="vllm",
    cpp_search_dirs=("csrc",),
    python_search_dirs=("vllm",),
    python_package_roots=("vllm",),
    test_search_dirs=("tests",),
    exclude_patterns=("/tests/", "/benchmarks/", "/examples/", "__pycache__"),
    cpp_macro_aliases={
        "TORCH_LIBRARY_EXPAND": "TORCH_LIBRARY",
        "TORCH_LIBRARY_IMPL_EXPAND": "TORCH_LIBRARY_IMPL",
    },
    cpp_token_map={"TORCH_EXTENSION_NAME": "_C"},
    decorator_registries={"CustomOp.register": "custom_ops"},
    string_registries=(
        "_TEXT_GENERATION_MODELS",
        "_EMBEDDING_MODELS",
        "_LATE_INTERACTION_MODELS",
        "_REWARD_MODELS",
        "_TOKEN_CLASSIFICATION_MODELS",
        "_SEQUENCE_CLASSIFICATION_MODELS",
        "_MULTIMODAL_MODELS",
        "_SPECULATIVE_DECODING_MODELS",
        "_TRANSFORMERS_SUPPORTED_MODELS",
        "_TRANSFORMERS_BACKEND_MODELS",
    ),
    registration_calls=(
        CallRegistration(
            "direct_register_custom_op", "op_name", "op_func", "custom_ops"
        ),
        CallRegistration("direct_register_custom_op", 0, 1, "custom_ops"),
    ),
    string_dispatchers={"collective_rpc": 0},
)

TORCHVISION_MANIFEST = ConventionManifest(
    package="torchvision",
    cpp_search_dirs=("torchvision/csrc",),
    python_search_dirs=("torchvision",),
    python_package_roots=("torchvision",),
    test_search_dirs=("test",),
    exclude_patterns=tuple(EXCLUDE_PATTERNS),
    cpp_macro_aliases={
        "STABLE_TORCH_LIBRARY_FRAGMENT": "TORCH_LIBRARY_FRAGMENT",
        "STABLE_TORCH_LIBRARY_IMPL": "TORCH_LIBRARY_IMPL",
    },
)

_REGISTRY: dict[str, Harness] = {}
_ACTIVE = "pytorch"


def register_harness(name: str, harness: Harness) -> None:
    """Register a harness under a package name."""
    _REGISTRY[name] = harness


def get_harness(name: str | None = None) -> Harness:
    """Return the named harness, or the active one when no name is given."""
    key = name or _ACTIVE
    if key not in _REGISTRY:
        raise KeyError(f"Unknown harness {key!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def set_active_harness(name: str) -> None:
    """Select the harness used when get_harness() is called without a name."""
    global _ACTIVE
    if name not in _REGISTRY:
        raise KeyError(f"Unknown harness {name!r}. Registered: {sorted(_REGISTRY)}")
    _ACTIVE = name


def active_harness_name() -> str:
    """Name of the harness used when get_harness() is called without a name."""
    return _ACTIVE


def active_manifest() -> ConventionManifest:
    return get_harness().manifest


register_harness("pytorch", PYTORCH_HARNESS)
register_harness("vllm", ManifestHarness(VLLM_MANIFEST))
register_harness("torchvision", ManifestHarness(TORCHVISION_MANIFEST))
