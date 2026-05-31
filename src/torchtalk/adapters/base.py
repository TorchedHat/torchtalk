"""Adapter interfaces for framework-specific TorchTalk bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

DEFAULT_FRAMEWORK = "pytorch"
KNOWN_FRAMEWORKS = ("pytorch", "vllm")
FrameworkId = str
FrameworkCapability = str


class FrameworkAdapter(Protocol):
    """Minimal contract for framework-aware bootstrap."""

    framework_id: FrameworkId
    display_name: str

    def resolve_source(self, cli_flag: str | None = None) -> str | None:
        """Resolve the source root for this framework."""

    def validate_source(self, path: str | Path) -> tuple[bool, str]:
        """Validate that a source path is usable for this framework."""

    def bootstrap(
        self,
        source: str | None = None,
        *,
        index_path: str | None = None,
    ) -> None:
        """Initialize TorchTalk state from a source tree or cached index."""

    def capabilities(
        self,
        state: Any | None = None,
    ) -> frozenset[FrameworkCapability]:
        """Return the currently available capabilities for this framework."""

    def build_index(self, source: str, wait_for_cpp: bool = True) -> dict[str, int]:
        """Build or refresh the framework index and return summary stats."""
