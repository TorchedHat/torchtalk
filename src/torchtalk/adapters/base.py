"""Adapter interfaces for framework-specific TorchTalk bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

DEFAULT_FRAMEWORK = "pytorch"
KNOWN_FRAMEWORKS = ("pytorch", "vllm")
FrameworkId = str


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
