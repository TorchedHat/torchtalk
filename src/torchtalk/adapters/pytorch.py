"""PyTorch adapter for legacy TorchTalk bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import resolve_framework_source, validate_framework_path


@dataclass(frozen=True)
class PyTorchAdapter:
    """Thin adapter that delegates to the existing PyTorch bootstrap."""

    framework_id: str = "pytorch"
    display_name: str = "PyTorch"

    def resolve_source(self, cli_flag: str | None = None) -> str | None:
        return resolve_framework_source(self.framework_id, cli_flag)

    def validate_source(self, path: str | Path) -> tuple[bool, str]:
        return validate_framework_path(self.framework_id, path)

    def bootstrap(
        self,
        source: str | None = None,
        *,
        index_path: str | None = None,
    ) -> None:
        from .. import indexer

        if source:
            indexer._init_from_source(source, framework=self.framework_id)
            return
        if index_path:
            indexer._load_from_json(
                index_path,
                framework=self.framework_id,
            )
            return
        raise ValueError("PyTorchAdapter.bootstrap requires source or index_path")
