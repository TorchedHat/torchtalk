"""Helpers shared across tool modules."""

from __future__ import annotations

from ..formatting import coverage_note, relative_path
from ..indexer import _state


def _rel_path(path: str) -> str:
    return relative_path(path, _state.pytorch_source)


def _with_note(text: str) -> str:
    note = coverage_note(_state.cpp_extractor)
    return f"{text}\n\n{note}" if note else text
