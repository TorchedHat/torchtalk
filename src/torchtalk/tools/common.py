"""Helpers shared across tool modules."""

from __future__ import annotations

from ..analysis.python_analyzer import PyClass, PyFunction
from ..formatting import coverage_note, relative_path
from ..indexer import _state


def _rel_path(path: str) -> str:
    return relative_path(path, _state.source)


def _with_note(text: str) -> str:
    note = coverage_note(_state.cpp_extractor)
    return f"{text}\n\n{note}" if note else text


def _python_symbol(target: str, rel_file: str) -> PyFunction | PyClass | None:
    """The function or class a registration target names.

    A bare name is accepted from the record's own file or, when unambiguous,
    anywhere; a dotted target must match its qualified name. Anything else
    is left unresolved rather than guessed.
    """
    bare = target.rsplit(".", 1)[-1]
    found = _state.py_functions.get(bare, []) + _state.py_classes.get(bare, [])
    pick = [s for s in found if _rel_path(s.file_path) == rel_file]
    pick = pick or [s for s in found if s.qualified_name == target]
    if not pick and len(found) == 1 and "." not in target:
        pick = found
    return pick[0] if pick else None
