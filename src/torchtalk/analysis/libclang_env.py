"""Check that the ``clang`` python bindings and ``libclang.so`` share an LLVM major.

A mismatched pair imports fine but fails partway through walking a translation
unit, so the call graph silently comes out mostly empty. Both must be >= 19.
"""

from __future__ import annotations

import importlib.metadata
import logging
import re
import sys
from dataclasses import dataclass

log = logging.getLogger(__name__)

MIN_MAJOR = 19

# Read by the bindings themselves; also handed to pool workers.
ENV_LIBRARY_FILE = "LIBCLANG_LIBRARY_FILE"

# First libclang entry point each bindings major added, newest first. Used to
# recover the major when the bindings ship without package metadata.
_BINDINGS_MARKERS = (
    (22, "clang_getCursorLanguage"),
    (21, "clang_visitCXXMethods"),
    (20, "clang_visitCXXBaseClasses"),
    (19, "clang_Cursor_getBinaryOpcode"),
)

_VERSION_RE = re.compile(r"version (\d+)\.(\d+)\.(\d+)")

_HINT = (
    "Install the python bindings and libclang from the same LLVM major (>= 19): "
    "e.g. `pip install 'clang==N.*'` with a libclang.so of major N on the "
    "library path or named by LIBCLANG_LIBRARY_FILE, or on Fedora "
    "`dnf install python3-clang clang-libs`."
)


class LibclangSetupError(RuntimeError):
    """The bindings/library pair is missing, mismatched, or too old."""


@dataclass(frozen=True)
class LibclangEnv:
    library_file: str
    library_version: str
    library_major: int
    bindings_version: str
    bindings_major: int

    def describe(self) -> str:
        return (
            f"libclang {self.library_version} ({self.library_file}), "
            f"python bindings {self.bindings_version}"
        )


def _cindex():
    import clang.cindex

    return clang.cindex


def configure(library_file: str | None) -> None:
    """Point the bindings at ``library_file`` before the library is loaded."""
    if not library_file:
        return
    cindex = _cindex()
    cfg = cindex.conf
    if cfg.loaded:
        if cfg.library_file == library_file:
            return
        raise LibclangSetupError(
            f"libclang already loaded from {cfg.library_file or cfg.get_filename()}; "
            f"cannot switch to {library_file}"
        )
    cfg.set_library_file(library_file)


def library_version() -> tuple[str, str]:
    """Load libclang and return ``(version, path)``."""
    cindex = _cindex()
    cfg = cindex.conf
    try:
        lib = cfg.lib
    except cindex.LibclangError as e:
        raise LibclangSetupError(
            f"could not load libclang ({cfg.get_filename()}): {e}. If the message "
            "names a missing symbol, the python bindings are newer than the "
            f"library. {_HINT}"
        ) from e
    except OSError as e:
        raise LibclangSetupError(
            f"could not load libclang ({cfg.get_filename()}): {e}"
        ) from e
    fn = lib.clang_getClangVersion
    fn.restype = cindex._CXString
    text = cindex._CXString.from_result(fn())
    m = _VERSION_RE.search(text)
    if not m:
        raise LibclangSetupError(f"unparseable libclang version string: {text!r}")
    return ".".join(m.groups()), cfg.library_file or cfg.get_filename()


def bindings_version() -> tuple[str, int, bool]:
    """Return ``(version, major, exact)`` for the imported bindings.

    ``exact`` is False when the major came from the API surface and the
    bindings may be newer than the newest marker.
    """
    for dist in ("clang", "libclang"):
        try:
            version = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
        return version, int(version.split(".", 1)[0]), True
    cindex = _cindex()
    table = getattr(cindex, "FUNCTION_LIST", None) or getattr(
        cindex, "functionList", ()
    )
    names = {entry[0] for entry in table}
    for major, marker in _BINDINGS_MARKERS:
        if marker in names:
            return f"{major}.x", major, major != _BINDINGS_MARKERS[0][0]
    return "<19", 18, False


def check_libclang() -> LibclangEnv:
    """Raise :class:`LibclangSetupError` unless the pair is usable."""
    try:
        cindex = _cindex()
    except ImportError as e:
        raise LibclangSetupError(
            f"the `clang` python bindings are not installed. {_HINT}"
        ) from e

    lib_version, lib_file = library_version()
    lib_major = int(lib_version.split(".", 1)[0])
    b_version, b_major, exact = bindings_version()

    if b_major < MIN_MAJOR:
        raise LibclangSetupError(
            f"python clang bindings {b_version} are too old (need >= {MIN_MAJOR}). "
            f"{_HINT}"
        )
    if b_major != lib_major and (exact or lib_major < b_major):
        raise LibclangSetupError(
            f"python clang bindings {b_version} do not match libclang "
            f"{lib_version} at {lib_file}. {_HINT}"
        )
    if not hasattr(cindex.CursorKind, "FLAG_ENUM"):
        raise LibclangSetupError(
            f"python clang bindings {b_version} lack CursorKind.FLAG_ENUM. {_HINT}"
        )

    env = LibclangEnv(lib_file, lib_version, lib_major, b_version, b_major)
    log.info(f"libclang check passed: {env.describe()}")
    return env


def main(argv: list[str] | None = None) -> int:
    try:
        env = check_libclang()
    except LibclangSetupError as e:
        print(f"libclang check FAILED: {e}", file=sys.stderr)
        return 1
    print(env.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
