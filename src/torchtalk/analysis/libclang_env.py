"""Find libclang and check that the ``clang`` python bindings are recent enough.

Bindings older than LLVM 19 do not know cursor kinds that current standard
library headers produce, so every translation unit fails to walk. The library
itself may be any LLVM release; only the entry points TorchTalk uses must exist.
"""

from __future__ import annotations

import ctypes.util
import glob
import importlib.metadata
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

MIN_MAJOR = 19

# Overrides the search below with the path of the libclang to load.
ENV_LIBRARY_FILE = "LIBCLANG_LIBRARY_FILE"

_SEARCH_PATTERNS = (
    "/usr/lib64/libclang*.so*",
    "/usr/lib/libclang*.so*",
    "/usr/lib/*-linux-gnu/libclang*.so*",
    "/usr/lib/llvm*/lib/libclang*.so*",
    "/usr/lib64/llvm*/lib/libclang*.so*",
    "/usr/local/lib/libclang*.so*",
    "/usr/local/opt/llvm/lib/libclang.dylib",
    "/opt/homebrew/opt/llvm/lib/libclang.dylib",
)

# First libclang entry point each bindings major added, newest first. Used to
# recover the major when the bindings ship without package metadata.
_BINDINGS_MARKERS = (
    (22, "clang_getCursorLanguage"),
    (21, "clang_visitCXXMethods"),
    (20, "clang_visitCXXBaseClasses"),
    (19, "clang_Cursor_getBinaryOpcode"),
)

_VERSION_RE = re.compile(r"version (\d+)\.(\d+)\.(\d+)")

_INSTALL_BINDINGS = f"Run `pip install 'clang>={MIN_MAJOR}'`."
_INSTALL_LIBRARY = (
    "Install your distribution's libclang package (clang-libs, libclang1-N, "
    f"llvm) or set {ENV_LIBRARY_FILE} to the path of a libclang shared library."
)


class LibclangSetupError(RuntimeError):
    """The bindings are missing or too old, or no libclang could be loaded."""


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


def _major(version: str) -> int:
    return int(version.split(".", 1)[0])


def configure(library_file: str | None) -> None:
    """Relax the bindings' symbol check and point them at ``library_file``."""
    cindex = _cindex()
    cfg = cindex.conf
    if not cfg.loaded:
        cindex.Config.set_compatibility_check(False)
    if not library_file:
        return
    if cfg.loaded:
        if cfg.library_file == library_file:
            return
        raise LibclangSetupError(
            f"libclang already loaded from {cfg.library_file or cfg.get_filename()}; "
            f"cannot switch to {library_file}"
        )
    cfg.set_library_file(library_file)


def _newest(paths: list[str]) -> str:
    return max(paths, key=lambda p: [int(n) for n in re.findall(r"\d+", p)])


def find_library_file() -> str | None:
    """Return the libclang to load: the env var, the linker's, or the newest on
    disk. None leaves the bindings' own default in place."""
    explicit = os.environ.get(ENV_LIBRARY_FILE)
    cfg = _cindex().conf
    if explicit or cfg.loaded or cfg.library_file or cfg.library_path:
        return explicit
    found = ctypes.util.find_library("clang")
    if found:
        return found
    candidates = [p for pat in _SEARCH_PATTERNS for p in glob.glob(pat)]
    candidates = [p for p in candidates if "libclang-cpp" not in p]
    return _newest(candidates) if candidates else None


def library_version() -> tuple[str, str]:
    """Load libclang and return ``(version, path)``."""
    cindex = _cindex()
    cfg = cindex.conf
    name = cfg.library_file or cfg.get_filename()
    # Load directly first: the bindings replace the loader's error text.
    try:
        ctypes.CDLL(name)
    except OSError as e:
        raise LibclangSetupError(
            f"could not load libclang ({name}): {e}. {_INSTALL_LIBRARY}"
        ) from e
    lib = cfg.lib
    fn = lib.clang_getClangVersion
    fn.restype = cindex._CXString
    text = cindex._CXString.from_result(fn())
    m = _VERSION_RE.search(text)
    if not m:
        raise LibclangSetupError(f"unparseable libclang version string: {text!r}")
    return ".".join(m.groups()), name


def bindings_version() -> tuple[str, int]:
    """Return ``(version, major)`` for the imported bindings."""
    cindex = _cindex()
    module = Path(cindex.__file__).resolve()
    for name in ("clang", "libclang"):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        # Only trust metadata that describes the module actually imported.
        if Path(dist.locate_file("clang/cindex.py")).resolve() == module:
            return dist.version, _major(dist.version)
    table = getattr(cindex, "FUNCTION_LIST", None) or getattr(
        cindex, "functionList", ()
    )
    names = {entry[0] for entry in table}
    for major, marker in _BINDINGS_MARKERS:
        if marker in names:
            return f"{major}.x", major
    return f"<{MIN_MAJOR}", MIN_MAJOR - 1


def check_libclang() -> LibclangEnv:
    """Raise :class:`LibclangSetupError` unless the call graph can be built."""
    try:
        cindex = _cindex()
    except ImportError as e:
        raise LibclangSetupError(
            f"the `clang` python bindings are not installed. {_INSTALL_BINDINGS}"
        ) from e

    b_version, b_major = bindings_version()
    if b_major < MIN_MAJOR or not hasattr(cindex.CursorKind, "FLAG_ENUM"):
        raise LibclangSetupError(
            f"python clang bindings {b_version} are too old (need >= {MIN_MAJOR}). "
            f"{_INSTALL_BINDINGS}"
        )

    configure(find_library_file())
    lib_version, lib_file = library_version()
    lib_major = _major(lib_version)
    env = LibclangEnv(lib_file, lib_version, lib_major, b_version, b_major)
    if lib_major != b_major:
        log.warning(
            f"python clang bindings {b_version} and libclang {lib_version} are from "
            "different LLVM releases; install matching versions if the C++ call "
            "graph build fails"
        )
    log.info(f"libclang check passed: {env.describe()}")
    return env


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(format="%(levelname)s: %(message)s")
    try:
        env = check_libclang()
    except LibclangSetupError as e:
        print(f"libclang check FAILED: {e}", file=sys.stderr)
        return 1
    print(env.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
