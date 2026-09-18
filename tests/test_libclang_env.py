"""Tests for libclang discovery and the bindings version check."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from torchtalk.analysis import libclang_env
from torchtalk.analysis.libclang_env import LibclangSetupError, check_libclang


def _stub(monkeypatch, *, lib="22.1.8", bindings="22.1.8", flag_enum=True):
    fake = SimpleNamespace(
        CursorKind=SimpleNamespace(FLAG_ENUM=1) if flag_enum else SimpleNamespace()
    )
    monkeypatch.setattr(libclang_env, "_cindex", lambda: fake)
    monkeypatch.setattr(libclang_env, "configure", lambda _f: None)
    monkeypatch.setattr(libclang_env, "find_library_file", lambda: None)
    monkeypatch.setattr(
        libclang_env, "library_version", lambda: (lib, "/fake/libclang.so")
    )
    major = int(bindings.split(".")[0])
    monkeypatch.setattr(libclang_env, "bindings_version", lambda: (bindings, major))


def _fake_conf(**attrs):
    conf = SimpleNamespace(loaded=False, library_file=None, library_path=None)
    conf.__dict__.update(attrs)
    return SimpleNamespace(conf=conf)


def _no_distribution(monkeypatch):
    def missing(_name):
        raise libclang_env.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(libclang_env.importlib.metadata, "distribution", missing)


def test_passes_on_matching_majors(monkeypatch):
    _stub(monkeypatch)
    env = check_libclang()
    assert (env.library_major, env.bindings_major) == (22, 22)


def test_rejects_old_bindings(monkeypatch):
    _stub(monkeypatch, lib="18.1.8", bindings="18.1.1")
    with pytest.raises(LibclangSetupError, match="too old"):
        check_libclang()


def test_rejects_bindings_without_flag_enum(monkeypatch):
    _stub(monkeypatch, bindings="19.1.0", flag_enum=False)
    with pytest.raises(LibclangSetupError, match="pip install 'clang>=19'"):
        check_libclang()


def test_mismatched_majors_pass_with_warning(monkeypatch, caplog):
    _stub(monkeypatch, lib="22.1.8", bindings="21.1.7")
    with caplog.at_level(logging.WARNING, logger=libclang_env.__name__):
        env = check_libclang()
    assert (env.bindings_major, env.library_major) == (21, 22)
    assert "bindings 21.1.7 and libclang 22.1.8" in caplog.text


def test_bindings_major_from_api_surface(monkeypatch, tmp_path):
    fake = SimpleNamespace(
        __file__=str(tmp_path / "cindex.py"),
        functionList=[("clang_visitCXXMethods",)],
    )
    monkeypatch.setattr(libclang_env, "_cindex", lambda: fake)
    _no_distribution(monkeypatch)
    assert libclang_env.bindings_version() == ("21.x", 21)


def test_bindings_without_markers_are_too_old(monkeypatch, tmp_path):
    fake = SimpleNamespace(__file__=str(tmp_path / "cindex.py"), functionList=[])
    monkeypatch.setattr(libclang_env, "_cindex", lambda: fake)
    _no_distribution(monkeypatch)
    assert libclang_env.bindings_version() == ("<19", 18)


def test_bindings_metadata_ignored_for_another_module(monkeypatch, tmp_path):
    fake = SimpleNamespace(
        __file__=str(tmp_path / "rpm" / "clang" / "cindex.py"),
        FUNCTION_LIST=[("clang_getCursorLanguage",)],
    )
    dist = SimpleNamespace(
        version="21.1.7", locate_file=lambda rel: tmp_path / "pip" / rel
    )
    monkeypatch.setattr(libclang_env, "_cindex", lambda: fake)
    monkeypatch.setattr(
        libclang_env.importlib.metadata, "distribution", lambda _n: dist
    )
    assert libclang_env.bindings_version() == ("22.x", 22)


def test_env_var_names_the_library(monkeypatch):
    monkeypatch.setenv("LIBCLANG_LIBRARY_FILE", "/opt/llvm/lib/libclang.so")
    monkeypatch.setattr(libclang_env, "_cindex", _fake_conf)
    assert libclang_env.find_library_file() == "/opt/llvm/lib/libclang.so"


def test_linker_name_preferred(monkeypatch):
    monkeypatch.delenv("LIBCLANG_LIBRARY_FILE", raising=False)
    monkeypatch.setattr(libclang_env, "_cindex", _fake_conf)
    monkeypatch.setattr(
        libclang_env.ctypes.util, "find_library", lambda _n: "libclang.so.22.1"
    )
    assert libclang_env.find_library_file() == "libclang.so.22.1"


def test_configured_bindings_left_alone(monkeypatch):
    monkeypatch.delenv("LIBCLANG_LIBRARY_FILE", raising=False)
    monkeypatch.setattr(libclang_env, "_cindex", lambda: _fake_conf(loaded=True))
    assert libclang_env.find_library_file() is None


def test_newest_library_on_disk_wins(monkeypatch, tmp_path):
    monkeypatch.delenv("LIBCLANG_LIBRARY_FILE", raising=False)
    monkeypatch.setattr(libclang_env, "_cindex", _fake_conf)
    monkeypatch.setattr(libclang_env.ctypes.util, "find_library", lambda _n: None)
    for name in (
        "libclang.so.19.1",
        "libclang-cpp.so.22.1",
        "llvm-22/lib/libclang.so.22.1",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(
        libclang_env,
        "_SEARCH_PATTERNS",
        (str(tmp_path / "libclang*.so*"), str(tmp_path / "llvm*/lib/libclang*.so*")),
    )
    assert libclang_env.find_library_file() == str(
        tmp_path / "llvm-22" / "lib" / "libclang.so.22.1"
    )


def test_nothing_found_keeps_bindings_default(monkeypatch, tmp_path):
    monkeypatch.delenv("LIBCLANG_LIBRARY_FILE", raising=False)
    monkeypatch.setattr(libclang_env, "_cindex", _fake_conf)
    monkeypatch.setattr(libclang_env.ctypes.util, "find_library", lambda _n: None)
    monkeypatch.setattr(
        libclang_env, "_SEARCH_PATTERNS", (str(tmp_path / "libclang*.so*"),)
    )
    assert libclang_env.find_library_file() is None


def test_main_reports_failure_as_nonzero(monkeypatch, capsys):
    _stub(monkeypatch, bindings="18.1.1")
    assert libclang_env.main() == 1
    assert "FAILED" in capsys.readouterr().err
