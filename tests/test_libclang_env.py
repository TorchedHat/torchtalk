"""Tests for the libclang bindings/library version check."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from torchtalk.analysis import libclang_env
from torchtalk.analysis.libclang_env import LibclangSetupError, check_libclang


def _stub(monkeypatch, *, lib="22.1.8", bindings="22.1.8", exact=True, flag_enum=True):
    fake = SimpleNamespace(
        CursorKind=SimpleNamespace(FLAG_ENUM=1) if flag_enum else SimpleNamespace()
    )
    monkeypatch.setattr(libclang_env, "_cindex", lambda: fake)
    monkeypatch.setattr(
        libclang_env, "library_version", lambda: (lib, "/fake/libclang.so")
    )
    major = int(bindings.split(".")[0])
    monkeypatch.setattr(
        libclang_env, "bindings_version", lambda: (bindings, major, exact)
    )


def test_passes_on_matching_majors(monkeypatch):
    _stub(monkeypatch)
    env = check_libclang()
    assert (env.library_major, env.bindings_major) == (22, 22)


def test_rejects_old_bindings(monkeypatch):
    _stub(monkeypatch, lib="18.1.8", bindings="18.1.1")
    with pytest.raises(LibclangSetupError, match="too old"):
        check_libclang()


def test_rejects_mismatched_majors(monkeypatch):
    _stub(monkeypatch, lib="22.1.8", bindings="21.1.7")
    with pytest.raises(LibclangSetupError, match="do not match"):
        check_libclang()


def test_accepts_newer_library_when_bindings_major_is_inexact(monkeypatch):
    _stub(monkeypatch, lib="23.1.0", bindings="22.x", exact=False)
    assert check_libclang().library_major == 23


def test_bindings_major_from_api_surface(monkeypatch):
    fake = SimpleNamespace(functionList=[("clang_visitCXXMethods",)])
    monkeypatch.setattr(libclang_env, "_cindex", lambda: fake)
    monkeypatch.setattr(
        libclang_env.importlib.metadata,
        "version",
        lambda _n: (_ for _ in ()).throw(
            libclang_env.importlib.metadata.PackageNotFoundError
        ),
    )
    assert libclang_env.bindings_version() == ("21.x", 21, True)


def test_main_reports_failure_as_nonzero(monkeypatch, capsys):
    _stub(monkeypatch, lib="22.1.8", bindings="21.1.7")
    assert libclang_env.main() == 1
    assert "FAILED" in capsys.readouterr().err
