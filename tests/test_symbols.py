"""Tests for package-qualified symbol identity."""

import subprocess

import pytest

from torchtalk.symbols import (
    UNKNOWN_REVISION,
    PackageIdentity,
    content_fingerprint,
    detect_package_identity,
    make_symbol_id,
    parse_symbol_id,
)


class TestSymbolIds:
    def test_make_and_parse_round_trip(self):
        pkg = PackageIdentity("pytorch", "a1b2c3d4e5f6")
        sid = make_symbol_id(pkg, "at::native::add")
        assert sid == "pytorch@a1b2c3d4e5f6/at::native::add"
        assert parse_symbol_id(sid) == (pkg, "at::native::add")

    def test_symbol_with_overload_dot(self):
        pkg = PackageIdentity("pytorch", "2.9.0a0")
        assert parse_symbol_id(make_symbol_id(pkg, "add.Tensor")) == (
            pkg,
            "add.Tensor",
        )

    def test_symbol_containing_slash(self):
        pkg = PackageIdentity("vllm", "abc123")
        sid = make_symbol_id(pkg, "csrc/ops.h::rotary_embedding")
        assert parse_symbol_id(sid) == (pkg, "csrc/ops.h::rotary_embedding")

    @pytest.mark.parametrize(
        "bad",
        ["", "pytorch/add", "pytorch@abc", "@abc/add", "pytorch@/add", "pytorch@abc/"],
    )
    def test_parse_rejects_malformed(self, bad):
        with pytest.raises(ValueError, match="Malformed symbol ID"):
            parse_symbol_id(bad)


class TestDetectPackageIdentity:
    def test_git_repo_head_sha(self, tmp_path):
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        (tmp_path / "f.txt").write_text("x")
        subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
            cwd=tmp_path,
            check=True,
        )
        head = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert detect_package_identity(tmp_path, name="myrepo") == PackageIdentity(
            "myrepo", head
        )

    def test_detached_head_file_fallback(self, tmp_path):
        sha = "1234567890abcdef1234567890abcdef12345678"
        git_dir = tmp_path / ".git"
        (git_dir / "objects").mkdir(parents=True)
        (git_dir / "refs").mkdir()
        (git_dir / "HEAD").write_text(sha + "\n")
        assert detect_package_identity(tmp_path, "pytorch").revision == sha[:12]

    def test_version_txt_fallback(self, tmp_path):
        (tmp_path / "version.txt").write_text("2.9.0a0\n")
        assert detect_package_identity(tmp_path, "pytorch") == PackageIdentity(
            "pytorch", "2.9.0a0"
        )

    def test_version_py_fallback(self, tmp_path):
        pkg_dir = tmp_path / "mylib"
        pkg_dir.mkdir()
        (pkg_dir / "version.py").write_text('__version__ = "1.2.3"\n')
        assert detect_package_identity(tmp_path, name="mylib") == PackageIdentity(
            "mylib", "1.2.3"
        )

    def test_unknown_when_nothing_found(self, tmp_path):
        assert detect_package_identity(tmp_path, "pytorch").revision == (
            UNKNOWN_REVISION
        )


class TestContentFingerprint:
    def _commit(self, cwd, msg):
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", msg],
            cwd=cwd,
            check=True,
        )

    def _repo(self, tmp_path):
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        (tmp_path / "kernel.cu").write_text("v1")
        subprocess.run(["git", "add", "kernel.cu"], cwd=tmp_path, check=True)
        self._commit(tmp_path, "init")
        return tmp_path

    def test_stable_across_noop_calls(self, tmp_path):
        repo = self._repo(tmp_path)
        assert content_fingerprint(repo) == content_fingerprint(repo)

    def test_changes_on_new_commit(self, tmp_path):
        repo = self._repo(tmp_path)
        before = content_fingerprint(repo)
        (repo / "kernel.cu").write_text("v2")
        self._commit(repo, "edit")
        assert content_fingerprint(repo) != before

    def test_changes_on_dirty_edit_and_re_edit(self, tmp_path):
        repo = self._repo(tmp_path)
        clean = content_fingerprint(repo)
        (repo / "kernel.cu").write_text("v2")
        dirty = content_fingerprint(repo)
        assert dirty != clean
        (repo / "kernel.cu").write_text("v3")
        assert content_fingerprint(repo) not in (clean, dirty)

    def test_non_git_returns_none(self, tmp_path):
        assert content_fingerprint(tmp_path) is None
