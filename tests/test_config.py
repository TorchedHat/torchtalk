"""Tests for per-harness source configuration."""

from __future__ import annotations

import pytest

from torchtalk import config
from torchtalk.config import (
    default_harness,
    resolve_source,
    set_source,
    source_env_var,
    validate_source_path,
)
from torchtalk.harness import get_harness


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point config reads/writes at tmp_path and clear source env vars."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.toml")
    for var in ("PYTORCH_SOURCE", "PYTORCH_PATH"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


class TestSourceEnvVar:
    def test_simple_name(self):
        assert source_env_var("vllm") == "TORCHTALK_SOURCE_VLLM"

    def test_non_alphanumeric_normalized(self):
        assert source_env_var("my-fork.v2") == "TORCHTALK_SOURCE_MY_FORK_V2"


class TestResolveSource:
    def test_cli_flag_wins(self, isolated_config):
        assert resolve_source("vllm", cli_flag="/cli/path") == "/cli/path"

    def test_env_var_beats_config(self, isolated_config, tmp_path, monkeypatch):
        env_dir = tmp_path / "env_src"
        env_dir.mkdir()
        set_source("vllm", "/config/path")
        monkeypatch.setenv("TORCHTALK_SOURCE_VLLM", str(env_dir))
        assert resolve_source("vllm") == str(env_dir)

    def test_env_var_with_missing_path_skipped(self, isolated_config, monkeypatch):
        monkeypatch.setenv("TORCHTALK_SOURCE_VLLM", "/does/not/exist")
        assert resolve_source("vllm") is None

    def test_legacy_pytorch_env_honored(self, isolated_config, tmp_path, monkeypatch):
        env_dir = tmp_path / "pt_src"
        env_dir.mkdir()
        monkeypatch.setenv("PYTORCH_SOURCE", str(env_dir))
        assert resolve_source("pytorch") == str(env_dir)
        assert resolve_source("vllm") is None

    def test_sources_table(self, isolated_config, tmp_path):
        src = tmp_path / "vllm_src"
        src.mkdir()
        set_source("vllm", str(src))
        assert resolve_source("vllm") == str(src)
        assert resolve_source("pytorch") is None

    def test_legacy_config_key_pytorch_only(self, isolated_config, tmp_path):
        src = tmp_path / "pt_src"
        src.mkdir()
        (tmp_path / "config.toml").write_text(f'[source]\npytorch_source = "{src}"\n')
        assert resolve_source("pytorch") == str(src)
        assert resolve_source("vllm") is None


class TestSetSource:
    def test_pytorch_mirrors_legacy_key(self, isolated_config):
        set_source("pytorch", "/pt")
        cfg = config.load_config()
        assert cfg["sources"]["pytorch"] == "/pt"
        assert cfg["source"]["pytorch_source"] == "/pt"

    def test_other_harness_no_legacy_key(self, isolated_config):
        set_source("vllm", "/vllm")
        cfg = config.load_config()
        assert cfg["sources"]["vllm"] == "/vllm"
        assert "source" not in cfg

    def test_make_default(self, isolated_config):
        set_source("vllm", "/vllm", make_default=True)
        assert default_harness() == "vllm"

    def test_default_unset(self, isolated_config):
        set_source("vllm", "/vllm")
        assert default_harness() is None

    def test_multiple_harnesses_coexist(self, isolated_config):
        set_source("pytorch", "/pt")
        set_source("vllm", "/vllm")
        cfg = config.load_config()
        assert cfg["sources"] == {"pytorch": "/pt", "vllm": "/vllm"}


class TestValidateSourcePath:
    def test_missing_path(self, tmp_path):
        manifest = get_harness("pytorch").manifest
        valid, msg = validate_source_path(tmp_path / "ghost", manifest)
        assert not valid
        assert "does not exist" in msg

    def test_pytorch_requires_yaml(self, tmp_path):
        (tmp_path / "torch").mkdir()
        manifest = get_harness("pytorch").manifest
        valid, msg = validate_source_path(tmp_path, manifest)
        assert not valid
        assert "native_functions.yaml" in msg

    def test_pytorch_valid_checkout(self, tmp_path):
        (tmp_path / "torch").mkdir()
        nf = tmp_path / "aten/src/ATen/native/native_functions.yaml"
        nf.parent.mkdir(parents=True)
        nf.write_text("")
        manifest = get_harness("pytorch").manifest
        valid, msg = validate_source_path(tmp_path, manifest)
        assert valid
        assert "pytorch" in msg

    def test_vllm_accepts_csrc_layout(self, tmp_path):
        (tmp_path / "csrc").mkdir()
        manifest = get_harness("vllm").manifest
        valid, _ = validate_source_path(tmp_path, manifest)
        assert valid

    def test_vllm_rejects_unrelated_dir(self, tmp_path):
        (tmp_path / "random").mkdir()
        manifest = get_harness("vllm").manifest
        valid, msg = validate_source_path(tmp_path, manifest)
        assert not valid
        assert "vllm" in msg
