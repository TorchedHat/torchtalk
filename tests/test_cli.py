"""Tests for torchtalk.cli helpers."""

from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

from torchtalk.cli import (
    _format_coverage,
    _framework_from_args,
    _read_cache_stats,
    _read_coverage_from_cache,
    _source_arg_from_args,
    cmd_index_build,
    cmd_init,
    cmd_lint,
)


class TestFormatCoverage:
    def test_orders_known_buckets_stably(self):
        cov = {"filtered": 3, "ok": 5, "parse_failed": 1, "unsupported_language": 2}
        assert (
            _format_coverage(cov)
            == "5 ok / 1 parse_failed / 2 unsupported_language / 3 filtered"
        )

    def test_appends_unknown_buckets_after_known(self):
        cov = {"ok": 1, "mystery": 7}
        assert _format_coverage(cov) == "1 ok / 7 mystery"

    def test_empty_returns_unknown(self):
        assert _format_coverage({}) == "unknown"

    def test_large_numbers_use_thousands_separator(self):
        assert _format_coverage({"ok": 12345}) == "12,345 ok"


class TestReadCoverageFromCache:
    def test_returns_coverage_when_present(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text(json.dumps({"stats": {"coverage": {"ok": 10}}}))
        assert _read_coverage_from_cache(path) == {"ok": 10}

    def test_returns_none_for_missing_stats(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text(json.dumps({"other": "data"}))
        assert _read_coverage_from_cache(path) is None

    def test_returns_none_for_missing_coverage(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text(json.dumps({"stats": {"total_functions": 5}}))
        assert _read_coverage_from_cache(path) is None

    def test_returns_none_for_corrupt_json(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text("{not valid json")
        assert _read_coverage_from_cache(path) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        assert _read_coverage_from_cache(tmp_path / "missing.json") is None


class TestIndexUpdateExitCode:
    """cmd_index_update must translate uncovered_fail → non-zero exit."""

    def _fake_stats(self, uncovered_fail: bool) -> dict:
        cg = {
            "files_updated": 0,
            "header_affected_tus": 0,
            "files_removed": 0,
            "total_functions": 0,
            "uncovered_headers": 33 if uncovered_fail else 0,
            "uncovered_sample": [],
            "on_uncovered": "fail" if uncovered_fail else "warn",
        }
        if uncovered_fail:
            cg["uncovered_fail"] = True
        return {
            "cpp_files_changed": 0,
            "cpp_files_removed": 0,
            "headers_changed": 0,
            "yaml_changed": False,
            "bindings_total": 0,
            "cuda_kernels_total": 0,
            "baseline_snapshot": "foo",
            "baseline_commit": "abc1234",
            "call_graph": cg,
        }

    def test_returns_one_when_uncovered_fail(self, monkeypatch):
        from argparse import Namespace

        import torchtalk.cli as cli_mod
        from torchtalk import indexer

        monkeypatch.setattr(
            "torchtalk.config.resolve_pytorch_source", lambda: "/tmp/fake"
        )
        monkeypatch.setattr(
            indexer, "update_index", lambda *a, **kw: self._fake_stats(True)
        )
        args = Namespace(since="baseline", pytorch_source=None, on_uncovered="fail")
        assert cli_mod.cmd_index_update(args) == 1

    def test_returns_zero_when_no_uncovered_fail(self, monkeypatch):
        from argparse import Namespace

        import torchtalk.cli as cli_mod
        from torchtalk import indexer

        monkeypatch.setattr(
            "torchtalk.config.resolve_pytorch_source", lambda: "/tmp/fake"
        )
        monkeypatch.setattr(
            indexer, "update_index", lambda *a, **kw: self._fake_stats(False)
        )
        args = Namespace(since="baseline", pytorch_source=None, on_uncovered="warn")
        assert cli_mod.cmd_index_update(args) == 0


class TestReadCacheStats:
    def test_returns_full_stats_dict(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text(
            json.dumps(
                {
                    "stats": {
                        "coverage": {"ok": 5},
                        "include_dirs_count": 42,
                        "total_functions": 100,
                    }
                }
            )
        )
        stats = _read_cache_stats(path)
        assert stats == {
            "coverage": {"ok": 5},
            "include_dirs_count": 42,
            "total_functions": 100,
        }

    def test_returns_none_for_missing_stats_key(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text(json.dumps({"callees": {}}))
        assert _read_cache_stats(path) is None

    def test_returns_none_for_corrupt_json(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text("not json")
        assert _read_cache_stats(path) is None


class TestLintCommand:
    def test_returns_one_when_ruff_missing(self, monkeypatch):
        import torchtalk.cli as cli_mod

        monkeypatch.setattr(cli_mod.shutil, "which", lambda _cmd: None)
        args = Namespace(paths=[], fix=False, format=False)
        assert cmd_lint(args) == 1

    def test_uses_default_paths(self, monkeypatch):
        import torchtalk.cli as cli_mod

        seen: list[list[str]] = []
        monkeypatch.setattr(cli_mod.shutil, "which", lambda _cmd: "/usr/bin/ruff")

        def fake_run(command):
            seen.append(command)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
        args = Namespace(paths=[], fix=False, format=False)

        assert cmd_lint(args) == 0
        assert seen == [["/usr/bin/ruff", "check", "src/torchtalk", "tests"]]

    def test_fix_and_format_issue_expected_commands(self, monkeypatch):
        import torchtalk.cli as cli_mod

        seen: list[list[str]] = []
        monkeypatch.setattr(cli_mod.shutil, "which", lambda _cmd: "/usr/bin/ruff")

        def fake_run(command):
            seen.append(command)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
        args = Namespace(paths=["src/torchtalk"], fix=True, format=True)

        assert cmd_lint(args) == 0
        assert seen == [
            ["/usr/bin/ruff", "check", "--fix", "src/torchtalk"],
            ["/usr/bin/ruff", "format", "src/torchtalk"],
        ]


class TestFrameworkCliHelpers:
    def test_framework_from_args_defaults_to_pytorch(self):
        args = Namespace()
        assert _framework_from_args(args) == "pytorch"

    def test_source_arg_prefers_generic_source(self):
        args = Namespace(
            framework="vllm",
            source="/tmp/generic",
            pytorch_source="/tmp/pytorch",
            vllm_source="/tmp/vllm",
        )
        assert _source_arg_from_args(args) == "/tmp/generic"

    def test_source_arg_prefers_vllm_source_for_vllm(self):
        args = Namespace(
            framework="vllm",
            source=None,
            pytorch_source="/tmp/pytorch",
            vllm_source="/tmp/vllm",
        )
        assert _source_arg_from_args(args) == "/tmp/vllm"


class TestVllmCliPaths:
    def test_cmd_init_vllm_writes_vllm_source(self, monkeypatch):

        saved = {}

        monkeypatch.setattr(
            "torchtalk.config.validate_framework_path",
            lambda framework, path: (True, f"Valid {framework}: {path}"),
        )
        monkeypatch.setattr("torchtalk.config.load_config", lambda: {})

        def fake_save_config(config):
            saved.update(config)
            return "/tmp/config.toml"

        monkeypatch.setattr("torchtalk.config.save_config", fake_save_config)

        args = Namespace(
            framework="vllm",
            source="/tmp/vllm-src",
            pytorch_source=None,
            vllm_source=None,
        )

        assert cmd_init(args) == 0
        assert saved["source"]["vllm_source"].endswith("/tmp/vllm-src")

    def test_cmd_index_build_vllm_uses_framework_build(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "torchtalk.config.resolve_framework_source",
            lambda framework: f"/tmp/{framework}",
        )

        def fake_build_index(source, wait_for_cpp=True, framework="pytorch"):
            assert source == "/tmp/vllm"
            assert wait_for_cpp is True
            assert framework == "vllm"
            return {
                "bindings": 11,
                "cuda_kernels": 0,
                "native_functions": 2,
                "call_graph_functions": 0,
                "call_graph_building": False,
                "python_modules": 7,
                "nn_modules": 3,
                "test_files": 0,
                "test_functions": 0,
            }

        monkeypatch.setattr("torchtalk.indexer.build_index", fake_build_index)

        args = Namespace(
            framework="vllm",
            source=None,
            pytorch_source=None,
            vllm_source=None,
            no_wait=False,
        )

        assert cmd_index_build(args) == 0
        output = capsys.readouterr().out
        assert "Index built for /tmp/vllm" in output
