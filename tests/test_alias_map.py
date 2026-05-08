"""Tests for the function alias map builder."""

from __future__ import annotations

from torchtalk.analysis.alias_map import build_function_alias_map


class TestBuildFunctionAliasMap:
    def test_default_namespace_function_variant_emitted(self):
        native = {
            "add": {
                "base_name": "add",
                "variants": "function, method",
                "python_module": "",
            }
        }
        assert build_function_alias_map(native) == {"torch.add": "aten::add"}

    def test_method_only_op_skipped(self):
        # `Tensor.foo` only — no `torch.foo` callable exists.
        native = {
            "foo": {
                "base_name": "foo",
                "variants": "method",
                "python_module": "",
            }
        }
        assert build_function_alias_map(native) == {}

    def test_python_module_namespaced_op_skipped(self):
        # `linalg_cross` exposes as `torch.linalg.cross` not `torch.linalg_cross`,
        # so the bare-base alias would be wrong — skip until wrapper analysis.
        native = {
            "linalg_cross": {
                "base_name": "linalg_cross",
                "variants": "function, method",
                "python_module": "linalg",
            }
        }
        assert build_function_alias_map(native) == {}

    def test_empty_variants_defaults_to_function(self):
        # torchgen's default when `variants:` is omitted is `function` — factory
        # ops like `zeros` / `cat` rely on this default.
        native = {
            "zeros": {
                "base_name": "zeros",
                "variants": "",
                "python_module": "",
            }
        }
        assert build_function_alias_map(native) == {"torch.zeros": "aten::zeros"}

    def test_method_only_explicit_skipped(self):
        # When `variants:` is explicitly `method`, the function form is unsafe
        # to alias (callable doesn't exist as `torch.<op>`).
        native = {
            "method_only": {
                "base_name": "method_only",
                "variants": "method",
                "python_module": "",
            }
        }
        assert build_function_alias_map(native) == {}

    def test_overload_entries_dedup_by_base_name(self):
        # Parser stores both `add.Tensor` and `add` keyed entries; the alias
        # builder must emit a single mapping per base.
        shared = {
            "base_name": "add",
            "variants": "function, method",
            "python_module": "",
        }
        native = {"add.Tensor": shared, "add": shared, "add.Scalar": shared}
        assert build_function_alias_map(native) == {"torch.add": "aten::add"}

    def test_underscore_prefixed_op_emitted(self):
        # Internal ops like `_safe_softmax` are still callable as `torch._safe_softmax`
        # when they have a function variant.
        native = {
            "_safe_softmax": {
                "base_name": "_safe_softmax",
                "variants": "function",
                "python_module": "",
            }
        }
        assert build_function_alias_map(native) == {
            "torch._safe_softmax": "aten::_safe_softmax",
        }

    def test_missing_base_name_skipped(self):
        # Defensive: malformed entries without base_name don't crash.
        native = {"corrupt": {"variants": "function"}}
        assert build_function_alias_map(native) == {}

    def test_handles_none_python_module_field(self):
        # YAML may omit `python_module:` entirely; treat absent same as empty.
        native = {
            "relu": {
                "base_name": "relu",
                "variants": "function, method",
                # python_module key absent
            }
        }
        assert build_function_alias_map(native) == {"torch.relu": "aten::relu"}

    def test_whitespace_in_variants_field_handled(self):
        native = {
            "x": {
                "base_name": "x",
                "variants": "  function ,  method  ",
                "python_module": "",
            }
        }
        assert build_function_alias_map(native) == {"torch.x": "aten::x"}
