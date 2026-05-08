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

    def test_linalg_namespace_strips_prefix(self):
        native = {
            "linalg_cross": {
                "base_name": "linalg_cross",
                "variants": "function, method",
                "python_module": "linalg",
            }
        }
        assert build_function_alias_map(native) == {
            "torch.linalg.cross": "aten::linalg_cross",
        }

    def test_fft_namespace(self):
        native = {
            "fft_fft": {
                "base_name": "fft_fft",
                "variants": "function",
                "python_module": "fft",
            }
        }
        assert build_function_alias_map(native) == {
            "torch.fft.fft": "aten::fft_fft",
        }

    def test_special_namespace(self):
        native = {
            "special_entr": {
                "base_name": "special_entr",
                "variants": "function",
                "python_module": "special",
            }
        }
        assert build_function_alias_map(native) == {
            "torch.special.entr": "aten::special_entr",
        }

    def test_nested_namespace(self):
        native = {
            "nested_to_padded_tensor": {
                "base_name": "nested_to_padded_tensor",
                "variants": "function",
                "python_module": "nested",
            }
        }
        assert build_function_alias_map(native) == {
            "torch.nested.to_padded_tensor": "aten::nested_to_padded_tensor",
        }

    def test_sparse_underscore_prefix_stripped(self):
        # `_sparse_mm` exposes as `torch.sparse.mm` — leading underscore is
        # part of the prefix to drop, not the public name.
        native = {
            "_sparse_mm": {
                "base_name": "_sparse_mm",
                "variants": "function",
                "python_module": "sparse",
            }
        }
        assert build_function_alias_map(native) == {
            "torch.sparse.mm": "aten::_sparse_mm",
        }

    def test_nn_namespace_routes_through_functional(self):
        # `python_module: nn` exposes via `torch.nn.functional`, not `torch.nn`.
        native = {
            "binary_cross_entropy": {
                "base_name": "binary_cross_entropy",
                "variants": "function",
                "python_module": "nn",
            }
        }
        assert build_function_alias_map(native) == {
            "torch.nn.functional.binary_cross_entropy": "aten::binary_cross_entropy",
        }

    def test_unknown_python_module_skipped(self):
        # An unrecognised `python_module:` value is skipped conservatively —
        # we don't know the exposed call form.
        native = {
            "weird_op": {
                "base_name": "weird_op",
                "variants": "function",
                "python_module": "_undocumented_namespace",
            }
        }
        assert build_function_alias_map(native) == {}

    def test_namespaced_op_without_prefix_uses_base_directly(self):
        # `vector_norm` lives under `python_module: linalg` but doesn't carry
        # a `linalg_` prefix; exposed as `torch.linalg.vector_norm`.
        native = {
            "vector_norm": {
                "base_name": "vector_norm",
                "variants": "function",
                "python_module": "linalg",
            }
        }
        assert build_function_alias_map(native) == {
            "torch.linalg.vector_norm": "aten::vector_norm",
        }

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
