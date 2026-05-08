"""Build a Python-call → C++-symbol alias map from native_functions.yaml.

Used by the M1 edge index so a Python source call like `torch.add(x, y)` is
recognised as an edge to `aten::add`. Without this map, only the `torch.ops.*`
and `torch._C.*` call forms get indexed — leaving the dominant `torch.<op>`
form invisible to caller analysis.
"""

from __future__ import annotations

# python_module → (Python display path under `torch.`, prefix to strip from base)
# `linalg_cross` → `torch.linalg.cross`; `fft_fft` → `torch.fft.fft`;
# `_sparse_mm` → `torch.sparse.mm`. `nn` is a special case: ops with
# `python_module: nn` expose under `torch.nn.functional`, not `torch.nn`.
_NAMESPACE_RULES = {
    "linalg": ("linalg", "linalg_"),
    "fft": ("fft", "fft_"),
    "special": ("special", "special_"),
    "nested": ("nested", "nested_"),
    "sparse": ("sparse", "sparse_"),
    "nn": ("nn.functional", ""),
}


def _exposed_name(base: str, strip: str) -> str:
    """Drop a `<ns>_` or `_<ns>_` prefix from `base`. Falls through if absent."""
    if not strip:
        return base
    if base.startswith(strip):
        return base[len(strip) :]
    underscored = f"_{strip}"
    if base.startswith(underscored):
        return base[len(underscored) :]
    return base


def build_function_alias_map(native_functions: dict[str, dict]) -> dict[str, str]:
    """Map Python `torch.<...>` call forms to canonical `aten::<base>` symbols.

    Default-namespace ops (no `python_module:`) emit `torch.<base>`. Ops with
    `python_module:` in `_NAMESPACE_RULES` emit a namespaced form (e.g.
    `torch.linalg.cross`). Method-only ops and ops in unrecognised namespaces
    are skipped.
    """
    aliases: dict[str, str] = {}
    seen: set[str] = set()
    for entry in native_functions.values():
        base = entry.get("base_name")
        if not base or base in seen:
            continue
        seen.add(base)

        # Skip ops without a function variant. torchgen treats an empty
        # `variants:` as default-`function`, so omitted/blank fields pass.
        variants_field = (entry.get("variants", "") or "").strip()
        if variants_field:
            variant_set = {v.strip() for v in variants_field.split(",") if v.strip()}
            if "function" not in variant_set:
                continue

        pm = entry.get("python_module") or ""
        if not pm:
            aliases[f"torch.{base}"] = f"aten::{base}"
            continue

        rule = _NAMESPACE_RULES.get(pm)
        if rule is None:
            continue
        display, strip = rule
        exposed = _exposed_name(base, strip)
        if exposed:
            aliases[f"torch.{display}.{exposed}"] = f"aten::{base}"
    return aliases
