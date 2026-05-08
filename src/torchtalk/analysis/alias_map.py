"""Build a Python-call → C++-symbol alias map from native_functions.yaml.

Used by the M1 edge index so a Python source call like `torch.add(x, y)` is
recognised as an edge to `aten::add`. Without this map, only the `torch.ops.*`
and `torch._C.*` call forms get indexed — leaving the dominant `torch.<op>`
form invisible to caller analysis.
"""

from __future__ import annotations


def build_function_alias_map(native_functions: dict[str, dict]) -> dict[str, str]:
    """Map `torch.<op>` → `aten::<op>` for default-namespace function variants.

    Only ops whose `variants:` field includes `function` AND have no
    `python_module:` are emitted. Namespaced ops (`torch.linalg.X`,
    `torch.fft.X`, etc.) require Python-wrapper analysis to resolve their
    exposed call form and are deliberately skipped here.
    """
    aliases: dict[str, str] = {}
    seen: set[str] = set()
    for entry in native_functions.values():
        base = entry.get("base_name")
        if not base or base in seen:
            continue
        seen.add(base)
        if entry.get("python_module"):
            continue
        variants_field = entry.get("variants", "") or ""
        # torchgen default when `variants:` is omitted is `function` — factory
        # ops (`zeros`, `cat`, `stack`) routinely rely on this default.
        if not variants_field.strip():
            aliases[f"torch.{base}"] = f"aten::{base}"
            continue
        variant_set = {v.strip() for v in variants_field.split(",") if v.strip()}
        if "function" in variant_set:
            aliases[f"torch.{base}"] = f"aten::{base}"
    return aliases
