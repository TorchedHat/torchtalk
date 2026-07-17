"""Config-driven registration extractors (Python AST) and qualname resolver.

All extraction is driven by ConventionManifest fields — no repo-specific
logic lives here. Honesty rule: records from string registries, string
dispatchers, and the qualname-literal resolver carry kind="candidate" with
the string evidence; only decorator and registration-call records (direct
AST evidence) are kind="resolved". Candidates are never static call edges.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ..harness import ConventionManifest

# Dotted identifier with at least one dot — the only literals the resolver
# considers, so plain words never become candidate edges.
_QUALNAME_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")


def _dotted(node: ast.AST) -> str | None:
    """Dotted source text of a Name/Attribute chain, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _matches(dotted: str | None, configured: str) -> bool:
    """True when `dotted` is `configured` or ends with it at a dot boundary."""
    if not dotted:
        return False
    return dotted == configured or dotted.endswith("." + configured)


def _module_name(rel_path: str) -> str:
    parts = Path(rel_path).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class _FileVisitor(ast.NodeVisitor):
    """Single-pass collector for primitives 2-5 plus resolver raw material."""

    def __init__(self, manifest: ConventionManifest, module: str, rel_path: str):
        self.manifest = manifest
        self.module = module
        self.file = rel_path
        self.scope: list[str] = []
        self.records: list[dict] = []
        self.candidate_edges: list[dict] = []
        self.literals: list[dict] = []
        self.qualnames: set[str] = {module}

    def _here(self) -> str:
        return ".".join([self.module, *self.scope]) if self.scope else self.module

    def _enter(self, node) -> None:
        qual = f"{self._here()}.{node.name}"
        self.qualnames.add(qual)
        self._check_decorators(node, qual)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node) -> None:
        self._enter(node)

    def visit_FunctionDef(self, node) -> None:
        self._enter(node)

    def visit_AsyncFunctionDef(self, node) -> None:
        self._enter(node)

    def _check_decorators(self, node, qual: str) -> None:
        for dec in node.decorator_list:
            call = dec if isinstance(dec, ast.Call) else None
            name = _dotted(call.func) if call else _dotted(dec)
            for configured, registry in self.manifest.decorator_registries.items():
                if not _matches(name, configured):
                    continue
                for key in self._decorator_keys(call) or [node.name]:
                    self.records.append(
                        {
                            "registry": registry,
                            "key": key,
                            "target": qual,
                            "kind": "resolved",
                            "via": "decorator",
                            "file": self.file,
                            "line": node.lineno,
                        }
                    )

    def _decorator_keys(self, call: ast.Call | None) -> list[str]:
        if call is None or not call.args:
            return []
        arg = call.args[0]
        elts = arg.elts if isinstance(arg, (ast.List, ast.Tuple)) else [arg]
        keys: list[str] = []
        for e in elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                keys.append(e.value)
            elif d := _dotted(e):
                keys.append(d)
        return keys

    def visit_Assign(self, node) -> None:
        if not self.scope and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in self.manifest.string_registries
                ):
                    self._harvest_registry(target.id, node.value)
        self.generic_visit(node)

    def _harvest_registry(self, name: str, dict_node: ast.Dict) -> None:
        for k, v in zip(dict_node.keys, dict_node.values, strict=True):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                continue
            targets: list[str] = []
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                targets = [v.value]
            elif isinstance(v, ast.Tuple) and len(v.elts) == 2:
                mod, cls = v.elts
                if all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str)
                    for e in (mod, cls)
                ):
                    targets = [f"{mod.value}.{cls.value}"]
            elif isinstance(v, ast.List):
                targets = [
                    f"{k.value}.{e.value}"
                    for e in v.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
            for t in targets:
                self.records.append(
                    {
                        "registry": name,
                        "key": k.value,
                        "target": t,
                        "kind": "candidate",
                        "via": "string_registry",
                        "evidence": t,
                        "file": self.file,
                        "line": k.lineno,
                    }
                )

    def visit_Call(self, node) -> None:
        name = _dotted(node.func)
        for cr in self.manifest.registration_calls:
            if not _matches(name, cr.call):
                continue
            key = self._arg_value(node, cr.key_arg)
            target = self._arg_value(node, cr.target_arg)
            if key and target:
                self.records.append(
                    {
                        "registry": cr.registry or cr.call,
                        "key": key,
                        "target": target,
                        "kind": "resolved",
                        "via": "call",
                        "file": self.file,
                        "line": node.lineno,
                    }
                )
        for dispatcher, arg_spec in self.manifest.string_dispatchers.items():
            if not _matches(name, dispatcher):
                continue
            method = self._arg_value(node, arg_spec, string_only=True)
            if method:
                self.candidate_edges.append(
                    {
                        "kind": "candidate",
                        "via": "string_dispatch",
                        "source": self._here(),
                        "target": method,
                        "evidence": method,
                        "file": self.file,
                        "line": node.lineno,
                    }
                )
        self.generic_visit(node)

    def _arg_value(
        self, call: ast.Call, spec: int | str, string_only: bool = False
    ) -> str | None:
        node: ast.AST | None = None
        if isinstance(spec, int):
            if spec < len(call.args):
                node = call.args[spec]
        else:
            node = next((kw.value for kw in call.keywords if kw.arg == spec), None)
        if node is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if not string_only:
            return _dotted(node)
        return None

    def visit_Constant(self, node) -> None:
        if isinstance(node.value, str) and _QUALNAME_RE.match(node.value):
            self.literals.append(
                {
                    "value": node.value,
                    "scope": self._here(),
                    "file": self.file,
                    "line": node.lineno,
                }
            )


def resolve_qualname_literals(literals: list[dict], qualnames: set[str]) -> list[dict]:
    """Candidate edges for string literals that exactly match indexed qualnames."""
    out: list[dict] = []
    for lit in literals:
        if lit["value"] in qualnames and lit["value"] != lit["scope"]:
            out.append(
                {
                    "kind": "candidate",
                    "via": "qualname_literal",
                    "source": lit["scope"],
                    "target": lit["value"],
                    "evidence": lit["value"],
                    "file": lit["file"],
                    "line": lit["line"],
                }
            )
    return out


def _configured(manifest: ConventionManifest) -> bool:
    return bool(
        manifest.decorator_registries
        or manifest.string_registries
        or manifest.registration_calls
        or manifest.string_dispatchers
    )


def extract_registrations(
    source: str | Path, manifest: ConventionManifest
) -> dict[str, list[dict]]:
    """Walk manifest Python dirs; return {"records", "candidate_edges"}."""
    if not _configured(manifest):
        return {"records": [], "candidate_edges": []}

    src = Path(source)
    records: list[dict] = []
    edges: list[dict] = []
    literals: list[dict] = []
    qualnames: set[str] = set()

    for d in manifest.python_search_dirs or ("",):
        root = src / d if d else src
        if not root.exists():
            continue
        for py in root.rglob("*.py"):
            path_str = str(py)
            if any(p in path_str.lower() for p in manifest.exclude_patterns):
                continue
            try:
                tree = ast.parse(
                    py.read_text(encoding="utf-8", errors="replace"),
                    filename=path_str,
                )
            except SyntaxError:
                continue
            rel = str(py.relative_to(src))
            visitor = _FileVisitor(manifest, _module_name(rel), rel)
            visitor.visit(tree)
            records.extend(visitor.records)
            edges.extend(visitor.candidate_edges)
            literals.extend(visitor.literals)
            qualnames.update(visitor.qualnames)

    edges.extend(resolve_qualname_literals(literals, qualnames))
    return {"records": records, "candidate_edges": edges}
