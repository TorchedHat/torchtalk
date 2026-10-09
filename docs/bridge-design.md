# Bridge design: cross-package references

TorchTalk indexes one package at a time (PyTorch, vLLM, torchvision, ...).
The *bridge* connects two indexed packages so questions like "which vLLM
kernels call this ATen op?" or "what breaks in vLLM if `torch.nn.Module`
changes?" can be answered from static analysis alone.

This document fixes the data model. Import, op and C++ edges are collected
today and resolved against a dependency's own index; the rest is roadmap.

## One primitive: `ExternalRef`

```python
@dataclass(frozen=True)
class ExternalRef:
    from_symbol: str  # qualified symbol in the *referencing* package
    to_name: str  # name as written ("torch.nn.Module", "at::empty")
    kind: str  # import | op | cpp | base_class | provides | version_pin
    evidence: str  # "path:line" — always points at real source
    confidence: float  # 1.0 for syntactic facts, lower for heuristics
    to_package: str  # harness name this ref resolves against ("pytorch")
```

`to_package` is an addition to the five-field sketch from the design notes:
collection already knows which `depends_on` entry a name belongs to, and
recording it saves the resolver a second lookup.

A bridge is then simply: for every `ExternalRef` in package A whose
`to_package == B`, look up `to_name` in B's symbol table. Unresolved refs
are kept (they are the "vLLM uses a symbol PyTorch no longer exports"
signal), not dropped.

## Kinds and their resolvers

| kind          | collected from                                   | resolver list (manifest)      | status |
|---------------|--------------------------------------------------|-------------------------------|--------|
| `import`      | top-level and nested `import` / `from ... import` statements         | `python_package_roots` of dep | done   |
| `op`          | `torch.ops.aten.X`, `torch.X` calls (the Py→C++ edges) | `[python] op_namespaces` of dep | done |
| `cpp`         | `at::X`, `c10::X` callees in the C++ call graph   | `[bridge] cpp_namespaces`     | done   |
| `base_class`  | `class Foo(torch.nn.Module)`                      | `[bridge] base_class_namespaces` | roadmap |
| `provides`    | `TORCH_LIBRARY` / `register_op` registrations     | (direction flipped: this package *defines* `to_name`) | roadmap |
| `version_pin` | `requirements*.txt`, `pyproject.toml`             | none — package-level edge     | roadmap |

Everything is a manifest list, not code: adding a framework means listing
which namespaces belong to its dependency, never adding a new edge class.

### Why imports first

Import edges are pure syntax, already parsed by `PythonAnalyzer`, and exist
in every Python package. They give the bridge a smoke test on day one
(`external_refs > 0` for vLLM) and a coarse dependency map (which vLLM
modules touch `torch.distributed` vs `torch.nn`) before any resolver exists.

### Direction

Refs always point *out* of the package being indexed. A registration
(`kind="provides"`) is the same record with the meaning flipped: vLLM
*provides* `_C::rms_norm` into the `torch.ops` namespace. The Workspace
joins A's `provides` with B's `op` refs to answer "who implements this".

## Manifest fields

```toml
[package]
depends_on = ["pytorch"]          # bridge targets, in resolution order

[python.op_namespaces]
torch = "aten"                    # torch.X  -> aten::X

[bridge]
cpp_namespaces = ["at", "c10", "torch"]
base_class_namespaces = ["torch.nn", "torch.autograd", "torch.optim"]

[bridge.cpp_op_namespaces]
at = "aten"                       # at::X -> aten::X when the call graph has no definition
```

`torch-extension.toml` carries these, so every extension profile inherits
them via `extends`.

## What is deliberately skipped

- **dynamo / `torch.compile` decorators** — runtime behaviour, no static
  target symbol.
- **Attribute chains through aliases** (`F = torch.nn.functional; F.relu`)
  — resolved later by the existing `alias_map`, not by the collector.
- **Third parties not in `depends_on`** (`numpy`, `triton`) — no symbol
  table to resolve against; listing them would only add noise.

## Storage

External refs live in `ServerState.external_refs` as plain dicts and are
recomputed per load: import and op refs from the Python pass, cpp refs once
the C++ call graph is ready. The count is reported as `external_refs` in the
stats returned by `build_index` / `update_index` and in `torchtalk index
build` output. The snapshot schema is untouched.

## Resolution

Refs resolve against the dependency's own cached index, loaded read-only
(`indexer.dependency_index`) from the source `resolve_source(dep)` names.
The dependency is never rebuilt from the extension side; `get_status` says
how to build it when it is missing.

- `op` `aten::X`: the dispatch kernels of the dependency's `native_functions`
  entry (a structured op's kernels sit on its `out=` delegate), then
  implementations named after the op, then registration bindings. Kernels
  with no indexed body are listed at the `native_functions.yaml` that
  declares them, with no line.
- `cpp`: an exact match in the dependency call graph's function locations;
  otherwise `cpp_op_namespaces` maps the namespace onto the op path. When
  neither resolves, the extension's own call graph may still show where the
  symbol is declared in the installed headers.
- `import`: listed, not resolved (dependency Python modules are not cached).

The dependency's `native_functions` also feed the extension's alias map, so
plain `torch.X()` calls become `aten::X` edges even though the extension
has no `native_functions.yaml` of its own.

## Roadmap

- `base_class` and `provides` edges, version pins.
- `affected --across`: changed PyTorch functions to impacted extension tests.
