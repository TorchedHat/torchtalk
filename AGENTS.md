# AGENTS.md

Instructions for any coding agent working in this repository. Claude Code,
Codex, Cursor, Gemini CLI and Copilot all read this file (see
`docs/agent-setup.md` for how each tool picks it up and registers the MCP
server).

## What this project is

TorchTalk is an MCP server that answers structural questions about PyTorch
and PyTorch-based codebases with file and line evidence. It indexes a source
checkout once, caches the result, and exposes eight tools (`get_status`,
`trace`, `search`, `graph`, `modules`, `tests`, `affected`, `bridge`) over
MCP.
Framework conventions live in TOML manifests ("harnesses"), not in code.

## Commands

```bash
# Install
pip install -e ".[dev]"

# Test (what CI runs)
python -m pytest tests/ -v --cov=torchtalk --cov-report=term-missing
PYTORCH_SOURCE=/path/to/pytorch pytest tests/test_binding_detector_pytorch.py

# Lint and format (what CI runs)
ruff check src/torchtalk tests
ruff format --check src/torchtalk tests

# Harness smoke test against a real checkout
python scripts/harness_smoke.py --harness pytorch --source /path/to/pytorch

# Build the index, then serve
torchtalk index build --source /path/to/pytorch
torchtalk mcp-serve --source /path/to/pytorch
```

`--harness <name>` selects a manifest anywhere a source is indexed or served.
A repo can ship its own `.torchtalk.toml` to activate a harness automatically.

## Project structure

```
src/torchtalk/
├── server.py                # FastMCP app: the 8 tool definitions and run_server
├── cli.py                   # CLI: init, status, index, mcp-serve, snapshot, cursor-add
├── indexer.py               # Index build, cache load/save, incremental update, _state
├── harness.py               # ConventionManifest: TOML loading, extends, registry
├── manifests/               # Built-in harnesses: pytorch, torch-extension, vllm, torchvision
├── config.py                # Source resolution (flag, env, config file, .torchtalk.toml)
├── snapshots.py             # Named index snapshots: save/load/diff/export/import
├── symbols.py               # Package-qualified symbol ids for cross-repo merging
├── integration_manifest.py  # Strict loader for tests/integration/*.yml anchors
├── formatting.py            # CompactText / Markdown response formatters
├── tools/                   # Mode handlers behind each MCP tool
│   ├── ops.py               # trace, search
│   ├── graph.py             # graph (callers / calls / impact)
│   ├── modules.py           # modules
│   ├── tests.py             # tests
│   ├── affected.py          # affected
│   └── common.py
└── analysis/
    ├── binding_detector.py  # pybind11 / TORCH_LIBRARY detection (tree-sitter)
    ├── cpp_call_graph.py    # C++ call graph extraction (libclang, parallel)
    ├── libclang_env.py      # libclang discovery and version check
    ├── python_analyzer.py   # Python module/class analysis (AST)
    ├── extractors.py        # Manifest-driven registration extractors (Python AST)
    ├── external_refs.py     # ExternalRef edges: imports, op and C++ calls into depends_on harnesses
    ├── affected.py          # Changed C++ funcs -> impacted Python tests
    ├── alias_map.py         # Python call -> C++ symbol aliases (native_functions.yaml)
    ├── backward_bridge.py   # Backward ATen functions -> forward ops
    ├── decomp_aliases.py    # @register_decomposition pairs
    ├── dispatch_stub_map.py # Kernel impl symbols -> ATen op
    ├── patterns.py          # Shared constants and exclusion patterns
    └── helpers.py
tests/                       # pytest suite; tests/integration/*.yml are harness anchors
scripts/harness_smoke.py     # Indexes a checkout and checks expected_minimums
docs/                        # adding-a-framework.md, bridge-design.md, agent-setup.md
```

## Architecture

**Server** (`server.py`): FastMCP. `run_server` loads or builds the index,
then starts a background thread for the C++ call graph so tools answer
immediately. Tool docstrings become the MCP schema descriptions.

**Indexer** (`indexer.py`): owns the module-level `_state`. Builds bindings,
Python modules, tests, YAML-derived maps and ExternalRef edges; caches them
under the platform cache dir (`~/.cache/torchtalk/` on Linux).
`index update --since <rev>` re-parses only changed files.

**Harness** (`harness.py`, `manifests/`): a `ConventionManifest` describes
search dirs, binding macros, registries and `expected_minimums` for one
framework. `extends` inherits (torchvision and vllm build on
`torch-extension`), `depends_on` names the harnesses that receive
`ExternalRef` edges. `indexer.dependency_index(name)` loads a dependency's
cached index read-only so `bridge` and `trace` can resolve those edges.

**Analysis** (`analysis/`): tree-sitter for bindings, libclang for the C++
call graph (needs `compile_commands.json` for full coverage), Python AST for
modules, tests and registrations.

**Data sources**: `native_functions.yaml`, `derivatives.yaml` (pytorch
harness only), `compile_commands.json`, C++/CUDA/Python source.

## MCP tools

Signatures below match `server.py`. When working with these tools from an
agent, call them over MCP; do not import or run `torchtalk.server` directly.
In Claude Code they appear as `mcp__torchtalk__<name>`.

| Tool | Signature | Notes |
|------|-----------|-------|
| `get_status` | `()` | Readiness summary: bindings, call graph, modules, tests |
| `trace` | `(function_name, focus="full")` | focus: `full`, `yaml`, `dispatch`. Python -> YAML -> C++ -> file:line |
| `search` | `(query, mode="bindings", backend="", limit=10)` | mode `bindings`: dispatch registrations; `kernels`: CUDA kernel launches |
| `graph` | `(function_name, mode="callers", depth=2, fuzzy_all_levels=False, walk_python=False, focus="callers")` | mode `callers`, `calls`, `impact`. depth/fuzzy/walk_python/focus apply to `impact` only; depth is clamped to 5 (`TORCHTALK_GRAPH_MAX_DEPTH` raises it to 10) |
| `modules` | `(name, mode="trace", focus="methods")` | mode `trace`: class details (focus `full` adds bases/docstring); `list`: browse a category (`nn`, `optim`, `all`) |
| `tests` | `(query="", mode="find", limit=10, focus="all")` | mode `find` (focus `functions`/`classes`/`files`), `utils`, `file_info` |
| `affected` | `(funcs, depth=3)` | Comma-separated C++ function names -> impacted Python test files |
| `bridge` | `(symbol, mode="uses", limit=20)` | mode `uses`: ops and C++ APIs `symbol` calls in a `depends_on` harness, resolved through that harness's index; `used_by`: symbols here that reference a dependency symbol |

`graph` needs the C++ call graph, which needs `compile_commands.json` from a
build of the indexed source. Everything else works from source alone.

## Working rules

1. **Dead code first.** Before a structural refactor of a file over 300 LOC,
   remove dead imports, unused functions and stale comments in a separate
   commit.
2. **Phased changes.** Multi-file refactors go in phases of at most 5 files.
   Verify each phase before starting the next.
3. **Fix structure, not symptoms.** If state is duplicated or a pattern is
   inconsistent, propose and make the structural fix instead of a band-aid.
4. **Verify before claiming done.** `pytest` and `ruff check` must pass. If
   no test covers the change, say so explicitly.
5. **Re-read before editing.** Do not edit from memory of a file; re-read the
   region first and read it back after the edit.
6. **No secrets.** Never put API keys, tokens, passwords or internal URLs in
   code, comments, commits or output.
7. **Research before design.** Read the existing pattern (manifest, extractor,
   tool module) before adding a new one.

## Code standards

- Python 3.10+: `list[str]` not `typing.List`, `X | None` not `Optional[X]`.
- Ruff for lint and format, line length 88. Rule set is in `pyproject.toml`.
- No dead code. Every function, import and constant must have a caller.
- Comment only non-obvious logic. No decorative docstrings.
- Keep modules under 500 LOC; split by functional cohesion when they grow.
- MCP tool docstrings are single sentences (they become schema descriptions).
- `Protocol` for interfaces when multiple implementations exist, not `ABC`.
- Composition over inheritance.
- Framework conventions go in a manifest under `manifests/`, never in code.
  `tests/test_harness.py` fails if `pytorch.toml` drifts from
  `analysis/patterns.py`.

## What not to do

- Don't add features beyond what was asked.
- Don't create new files unless a module would exceed 500 LOC otherwise.
- Don't add cross-tool referral language to tool responses ("Try using X").
- Don't use substring matching where word-boundary matching is needed.
- Don't introduce abstractions for hypothetical future frameworks; three
  concrete uses or duplicate it.
- Don't hardcode PyTorch-specific paths or macros in `analysis/`; put them in
  the harness manifest.
