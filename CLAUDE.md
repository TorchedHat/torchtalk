# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Commands

```bash
# Install
pip install -e .
pip install -e ".[dev]"

# Test
pytest
PYTORCH_SOURCE=/path/to/pytorch pytest tests/test_binding_detector_pytorch.py
VLLM_SOURCE=/path/to/vllm pytest tests/test_vllm_adapter.py

# Lint
torchtalk lint
torchtalk lint --fix --format

# Run MCP server
python -m torchtalk mcp-serve --pytorch-source /path/to/pytorch
python -m torchtalk mcp-serve --framework vllm --source /path/to/vllm
```

## Project Structure

```
src/torchtalk/
├── server.py              # MCP server (7 tools: status + 6 query)
├── cli.py                 # CLI (torchtalk mcp-serve)
├── adapters/              # Framework adapters (PyTorch, vLLM)
├── formatting.py          # Response formatting (CompactText/Markdown)
└── analysis/
    ├── binding_detector.py    # pybind11/TORCH_LIBRARY detection (tree-sitter)
    ├── cpp_call_graph.py      # C++ call graph extraction (libclang)
    ├── vllm_index.py          # Static-first vLLM indexing and proof traces
    ├── python_analyzer.py     # Python module/class analysis (AST)
    ├── patterns.py            # Search directories, exclusion patterns
    └── helpers.py             # Utility functions
```

## Architecture

MCP server providing framework-aware structural analysis for PyTorch and vLLM.

**Server** (`server.py`): FastMCP-based, selects the active adapter, then auto-builds and caches the framework index. PyTorch optionally builds a C++ call graph in the background.

**Analysis** (`analysis/`): BindingDetector (tree-sitter), CppCallGraphExtractor (libclang, 60K+ functions), PythonAnalyzer (AST), and vLLM static index extraction.

**Data Sources**:
- PyTorch: `native_functions.yaml`, `derivatives.yaml`, `compile_commands.json`, tree-sitter AST
- vLLM: API entrypoints, engine anchors, registries, IR/custom-op decorators, and `csrc/*torch_bindings.cpp`

**Cache**: `~/.cache/torchtalk/` — bindings (~10MB), call graph (~50MB).

## MCP Tools

**IMPORTANT**: Use `mcp__torchtalk__*` tools directly. Do NOT import or run Python code from `torchtalk.server`.

| Tool | Description |
|------|-------------|
| `get_status()` | TorchTalk readiness summary across framework, capabilities, entities, and optional PyTorch call graph/test layers |
| `trace(func, focus?)` | Trace a PyTorch op or a vLLM API/op flow depending on the active framework |
| `search(query, mode?, backend?)` | PyTorch: bindings/kernels. vLLM: bindings/apis/models/backends/ops |
| `graph(func, mode?, depth?, fuzzy_all_levels?, walk_python?, focus?)` | PyTorch: C++ call graph. vLLM: condition-aware flow graph |
| `modules(name, mode?, focus?)` | mode="trace": class details (focus="full" adds bases/docstring). mode="list": browse by category ("nn", "optim", "all") |
| `tests(query?, mode?, limit?, focus?)` | mode="find": search tests (focus narrows to functions/classes/files). mode="utils": list utilities (query/focus ignored). mode="file_info": test file details |
| `affected(funcs, depth?)` | Map changed C++ functions (comma-separated) to impacted Python test files |

## Mechanical Overrides

### 1. Step 0 Rule
Before any structural refactor on a file over 300 LOC, first remove all dead imports, unused functions, and stale comments. Commit cleanup separately before starting the real work.

### 2. Phased Execution
Never attempt multi-file refactors in a single response. Break work into phases of no more than 5 files. Complete each phase, run verification, and wait for approval before starting the next.

### 3. Senior Dev Override
Do not default to band-aid fixes. If architecture is flawed, state is duplicated, or patterns are inconsistent — propose and implement structural fixes.

### 4. Forced Verification
Never report a task as complete until `pytest` passes. If tests fail, fix them before claiming done. If no tests cover the change, state that explicitly.

### 5. Context Decay Awareness
After 10+ messages in a conversation, re-read any file before editing it. Do not trust memory of file contents.

### 6. Edit Integrity
Before every file edit, re-read the file. After editing, read the changed section to confirm it applied correctly. Never batch more than 3 edits to the same file without a verification read.

### 7. IP Boundary
Never include real API keys, tokens, passwords, or internal URLs in code, comments, commits, or conversation output.

### 8. Research Before Design
Do not jump into implementation without understanding the problem space. Research best practices and existing patterns before writing code.

## Code Standards

- Python 3.10+. Use `list[str]` not `typing.List[str]`, `X | None` not `Optional[X]`.
- Ruff for linting and formatting. Line length 88.
- No dead code. Every function, import, and constant must be used.
- No verbose comments. Only comment non-obvious logic.
- Keep modules under 500 LOC. Split by functional cohesion when they grow.
- Tool docstrings are single sentences (FastMCP uses them as schema descriptions).
- Protocol for interfaces (not ABC) when multiple implementations exist.
- Favor composition over inheritance.

## What Not To Do

- Don't add features beyond what's requested.
- Don't create new files unless necessary.
- Don't add verbose docstrings or decorative comments.
- Don't add cross-tool referral language in tool responses ("Try using X", "Did you mean").
- Don't use substring matching where word-boundary matching is needed.
- Don't over-engineer with patterns unless the problem demands it.
