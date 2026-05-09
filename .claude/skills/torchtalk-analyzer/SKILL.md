---
name: torchtalk-analyzer
description: Analyze PyTorch internals across Python, C++, and CUDA layers. Use when asked about how PyTorch operators work internally, where functions are implemented (CPU/CUDA backends), what would break if code is modified (impact analysis), how torch.nn modules connect to native code, or finding tests for PyTorch operators. Covers ATen ops, nn.Module classes, dispatch mechanisms, and test infrastructure.
allowed-tools: mcp__torchtalk__get_status, mcp__torchtalk__trace, mcp__torchtalk__search, mcp__torchtalk__graph, mcp__torchtalk__modules, mcp__torchtalk__tests, mcp__torchtalk__affected, Read, Grep, Glob
---

# TorchTalk PyTorch Analyzer

## When to Use

- "How does torch.X work internally?"
- "Where is X implemented?" (Python/C++/CUDA)
- "What would break if I change X?"
- "How does nn.Linear connect to native code?"
- "Find tests for the softmax operator"

## Quick Start

```
get_status()  # Check what's loaded and available
```

## Tools

| Tool | Use For |
|------|---------|
| `trace(name, focus?)` | Trace any PyTorch op: Python → YAML → C++ → file:line |
| `search(query, mode?, backend?)` | mode="bindings": find dispatch registrations. mode="kernels": find CUDA kernel launches |
| `graph(name, mode?, depth?, fuzzy_all_levels?, walk_python?, focus?)` | mode="callers": inbound. mode="calls": outbound. mode="impact": transitive callers (depth/walk_python/focus apply to impact only) |
| `modules(name, mode?, focus?)` | mode="trace": class details (focus="full" adds bases/docstring). mode="list": browse by category ("nn", "optim", "all") |
| `tests(query?, mode?, limit?, focus?)` | mode="find": search tests (focus narrows to functions/classes/files). mode="utils": list utilities (query/focus ignored). mode="file_info": test file details |
| `affected(funcs, depth?)` | Map changed C++ functions (comma-separated) to impacted Python test files |

## Common Workflows

### "How does torch.matmul work?"
```
trace("matmul")                    # Get binding chain
graph("matmul", mode="calls")     # See internal dependencies
```

### "What breaks if I modify GEMM?"
```
graph("gemm", mode="impact", depth=4)
```

### "Where is conv2d for CUDA?"
```
search("conv2d", backend="CUDA")
trace("conv2d", focus="dispatch")
```

### "How does nn.Linear work?"
```
modules("Linear")                  # Python class details
trace("linear")                    # Underlying ATen op
```

### "Find tests for softmax"
```
tests("softmax")
```

## Requirements

- **Always available:** trace, search, modules, tests
- **Requires PyTorch build:** graph (needs `compile_commands.json`)

Run `get_status()` to check availability.
