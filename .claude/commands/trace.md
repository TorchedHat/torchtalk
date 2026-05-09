---
argument-hint: [function-name]
description: Trace a PyTorch function's cross-language binding chain
allowed-tools: mcp__torchtalk__trace, mcp__torchtalk__graph, Read
---

Trace the PyTorch function `$ARGUMENTS`:

1. Use the `mcp__torchtalk__trace` tool with function_name="$ARGUMENTS" to get the binding chain
2. Use the `mcp__torchtalk__graph` tool with function_name="$ARGUMENTS" and mode="calls" to show outbound dependencies
3. Summarize the dispatch path and implementation locations

IMPORTANT: Use the MCP tools directly. Do NOT try to import/run Python code from torchtalk.server.

Show file:line references for each layer.
