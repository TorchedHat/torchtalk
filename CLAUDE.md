@AGENTS.md

## Claude Code

- The MCP tools are exposed as `mcp__torchtalk__get_status`, `mcp__torchtalk__trace`,
  `mcp__torchtalk__search`, `mcp__torchtalk__graph`, `mcp__torchtalk__modules`,
  `mcp__torchtalk__tests`, `mcp__torchtalk__affected` and `mcp__torchtalk__bridge`.
  Call them directly;
  never import or run `torchtalk.server` from Python.
- `.mcp.json` registers the server for this checkout. `.claude-plugin/plugin.json`
  and `hooks/hooks.json` package it as a plugin, and `scripts/plugin-setup.sh`
  installs it on session start.
- `/trace <name>` (in `.claude/commands/`) and the `torchtalk-analyzer` skill
  (in `.claude/skills/`) wrap the common workflows.
