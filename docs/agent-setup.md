# Using TorchTalk from coding agents

TorchTalk is a plain MCP server, so any MCP client can use it. The repo
instructions for agents live in one file, `AGENTS.md`, which the tools below
read natively.

## Instructions file

| Tool | How it finds the instructions |
|------|-------------------------------|
| Claude Code | `CLAUDE.md` imports `AGENTS.md` with `@AGENTS.md` and adds a short Claude-only section |
| Codex | Reads `AGENTS.md` from the repo root (32 KiB cap per file) |
| Cursor | Reads `AGENTS.md` at the root; skills under `.claude/skills/` are picked up too |
| Gemini CLI | `.gemini/settings.json` sets `context.fileName` to `AGENTS.md` |
| Copilot coding agent, Aider, others | Read `AGENTS.md` by convention |

Keep tool-specific content (tool name prefixes, plugin wiring) out of
`AGENTS.md` and in that tool's own file.

## Register the MCP server

Replace `/path/to/pytorch` with a source checkout. Add `--harness vllm` (or
another manifest name) after `mcp-serve` to index a different framework.

```bash
# Claude Code
claude mcp add torchtalk -s user -- torchtalk mcp-serve --source /path/to/pytorch

# Codex
codex mcp add torchtalk -- torchtalk mcp-serve --source /path/to/pytorch

# Cursor (project-local .cursor/mcp.json plus a copy of the skill and command)
torchtalk cursor-add -C /path/to/your/project -p /path/to/pytorch

# Gemini CLI
gemini mcp add torchtalk torchtalk mcp-serve --source /path/to/pytorch

# Any other client: a stdio server launched with
torchtalk mcp-serve --source /path/to/pytorch
```

For clients configured by JSON (Cursor, Windsurf, VS Code), the server block
in `.mcp.json` at the repo root is the reference shape.

The source can also come from the environment instead of `--source`:
`PYTORCH_SOURCE` for the pytorch harness, or `TORCHTALK_SOURCE_<HARNESS>`
for any harness, or `torchtalk init --source <path>` once to persist it.

## Claude Code plugin

`.claude-plugin/plugin.json` packages the server, the `/trace` command and
the `torchtalk-analyzer` skill as a Claude Code plugin. `hooks/hooks.json`
runs `scripts/plugin-setup.sh` on session start to install the console
script if it is missing.
