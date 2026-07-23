#!/usr/bin/env bash
# SessionStart hook: install the torchtalk MCP server if missing, then remind
# the user to point it at a PyTorch source. Non-blocking; never fails the session.

set -euo pipefail

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Install the torchtalk console script if it isn't already on PATH.
if ! command -v torchtalk >/dev/null 2>&1; then
    echo "Installing the torchtalk MCP server from ${PLUGIN_ROOT} ..."
    if pip install -e "${PLUGIN_ROOT}" >/dev/null 2>&1 \
        || pip install "${PLUGIN_ROOT}" >/dev/null 2>&1; then
        echo "torchtalk installed."
    else
        echo "Warning: could not install torchtalk automatically." \
             "Install it manually with: pip install -e ${PLUGIN_ROOT}" >&2
    fi
fi

# Nudge if no source resolves, matching config.resolve_source: PYTORCH_SOURCE,
# PYTORCH_PATH, or any TORCHTALK_SOURCE_* var (each must exist), then the
# config file's [sources] table or legacy pytorch_source key.
config_file="${XDG_CONFIG_HOME:-${HOME}/.config}/torchtalk/config.toml"
source_configured=false

if [[ -n "${PYTORCH_SOURCE:-}" && -d "${PYTORCH_SOURCE}" ]]; then
    source_configured=true
elif [[ -n "${PYTORCH_PATH:-}" && -d "${PYTORCH_PATH}" ]]; then
    source_configured=true
elif compgen -A variable | grep -q "^TORCHTALK_SOURCE_"; then
    source_configured=true
elif [[ -f "${config_file}" ]] \
    && grep -qE "pytorch_source|^\[sources\]" "${config_file}" 2>/dev/null; then
    source_configured=true
fi

if [[ "${source_configured}" == false ]]; then
    echo "TorchTalk: no source configured. Set PYTORCH_SOURCE or run" \
         "'torchtalk init --source <path>' to enable cross-language" \
         "analysis of a PyTorch checkout (recommended)."
fi

exit 0
