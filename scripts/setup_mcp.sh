#!/usr/bin/bash
# Create the asset-studio MCP server's environment and run its tests. Pure
# Python (no GPU), so no Nix shell: uv provides Python 3.12. The server is
# then ~/.cache/assets-studio/venvs/mcp/bin/asset-studio-mcp.
set -euo pipefail
scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UV_PROJECT_ENVIRONMENT="${XDG_CACHE_HOME:-$HOME/.cache}/assets-studio/venvs/mcp"
cd -- "$scripts/../mcp_server"
uv sync --extra test --python 3.12
uv run --no-sync pytest -q tests
echo "MCP server: $UV_PROJECT_ENVIRONMENT/bin/asset-studio-mcp"
