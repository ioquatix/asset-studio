#!/usr/bin/bash
# Launch one model server in the reference Nix CUDA shell.
#
#   model_server.sh <project-dir> <venv-name> <console-script> [args...]
#
# <project-dir> is the server's uv project; its environment is
# ${XDG_CACHE_HOME:-~/.cache}/assets-studio/venvs/<venv-name>. Output is
# appended to $ASSET_STUDIO_LOG_DIR/<console-script>.log (default:
# the cache's logs/ directory). The Nix flake is $ASSET_STUDIO_NIX_FLAKE,
# else the parent of $ASSET_STUDIO_ROOT (the reference repository).
set -euo pipefail
scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="${1:?project directory}" venv="${2:?venv name}" script="${3:?console script}"
shift 3
cache="${XDG_CACHE_HOME:-$HOME/.cache}/assets-studio"
root="${ASSET_STUDIO_ROOT:-$HOME/Developer/kvladsrc/gym/assets_studio}"
flake="${ASSET_STUDIO_NIX_FLAKE:-$(cd -- "$root/.." && pwd)}"
nix="${NIX_BIN:-$(command -v nix || true)}"
if [[ -z "$nix" && -x "$HOME/.nix-profile/bin/nix" ]]; then
    nix="$HOME/.nix-profile/bin/nix"
fi
if [[ -z "$nix" ]]; then
    echo 'Nix not found; set NIX_BIN to its absolute path.' >&2
    exit 1
fi
logs="${ASSET_STUDIO_LOG_DIR:-$cache/logs}"
mkdir -p "$logs"
exec >> "$logs/$script.log" 2>&1
echo "=== $(date -Is) starting $script (port ${MODEL_SERVER_PORT:-default})"
cd -- "$project"
export UV_PROJECT_ENVIRONMENT="$cache/venvs/$venv"
exec "$nix" --extra-experimental-features 'nix-command flakes' \
    develop "$flake#assets-studio-cuda" --no-write-lock-file \
    --command /usr/bin/bash "$scripts/gpu_env.sh" \
    uv run --no-sync "$script" "$@"
