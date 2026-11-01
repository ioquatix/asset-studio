#!/usr/bin/bash
set -euo pipefail
scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="${ASSET_STUDIO_ROOT:?Set ASSET_STUDIO_ROOT to the reference assets_studio directory}"
project="$root/model_servers/trellis2_paint"
flake="${ASSET_STUDIO_NIX_FLAKE:-$(cd -- "$root/.." && pwd)}"
nix="${NIX_BIN:-$(command -v nix || true)}"
if [[ -z "$nix" && -x "$HOME/.nix-profile/bin/nix" ]]; then
    nix="$HOME/.nix-profile/bin/nix"
fi
if [[ -z "$nix" ]]; then
    echo 'Nix not found; set NIX_BIN to its absolute path.' >&2
    exit 1
fi
if [[ -n "${ASSET_STUDIO_MODEL_LOG:-}" ]]; then
    exec >> "$ASSET_STUDIO_MODEL_LOG" 2>&1
fi
cd -- "$project"
exec "$nix" --extra-experimental-features 'nix-command flakes' \
    develop "$flake#assets-studio-cuda" --no-write-lock-file \
    --command /usr/bin/bash "$scripts/trellis2_env.sh" \
    uv run --no-sync trellis2-paint-model-server "$@"
