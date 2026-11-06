#!/usr/bin/bash
# Create the environment for asset-studio's own model servers
# (model_servers/: Z-Image text-to-image, TRELLIS.2 image-to-3D) and install
# the TRELLIS.2 native wheels built by the reference trellis2_paint setup
# (cached in ~/.cache/assets-studio/trellis2-paint-native/wheels, sm_89).
set -euo pipefail
scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cache="${XDG_CACHE_HOME:-$HOME/.cache}/assets-studio"
root="${ASSET_STUDIO_ROOT:-$HOME/Developer/kvladsrc/gym/assets_studio}"
flake="${ASSET_STUDIO_NIX_FLAKE:-$(cd -- "$root/.." && pwd)}"
nix="${NIX_BIN:-$(command -v nix || echo "$HOME/.nix-profile/bin/nix")}"
wheels="$cache/trellis2-paint-native/wheels"
export UV_PROJECT_ENVIRONMENT="$cache/venvs/studio_servers"
run() {
    "$nix" --extra-experimental-features 'nix-command flakes' \
        develop "$flake#assets-studio-gpu" --no-write-lock-file \
        --command /usr/bin/bash "$scripts/gpu_env.sh" "$@"
}
cd -- "$scripts/../model_servers"
run uv sync --extra test
shopt -s nullglob
native=("$wheels"/{cumesh,flex_gemm,o_voxel,nvdiffrast}-*cp311*linux_x86_64.whl)
if (( ${#native[@]} != 4 )); then
    echo "expected 4 native wheels in $wheels (run the trellis2_paint native setup first)" >&2
    exit 1
fi
run uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps "${native[@]}"
run uv run --no-sync pytest -q tests
