#!/usr/bin/bash
# Environment for a GPU model server, run inside the reference
# assets-studio-gpu/assets-studio-cuda Nix shell.
# Invoke with /usr/bin/bash: the reference shell may put host libc on the
# library path, which breaks Nix's bash before it can repair the environment.
set -euo pipefail

cache="${XDG_CACHE_HOME:-$HOME/.cache}/assets-studio"
driver="$cache/nvidia-driver"

# Keep Nix's runtime libraries, but never expose the host's libc directory.
# Isolate only the NVIDIA driver libraries needed by the CUDA wheels.
IFS=: read -ra library_paths <<< "${LD_LIBRARY_PATH:-}"
clean_paths=()
for path in "${library_paths[@]}"; do
    case "$path" in
        ''|/usr/lib64|/usr/lib/x86_64-linux-gnu) ;;
        *) clean_paths+=("$path") ;;
    esac
done
export LD_LIBRARY_PATH="$(IFS=:; echo "${clean_paths[*]}")"
mkdir -p "$driver"
shopt -s nullglob
for path in /usr/lib64/libcuda.so* /usr/lib64/libnvidia*.so* \
    /usr/lib/x86_64-linux-gnu/libcuda.so* /usr/lib/x86_64-linux-gnu/libnvidia*.so*; do
    ln -sfn "$path" "$driver/${path##*/}"
done
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$driver"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$cache/venvs/trellis2_paint}"
export TRELLIS2_SOURCE="${TRELLIS2_SOURCE:-$cache/trellis2-source}"
export TRELLIS2_PAINT_WEIGHTS="${TRELLIS2_PAINT_WEIGHTS:-$cache/trellis2-paint}"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
export ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-2}"
exec "$@"
