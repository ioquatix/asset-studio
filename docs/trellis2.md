# TRELLIS.2 GPU smoke test

The Rust studio launches a separate Python process and uses synchronous
`POST /v1/generate`; `/v1/info` reports model readiness. A successful Rust
mock test does not verify the Python/CUDA environment.

## Setup (RTX 4090)

Prerequisites: a free GPU, Nix, and a checkout of the reference repository
containing `assets_studio/` and the `assets-studio-cuda` development shell.
The following uses Python 3.11, PyTorch 2.8/CUDA 12.9, and `sm_89`.
The host CUDA toolkit is not used.

```sh
export STUDIO=/absolute/path/to/asset-studio
export ASSET_STUDIO_ROOT=/absolute/path/to/gym/assets_studio
export NIX_BIN="$HOME/.nix-profile/bin/nix" # or the location of your nix binary
export TORCH_CUDA_ARCH_LIST=8.9
cd "$ASSET_STUDIO_ROOT/model_servers/trellis2_paint"

"$NIX_BIN" --extra-experimental-features 'nix-command flakes' \
  develop "$ASSET_STUDIO_ROOT/..#assets-studio-gpu" --no-write-lock-file \
  --command /usr/bin/bash "$STUDIO/scripts/trellis2_env.sh" \
  uv sync --frozen --extra test

"$NIX_BIN" --extra-experimental-features 'nix-command flakes' \
  develop "$ASSET_STUDIO_ROOT/..#assets-studio-cuda" --no-write-lock-file \
  --command /usr/bin/bash "$STUDIO/scripts/trellis2_env.sh" \
  uv run --no-sync python -m trellis2_paint_model_server.setup_native

"$NIX_BIN" --extra-experimental-features 'nix-command flakes' \
  develop "$ASSET_STUDIO_ROOT/..#assets-studio-gpu" --no-write-lock-file \
  --command /usr/bin/bash "$STUDIO/scripts/trellis2_env.sh" \
  uv run --no-sync python -c \
  'from trellis2_paint_model_server.server import download; download()'
```

Source, native wheels, weights, and the virtual environment are kept under
`${XDG_CACHE_HOME:-$HOME/.cache}/assets-studio/`, outside the Rust repository.
The setup downloads pinned texture-only TRELLIS.2 and DINOv3 weights, not
the shape generator. Do not reuse native wheels built for another GPU:
the reference wheel cache does not encode CUDA architecture. Use a separate
`XDG_CACHE_HOME` consistently for setup and runtime in that case.

On Gentoo, the reference shell appends `/usr/lib64` to `LD_LIBRARY_PATH`.
That loads the host libc into Nix programs and produces `GLIBC_PRIVATE`
symbol errors. `trellis2_env.sh` removes the host libc directory and creates
a driver-only symlink directory. The shell hook itself may still emit one
`rm: symbol lookup error` before the wrapper takes control. Invoke the
wrapper with the **host** `/usr/bin/bash`, as shown above.

## Run through the Rust studio

```sh
cd "$STUDIO"
export ASSET_STUDIO_TRELLIS2_CMD=/usr/bin/bash
export ASSET_STUDIO_TRELLIS2_ARGS="$STUDIO/scripts/trellis2_launch.sh"
export ASSET_STUDIO_MODEL_LOG="$HOME/.cache/assets-studio/trellis2-model.log"
cargo run -p asset-studio-server -- --model-server trellis2 \
  --data "$HOME/.asset-studio" --listen 127.0.0.1:8090
```

`--model-server` selects which orchestrators start; the API still advertises
the full catalog. It is repeatable. Without it the current implementation
starts every catalog orchestrator at boot, and GPU servers contend for one
slot; unsuccessful orchestrators do not automatically restart.

In another terminal, supply a valid mesh and a cut-out RGBA reference image:

```sh
python3 scripts/trellis2_smoke.py mesh.glb reference.png \
  --output /path/to/painted.glb
```

The client imports both files, submits one seeded 1024-texture paint job,
waits for completion, downloads the resulting asset, verifies its SHA-256
against the library record, and checks the GLB for materials and images.
It writes the GLB and an adjacent JSON report with the job and asset IDs.
Timeout does not cancel the job. Inspect `ASSET_STUDIO_MODEL_LOG` for Python
loading errors, timings, and per-phase GPU memory use.

## Verified result

A real RTX 4090 run through the Rust launcher, model client, orchestrator,
SQLite library, and download endpoint produced:

- Seed 77, 12 diffusion steps, 1024×1024 textures.
- 683,248-byte GLB with one PBR material and two embedded texture images
  (base color and metallic/roughness).
- 2,112 triangles and 1,058 vertices; the input bounds and UVs were
  preserved, with measured relative surface error `0.0`.
- Model-reported generation time: 56.78 seconds, including first-use work.
- Largest logged per-phase PyTorch allocation: 3.073 GiB (not total VRAM).
- Downloaded SHA-256 matched the content-addressed library record.

The real server exposed a missing JSON `Content-Type` header that the
permissive mock did not catch. The model client now sends it explicitly,
with a regression test. Local GLB parser checks run without GPU packages:

```sh
python3 -m unittest discover -s scripts -p 'test_*.py'
```
