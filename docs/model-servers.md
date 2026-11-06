# Model servers and the text-to-3D chain

The Rust studio runs each model as a separate Python process speaking contract
v1 (synchronous `POST /v1/generate`; readiness from `GET /v1/info`). Processes
start **on demand**: when a job for a server is ready (its input jobs have
succeeded), and the GPU slot and enough GPU memory (the catalog's `vram_mib`,
checked with `nvidia-smi`) are free. A server stops after
`ASSET_STUDIO_IDLE_SECONDS` (default 120) without work, or as soon as it is
idle while another GPU server has a job waiting, so a chain of dependent jobs
hands the one GPU from model to model. If another program holds the memory
(e.g. Ollama), the job waits and the log says why.

| Server id | Task | Model (licence) | Code | Port |
|---|---|---|---|---|
| `zimage` | `text-to-image` | Z-Image-Turbo (Apache-2.0) | `model_servers/` (this repo, MIT) | 8127 |
| `birefnet` | `image-to-image` (cut-out) | BiRefNet dynamic (MIT) | reference repo (GPL-3.0) | 8128 |
| `trellis2_3d` | `image-to-3d` | TRELLIS.2-4B (MIT) | `model_servers/` (this repo, MIT) | 8129 |
| `trellis2` | `3d-paint` | TRELLIS.2-4B texturing (MIT) | reference repo (GPL-3.0) | 8123 |

Our own servers implement the contract in `model_servers/src/asset_studio_servers/contract.py`
rather than importing the reference SDK, which is GPL. The reference servers
run as separate processes and are not linked into this repository.

**Licence note:** the TRELLIS.2 servers bake textures with nvdiffrast, whose
licence (NVIDIA Source Code License) does not permit commercial use. Treat
`trellis2_3d` and `trellis2` outputs as non-commercial until that dependency is
replaced (Effigy replaced it with a pure-torch UV rasterizer).

**Memory:** `trellis2_3d` keeps all its models on the GPU by default
(`TRELLIS2_LOW_VRAM=0`, ~15 GiB resident, ~27 s per model at `1024_cascade`).
Its low-VRAM mode (`ASSET_STUDIO_TRELLIS2_3D_LOW_VRAM=1`, for cards under
24 GB) moves each model to the GPU only for its phase but needs 16-31 GiB of
*host* RAM; on a 46 GB machine with other work and a large tmpfs `/tmp`, the
kernel's OOM killer stopped it (exit 137). `1536_cascade` needs even more.

## Setup (RTX 4090, Linux)

Prerequisites: Nix, and the reference repository
([kvladsrc/gym](https://github.com/kvladsrc/gym)) checked out at
`$ASSET_STUDIO_ROOT/..` (default `~/Developer/kvladsrc/gym`). Python 3.11,
PyTorch 2.8 / CUDA 12.9 and `sm_89` are used; the host CUDA toolkit is not.

```sh
export STUDIO=/absolute/path/to/asset-studio
export ASSET_STUDIO_ROOT=~/Developer/kvladsrc/gym/assets_studio
export NIX_BIN="$HOME/.nix-profile/bin/nix"
in_shell() {  # in_shell <gpu|cuda> <project dir> <venv> <command...>
  (cd "$2" && UV_PROJECT_ENVIRONMENT="$HOME/.cache/assets-studio/venvs/$3" \
    "$NIX_BIN" --extra-experimental-features 'nix-command flakes' \
    develop "$ASSET_STUDIO_ROOT/..#assets-studio-$1" --no-write-lock-file \
    --command /usr/bin/bash "$STUDIO/scripts/gpu_env.sh" "${@:4}")
}

# 1. TRELLIS.2 paint, which also builds the native wheels (CuMesh, FlexGEMM,
#    o-voxel, nvdiffrast) into ~/.cache/assets-studio/trellis2-paint-native.
P=$ASSET_STUDIO_ROOT/model_servers/trellis2_paint
in_shell gpu  $P trellis2_paint uv sync --frozen --extra test
in_shell cuda $P trellis2_paint uv run --no-sync python -m trellis2_paint_model_server.setup_native
in_shell gpu  $P trellis2_paint uv run --no-sync python -c \
  'from trellis2_paint_model_server.server import download; download()'

# 2. BiRefNet.
B=$ASSET_STUDIO_ROOT/model_servers/birefnet
in_shell gpu $B birefnet uv sync --frozen --extra test
in_shell gpu $B birefnet uv run --no-sync python -c \
  'from birefnet_model_server.server import download; download()'

# 3. Our servers (reuses the native wheels from step 1), then their weights:
#    Z-Image-Turbo (~33 GB, published in fp32) and the rest of TRELLIS.2-4B
#    plus TRELLIS-image-large's structure decoder (~11 GB).
"$STUDIO/scripts/setup_model_servers.sh"
in_shell gpu "$STUDIO/model_servers" studio_servers uv run --no-sync python -c '
from asset_studio_servers import trellis2, zimage
zimage.download(); trellis2.download()'
```

Everything heavy lives under `${XDG_CACHE_HOME:-$HOME/.cache}/assets-studio/`:
`venvs/`, `trellis2-source`, `trellis2-paint/` (TRELLIS.2 weights, shared by
both TRELLIS.2 servers, and DINOv3), `z-image-turbo/`, and `logs/` (one log
per server, appended by `scripts/model_server.sh`). Native wheels are not
tagged with a CUDA architecture; on another GPU use a separate
`XDG_CACHE_HOME` for setup and runtime.

On some hosts (Gentoo, Arch) the reference shell appends `/usr/lib64` to
`LD_LIBRARY_PATH`, which loads the host libc into Nix programs (`GLIBC_PRIVATE`
symbol errors). `gpu_env.sh` removes it and links only the NVIDIA driver
libraries; the shell hook still prints one harmless `rm: symbol lookup error`.
Always run the wrapper with the host's `/usr/bin/bash`.

## Use

```sh
cargo run -p asset-studio-server -- --data ~/.asset-studio --listen 127.0.0.1:8090

# In another terminal: a sentence to a textured GLB, 0.8 m tall.
cargo run -p asset-studio-client --bin asset-studio -- text-to-3d \
  "a small mossy boulder with a few ferns, stylised game asset" \
  --seed 11 --height 0.8 -o boulder.glb
```

`text-to-3d` calls `POST /pipelines/text-to-3d`, which queues three jobs:
`zimage` (with `object_reference`: one whole object on a plain background),
`birefnet` on its picture, and `trellis2_3d` on the cut-out. Each stage's
picture or model is a library asset like any other. If a stage fails, the
jobs after it fail with "dependency … did not succeed". Optional body fields:
`seed`, `height_m`, `image_params`, `cutout_params`, `model_params`
(`resolution`: `512` | `1024_cascade` | `1536_cascade`, `steps`, `faces`,
`texture_size`).

The paint smoke test (an existing mesh + cut-out → PBR GLB) is
`python3 scripts/trellis2_smoke.py mesh.glb reference.png --output painted.glb`.

## Verified results (RTX 4090)

**Text to 3D** through the studio, all servers cold (models load from the
page cache), seed 11, defaults (`1024_cascade`, 12 steps, 200k faces, 2048
textures): 2 min 50 s end to end. Z-Image 33 s for one 1024² picture, BiRefNet
0.6 s, TRELLIS.2 63 s generation after a 58 s load. 22 MB GLB, scaled to 0.8 m.
Each stage stopped as soon as the next one had work ("GPU handed over").
Standalone, TRELLIS.2's largest PyTorch allocation was 11.9 GiB.

**Painting**: seed 77, 12 steps, 1024 textures on a 2,112-triangle sphere:
surface and UVs preserved (relative surface error 0.0), 56.8 s, 3.07 GiB peak
PyTorch allocation; the download matched the library's SHA-256.
