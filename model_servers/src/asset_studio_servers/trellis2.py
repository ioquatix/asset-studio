"""TRELLIS.2 image-to-3D server (contract v1).

Microsoft TRELLIS.2-4B (MIT): a cut-out image becomes a textured, PBR GLB.
The sparse structure, the shape and the texture are three flow models; in
low-VRAM mode each moves to the GPU only for its own phase. The mesh is then
simplified, UV-unwrapped and its base colour / metallic / roughness / alpha
are baked from the generated voxel attributes (``o_voxel.postprocess.to_glb``).

The input must already be a cut-out (RGBA PNG with a transparent background):
run BiRefNet first. Upstream's own background remover (RMBG-2.0) is not used:
its weights are non-commercial. DINOv3 conditioning loads from the local
mirror that the TRELLIS.2 paint server also uses.

The output follows glTF: Y-up, metres, with the object inside a unit cube
centred on the origin (upstream's normalisation). ``height_m`` rescales it so
its height is that many metres and puts it on the ground (min Y = 0).
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from .contract import GenerationError, InputSpec, InvalidInput, Job, ModelServer, Output, Task, serve

SOURCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
WEIGHTS_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
STRUCTURE_DECODER_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"  # TRELLIS-image-large
DINO_REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
MODEL_NAMES = (
    "sparse_structure_flow_model",
    "sparse_structure_decoder",
    "shape_slat_flow_model_512",
    "shape_slat_flow_model_1024",
    "shape_slat_decoder",
    "tex_slat_flow_model_1024",
    "tex_slat_decoder",
)
# Only the "512" resolution uses it: loaded on first use, to keep ~2.6 GB of
# host memory free for the cascades (the process holds every model in RAM).
LAZY_512 = "tex_slat_flow_model_512"


def _path(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable, default)).expanduser()


def source_directory() -> Path:
    return _path("TRELLIS2_SOURCE", "~/.cache/assets-studio/trellis2-source")


def weights_root() -> Path:
    """Holds ``weights/`` (TRELLIS.2-4B, plus TRELLIS-image-large's decoder) and ``dino/``."""
    return _path("TRELLIS2_WEIGHTS", "~/.cache/assets-studio/trellis2-paint")


def download() -> None:
    """The image-to-3D checkpoints of TRELLIS.2-4B, TRELLIS-image-large's structure decoder and DINOv3."""
    from huggingface_hub import snapshot_download

    root = weights_root()
    stems = [
        "ss_flow_img_dit_1_3B_64_bf16",
        "shape_dec_next_dc_f16c32_fp16",
        "slat_flow_img2shape_dit_1_3B_512_bf16",
        "slat_flow_img2shape_dit_1_3B_1024_bf16",
        "tex_dec_next_dc_f16c32_fp16",
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16",
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
    ]
    snapshot_download(
        "microsoft/TRELLIS.2-4B",
        revision=WEIGHTS_REVISION,
        allow_patterns=["pipeline.json"] + [f"ckpts/{s}{x}" for s in stems for x in (".json", ".safetensors")],
        local_dir=root / "weights",
    )
    snapshot_download(
        "microsoft/TRELLIS-image-large",
        revision=STRUCTURE_DECODER_REVISION,
        allow_patterns=["ckpts/ss_dec_conv3d_16l8_fp16.*"],
        local_dir=root / "weights/microsoft/TRELLIS-image-large",
    )
    # The same DINOv3 mirror as the reference TRELLIS.2 paint server.
    snapshot_download(
        "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        revision=DINO_REVISION,
        allow_patterns=["config.json", "model.safetensors"],
        local_dir=root / "dino",
    )


def low_vram() -> bool:
    """``TRELLIS2_LOW_VRAM`` (default on): keep the models in host RAM and move each to the GPU
    only for its phase (~3-12 GiB VRAM, but ~16-31 GiB host RAM at peak). Off: keep them on the
    GPU (~14 GiB of weights plus activations; needs a 24 GB card, little host RAM)."""
    return os.environ.get("TRELLIS2_LOW_VRAM", "1").lower() not in {"0", "false", "no", "off"}


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution: Literal["512", "1024_cascade", "1536_cascade"] = Field(
        default="1024_cascade",
        description="Shape detail: 512 is fastest; the cascades refine a 512 shape to 1024 or 1536",
    )
    steps: int = Field(default=12, ge=1, le=50, description="Sampler steps for structure, shape and texture")
    faces: int = Field(
        default=200_000, ge=5_000, le=2_000_000, description="Target triangle count after simplification"
    )
    texture_size: Literal[1024, 2048, 4096] = Field(default=2048, description="Baked texture size")
    height_m: float | None = Field(
        default=None, gt=0, le=100, description="Scale to this height in metres and stand it on the ground"
    )


TASK = Task(
    task="image-to-3d",
    params=Params,
    output_mime=["model/gltf-binary"],
    inputs=[InputSpec("image", ["image/png"], "Cut-out reference image (RGBA PNG, transparent background)")],
    prompt="none",
    max_count=4,
)


def read_cutout(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise InvalidInput(f"cannot read image: {error}") from error
    if image.mode != "RGBA":
        raise InvalidInput("the image has no alpha channel: remove its background with BiRefNet first")
    alpha = np.asarray(image.getchannel("A"))
    if alpha.min() >= 128:
        raise InvalidInput("the image has no transparent background: remove it with BiRefNet first")
    if (alpha > 204).sum() < 64:
        raise InvalidInput("the image has no opaque foreground")
    return image


def place(mesh: Any, height_m: float | None) -> dict[str, Any]:
    """Scale a Y-up trimesh to ``height_m`` and stand it on the ground; returns its bounds."""
    if height_m is not None:
        lo, hi = mesh.bounds
        height = float(hi[1] - lo[1])
        if height <= 0:
            raise GenerationError("generated mesh has no height")
        mesh.apply_scale(height_m / height)
        lo, hi = mesh.bounds
        mesh.apply_translation([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    lo, hi = mesh.bounds
    return {"bounds_min": lo.tolist(), "bounds_max": hi.tolist(), "extents_m": (hi - lo).tolist()}


class Trellis2Server(ModelServer):
    model = {
        "id": "trellis2-image-to-3d",
        "name": "TRELLIS.2 Image to 3D",
        "revision": WEIGHTS_REVISION[:12],
        "license": "MIT",
        "source": "https://github.com/microsoft/TRELLIS.2",
        "details": {"offloading": "TRELLIS2_LOW_VRAM: models in host RAM, each on the GPU for its phase (default), or all on the GPU"},
    }
    tasks = (TASK,)

    def __init__(self) -> None:
        self.pipeline: Any = None

    def load(self) -> None:
        import torch

        sys.path.insert(0, str(source_directory()))
        from trellis2 import models
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
        from trellis2.pipelines import samplers
        from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

        torch.set_grad_enabled(False)
        root = weights_root()
        args = json.loads((root / "weights/pipeline.json").read_text())["args"]
        # Every path is local: TRELLIS-image-large's decoder is under weights/microsoft/...
        parts = {name: models.from_pretrained(str(root / "weights" / args["models"][name])) for name in MODEL_NAMES}
        pipeline = Trellis2ImageTo3DPipeline(
            models=parts,
            shape_slat_normalization=args["shape_slat_normalization"],
            tex_slat_normalization=args["tex_slat_normalization"],
            image_cond_model=DinoV3FeatureExtractor(str(root / "dino")),
            rembg_model=None,
            low_vram=low_vram(),
            default_pipeline_type=args.get("default_pipeline_type", "1024_cascade"),
        )
        for key in ("sparse_structure", "shape_slat", "tex_slat"):
            spec = args[f"{key}_sampler"]
            setattr(pipeline, f"{key}_sampler", getattr(samplers, spec["name"])(**spec["args"]))
            setattr(pipeline, f"{key}_sampler_params", spec["params"])
        pipeline.cuda()
        self.pipeline = pipeline
        self.root, self.args, self.models = root, args, models

    def generate(self, job: Job) -> list[Output]:
        assert isinstance(job.params, Params)
        image = read_cutout(job.inputs["image"][1])
        return [self._one(image, seed, job.params) for seed in job.seeds]

    def _one(self, image: Image.Image, seed: int, params: Params) -> Output:
        import o_voxel
        import torch

        steps = {"steps": params.steps}
        if params.resolution == "512" and LAZY_512 not in self.pipeline.models:
            model = self.models.from_pretrained(str(self.root / "weights" / self.args["models"][LAZY_512]))
            model.eval()
            self.pipeline.models[LAZY_512] = model
        torch.cuda.reset_peak_memory_stats()
        try:
            mesh = self.pipeline.run(
                image,
                seed=seed,
                pipeline_type=params.resolution,
                sparse_structure_sampler_params=steps,
                shape_slat_sampler_params=steps,
                tex_slat_sampler_params=steps,
            )[0]
            mesh.simplify(16_777_216)  # nvdiffrast's limit
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=params.faces,
                texture_size=params.texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
            )
        except torch.cuda.OutOfMemoryError as error:
            raise GenerationError("TRELLIS.2 ran out of GPU memory; try resolution 512") from error
        finally:
            torch.cuda.empty_cache()
        bounds = place(glb, params.height_m)
        data = glb.export(file_type="glb")
        return Output(
            "model/gltf-binary",
            data,
            {
                "seed": seed,
                "triangles": len(glb.faces),
                "vertices": len(glb.vertices),
                "resolution": params.resolution,
                "texture_size": params.texture_size,
                "steps": params.steps,
                "peak_torch_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "up_axis": "Y",
                "materials": "PBR",
                **bounds,
                "source_revision": SOURCE_REVISION,
                "weights_revision": WEIGHTS_REVISION,
            },
        )


def main() -> None:
    serve(Trellis2Server(), default_port=9142, description="TRELLIS.2 image-to-3D")


if __name__ == "__main__":
    main()
