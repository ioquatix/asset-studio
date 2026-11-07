"""Z-Image-Turbo text-to-image server (contract v1).

Z-Image-Turbo (Tongyi-MAI, Apache-2.0): a 6B single-stream diffusion
transformer distilled to 8 steps, with a Qwen3-4B text encoder. Weights are
published in fp32 (~33 GB on disk); they are loaded in bf16 and the pipeline's
components move to the GPU one at a time (model CPU offload), so peak VRAM is
about the transformer's 12 GB.

``object_reference`` appends a short instruction that makes the picture a
usable reference for image-to-3D: one whole object, centered, on a plain
background, without a cast shadow. Remove the background (BiRefNet) before
passing the picture to TRELLIS.2.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contract import GenerationError, InputSpec, InvalidInput, Job, ModelServer, Output, Task, serve

WEIGHTS_REPO = "Tongyi-MAI/Z-Image-Turbo"
WEIGHTS_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
OBJECT_REFERENCE = (
    "A single complete object, centered and fully in frame, three-quarter view, "
    "isolated on a plain light grey background, soft even studio lighting, no cast shadow, "
    "no text, no other objects."
)


def weights_directory() -> Path:
    default = Path.home() / ".cache/assets-studio/z-image-turbo"
    return Path(os.environ.get("ZIMAGE_WEIGHTS", default)).expanduser()


def download() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(WEIGHTS_REPO, revision=WEIGHTS_REVISION, local_dir=weights_directory())


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=1024, ge=512, le=2048, description="Width in pixels, a multiple of 16")
    height: int = Field(default=1024, ge=512, le=2048, description="Height in pixels, a multiple of 16")
    steps: int = Field(default=9, ge=1, le=30, description="Inference steps (Turbo is distilled for 8-9)")
    object_reference: bool = Field(
        default=False,
        description="Frame one whole object on a plain background, as a reference for image-to-3D",
    )

    @field_validator("width", "height")
    @classmethod
    def multiple_of_16(cls, value: int) -> int:
        if value % 16:
            raise ValueError("must be a multiple of 16")
        return value


TASK = Task(
    task="text-to-image",
    params=Params,
    output_mime=["image/png"],
    prompt="required",
    max_count=4,
)


class RedrawParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strength: float = Field(default=0.45, ge=0.05, le=1.0,
                            description="How far to move from the input: low keeps its layout, 1 ignores it")
    steps: int = Field(default=9, ge=1, le=30, description="Inference steps before strength is applied")
    size: int = Field(default=1024, ge=512, le=2048,
                      description="Longer side of the output in pixels (a multiple of 16); the input is resized")

    @field_validator("size")
    @classmethod
    def multiple_of_16(cls, value: int) -> int:
        if value % 16:
            raise ValueError("must be a multiple of 16")
        return value


REDRAW = Task(
    task="image-to-image",
    params=RedrawParams,
    output_mime=["image/png"],
    inputs=[InputSpec("image", ["image/png", "image/jpeg", "image/webp"], "The picture to redraw")],
    prompt="required",
    max_count=4,
)


def redraw_size(width: int, height: int, longer: int) -> tuple[int, int]:
    """The output size: the input's aspect ratio, longer side `longer`, both multiples of 16."""
    scale = longer / max(width, height)
    return max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16)


def full_prompt(prompt: str, params: Params) -> str:
    prompt = prompt.strip()
    return f"{prompt}. {OBJECT_REFERENCE}" if params.object_reference else prompt


class ZImageServer(ModelServer):
    model = {
        "id": "z-image-turbo",
        "name": "Z-Image-Turbo",
        "revision": WEIGHTS_REVISION[:12],
        "license": "Apache-2.0",
        "source": "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
        "details": {"precision": "bf16", "offloading": "model CPU offload"},
    }
    tasks = (TASK, REDRAW)

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.redraw: Any = None

    def load(self) -> None:
        import torch
        from diffusers import ZImagePipeline

        root = weights_directory()
        if not (root / "model_index.json").is_file():
            raise FileNotFoundError(f"Z-Image-Turbo weights not found in {root}")
        self.pipeline = ZImagePipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16)
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.set_progress_bar_config(disable=True)
        from diffusers import ZImageImg2ImgPipeline

        # the same components, so image-to-image costs no extra memory
        self.redraw = ZImageImg2ImgPipeline(**self.pipeline.components)
        self.redraw.set_progress_bar_config(disable=True)

    def generate(self, job: Job) -> list[Output]:
        if job.task.task == "image-to-image":
            return self._redraw(job)
        return self._generate(job)

    def _redraw(self, job: Job) -> list[Output]:
        import torch
        from PIL import Image, UnidentifiedImageError

        assert isinstance(job.params, RedrawParams)
        try:
            source = Image.open(io.BytesIO(job.inputs["image"][1]))
            source.load()
        except (UnidentifiedImageError, OSError) as error:
            raise InvalidInput(f"cannot read image: {error}") from error
        alpha = source.getchannel("A") if source.mode in ("RGBA", "LA") else None
        width, height = redraw_size(*source.size, job.params.size)
        source = source.convert("RGB").resize((width, height), Image.LANCZOS)
        outputs = []
        for seed in job.seeds:
            try:
                image = self.redraw(
                    prompt=job.prompt or "",
                    image=source,
                    strength=job.params.strength,
                    width=width,
                    height=height,
                    num_inference_steps=job.params.steps,
                    guidance_scale=0.0,
                    generator=torch.Generator("cpu").manual_seed(seed),
                ).images[0]
            except torch.cuda.OutOfMemoryError as error:
                raise GenerationError("Z-Image ran out of GPU memory; try a smaller size") from error
            if alpha is not None:  # keep the input's cut-out
                image = image.convert("RGBA")
                image.putalpha(alpha.resize(image.size, Image.LANCZOS))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            outputs.append(Output("image/png", buffer.getvalue(),
                                  {"seed": seed, "width": width, "height": height,
                                   "strength": job.params.strength, "steps": job.params.steps}))
        return outputs

    def _generate(self, job: Job) -> list[Output]:
        import torch

        assert isinstance(job.params, Params)
        prompt = full_prompt(job.prompt or "", job.params)
        outputs = []
        for seed in job.seeds:
            try:
                image = self.pipeline(
                    prompt=prompt,
                    width=job.params.width,
                    height=job.params.height,
                    num_inference_steps=job.params.steps,
                    guidance_scale=0.0,
                    generator=torch.Generator("cpu").manual_seed(seed),
                ).images[0]
            except torch.cuda.OutOfMemoryError as error:
                raise GenerationError("Z-Image ran out of GPU memory; try a smaller size") from error
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            outputs.append(
                Output(
                    "image/png",
                    buffer.getvalue(),
                    {"seed": seed, "width": image.width, "height": image.height,
                     "steps": job.params.steps, "prompt": prompt},
                )
            )
        return outputs


def main() -> None:
    serve(ZImageServer(), default_port=9141, description="Z-Image-Turbo text-to-image")


if __name__ == "__main__":
    main()
