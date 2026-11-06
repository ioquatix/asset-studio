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

from .contract import GenerationError, Job, ModelServer, Output, Task, serve

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
    tasks = (TASK,)

    def __init__(self) -> None:
        self.pipeline: Any = None

    def load(self) -> None:
        import torch
        from diffusers import ZImagePipeline

        root = weights_directory()
        if not (root / "model_index.json").is_file():
            raise FileNotFoundError(f"Z-Image-Turbo weights not found in {root}")
        self.pipeline = ZImagePipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16)
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.set_progress_bar_config(disable=True)

    def generate(self, job: Job) -> list[Output]:
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
