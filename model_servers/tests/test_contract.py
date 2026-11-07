import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import BaseModel, ConfigDict

from asset_studio_servers import trellis2, zimage
from asset_studio_servers.contract import (
    GenerationError,
    InputSpec,
    InvalidInput,
    ModelServer,
    Output,
    Task,
    create_app,
)


class EchoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: int = 1


class Echo(ModelServer):
    model = {"id": "echo", "name": "Echo"}
    tasks = (
        Task("echo", EchoParams, ["text/plain"], [InputSpec("text", ["text/plain"])], prompt="optional", max_count=3),
    )

    def load(self):
        pass

    def generate(self, job):
        if job.prompt == "fail":
            raise GenerationError("out of memory")
        if job.prompt == "bad":
            raise InvalidInput("unusable")
        data = job.inputs["text"][1]
        return [Output("text/plain", data * job.params.size, {"seed": s}) for s in job.seeds]


def client():
    return TestClient(create_app(Echo(), load=False))


def body(**extra):
    payload = {"task": "echo", "inputs": [{"role": "text", "mime": "text/plain", "data_b64": base64.b64encode(b"hi").decode()}]}
    return payload | extra


def test_info_reports_ready_and_tasks():
    with client() as c:
        info = c.get("/v1/info").json()
    assert info["contract"] == 1 and info["status"] == "ready"
    assert info["tasks"][0]["task"] == "echo"
    assert info["tasks"][0]["params_schema"]["properties"]["size"]["default"] == 1


def test_generate_returns_outputs_with_derived_seeds():
    with client() as c:
        response = c.post("/v1/generate", json=body(count=2, seed=2**32 - 1, params={"size": 2}))
    assert response.status_code == 200
    result = response.json()
    assert [o["meta"]["seed"] for o in result["outputs"]] == [2**32 - 1, 0]
    assert base64.b64decode(result["outputs"][0]["data_b64"]) == b"hihi"
    assert result["params"] == {"size": 2}


@pytest.mark.parametrize(
    "payload, status, code",
    [
        (body(task="nope"), 400, "unsupported_task"),
        (body(params={"size": 1, "extra": 2}), 422, "invalid_request"),
        (body(count=4), 422, "invalid_request"),
        ({"task": "echo"}, 422, "invalid_request"),
        (body(prompt="fail"), 500, "generation_failed"),
        (body(prompt="bad"), 422, "invalid_request"),
        (body(unknown=1), 422, "invalid_request"),
    ],
)
def test_errors_follow_the_contract(payload, status, code):
    with client() as c:
        response = c.post("/v1/generate", json=payload)
    assert response.status_code == status
    error = response.json()["error"]
    assert error["code"] == code and error["retryable"] is False


def test_not_ready_while_loading():
    class Slow(Echo):
        def load(self):
            import time
            time.sleep(5)

    with TestClient(create_app(Slow())) as c:
        assert c.get("/v1/info").json()["status"] == "loading"
        response = c.post("/v1/generate", json=body())
    assert response.status_code == 503 and response.json()["error"]["retryable"] is True


def test_zimage_params_and_prompt():
    with pytest.raises(ValueError):
        zimage.Params(width=1000)
    params = zimage.Params(object_reference=True)
    assert zimage.full_prompt(" a small bush ", params).startswith("a small bush. A single complete object")
    assert zimage.full_prompt("a bush", zimage.Params()) == "a bush"
    assert zimage.TASK.prompt == "required"


def png(mode, alpha=None):
    image = Image.new(mode, (64, 64), (200, 50, 50, 255)[: len(mode)])
    if alpha is not None:
        image.putalpha(Image.fromarray(alpha))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_trellis2_requires_a_cutout():
    with pytest.raises(InvalidInput, match="alpha"):
        trellis2.read_cutout(png("RGB"))
    with pytest.raises(InvalidInput, match="transparent"):
        trellis2.read_cutout(png("RGB", np.full((64, 64), 255, np.uint8)))
    alpha = np.zeros((64, 64), np.uint8)
    alpha[16:48, 16:48] = 255
    assert trellis2.read_cutout(png("RGB", alpha)).mode == "RGBA"
    with pytest.raises(InvalidInput, match="cannot read"):
        trellis2.read_cutout(b"not a png")


def test_trellis2_places_mesh_on_the_ground():
    import trimesh

    mesh = trimesh.creation.box(extents=(0.4, 0.8, 0.2))
    bounds = trellis2.place(mesh, 2.0)
    assert np.allclose(bounds["extents_m"], [1.0, 2.0, 0.5])
    assert np.isclose(bounds["bounds_min"][1], 0.0)
    assert np.allclose([bounds["bounds_min"][0], bounds["bounds_min"][2]], [-0.5, -0.25])


def test_zimage_redraw_keeps_aspect_in_multiples_of_16():
    assert zimage.redraw_size(1000, 500, 1024) == (1024, 512)
    assert zimage.redraw_size(333, 777, 1024) == (432, 1024)
    assert zimage.REDRAW.task == "image-to-image" and zimage.REDRAW.inputs[0].role == "image"
    with pytest.raises(ValueError):
        zimage.RedrawParams(size=1000)
