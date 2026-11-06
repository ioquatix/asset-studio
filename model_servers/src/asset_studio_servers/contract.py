"""A minimal implementation of the asset-studio model-server contract, version 1.

Wire format (the same one the Rust `asset-studio-models` client speaks):

* ``GET /v1/info`` -> ``{contract, status, status_message, model, tasks}``;
  ``status`` is ``loading | ready | busy | error``.
* ``POST /v1/generate`` -> blocks until the generation finishes and returns
  ``{model, task, seed, params, outputs: [{mime, data_b64, meta}], timing}``.
* Errors are ``{"error": {code, message, retryable}}`` with the codes and
  HTTP statuses in :data:`ERROR_STATUS`.

The model loads in a background thread when the server starts, so ``/v1/info``
answers at once. One generation runs at a time; a second request while one is
running gets ``busy`` (retryable). Per-output seeds are ``(seed + i) % 2**32``.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

CONTRACT_VERSION = 1
SEED_MODULUS = 2**32

ERROR_STATUS = {
    "busy": 409,
    "not_ready": 503,
    "unsupported_task": 400,
    "invalid_request": 422,
    "generation_failed": 500,
    "internal": 500,
}
RETRYABLE = {"busy", "not_ready"}

logger = logging.getLogger("asset_studio_servers")


class InvalidInput(Exception):
    """The request is well-formed but its content is unusable (422, not retryable)."""


class GenerationError(Exception):
    """The model could not produce a result, e.g. out of GPU memory (500)."""


@dataclass
class InputSpec:
    role: str
    mime: list[str]
    description: str = ""
    required: bool = True


@dataclass
class Task:
    task: str
    params: type[BaseModel]
    output_mime: list[str]
    inputs: list[InputSpec] = field(default_factory=list)
    # "required" | "optional" | "none"
    prompt: str = "none"
    max_count: int = 1

    def info(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "prompt": self.prompt,
            "inputs": [
                {"role": i.role, "mime": i.mime, "required": i.required, "description": i.description}
                for i in self.inputs
            ],
            "params_schema": self.params.model_json_schema(),
            "max_count": self.max_count,
            "output_mime": self.output_mime,
        }


@dataclass
class Output:
    mime: str
    data: bytes
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    task: Task
    prompt: str | None
    inputs: dict[str, tuple[str, bytes]]
    params: BaseModel
    seeds: list[int]


class ModelServer:
    """Subclass and set :attr:`model` and :attr:`tasks`; implement :meth:`load` and :meth:`generate`."""

    model: dict[str, Any] = {}
    tasks: tuple[Task, ...] = ()

    def load(self) -> None:  # pragma: no cover - model specific
        raise NotImplementedError

    def generate(self, job: Job) -> list[Output]:  # pragma: no cover - model specific
        raise NotImplementedError


def _error(code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message, "retryable": code in RETRYABLE}},
        status_code=ERROR_STATUS[code],
    )


def create_app(server: ModelServer, *, load: bool = True) -> FastAPI:
    """The FastAPI app for ``server``. With ``load``, the model loads in a background thread at startup."""
    state = {"status": "loading", "message": None}
    busy = threading.Lock()
    tasks = {t.task: t for t in server.tasks}

    def do_load() -> None:
        started = time.monotonic()
        try:
            server.load()
        except Exception as error:  # reported through /v1/info, never raised
            logger.exception("model failed to load")
            state.update(status="error", message=f"{type(error).__name__}: {error}")
            return
        state.update(status="ready", message=None)
        logger.info("model %s loaded in %.1f s", server.model.get("id"), time.monotonic() - started)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        if load:
            threading.Thread(target=do_load, name="model-load", daemon=True).start()
        else:
            state.update(status="ready")
        yield

    app = FastAPI(title=server.model.get("name", "model server"), lifespan=lifespan)

    @app.get("/v1/info")
    def info() -> dict[str, Any]:
        status = state["status"]
        if status == "ready" and busy.locked():
            status = "busy"
        return {
            "contract": CONTRACT_VERSION,
            "status": status,
            "status_message": state["message"],
            "model": server.model,
            "tasks": [t.info() for t in server.tasks],
        }

    @app.post("/v1/generate")
    async def generate(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError as error:
            return _error("invalid_request", f"body: {error}")
        if not isinstance(body, dict):
            return _error("invalid_request", "body: expected a JSON object")
        if state["status"] != "ready":
            return _error("not_ready", state["message"] or f"model is {state['status']}")
        task = tasks.get(body.get("task"))
        if task is None:
            return _error("unsupported_task", f"unknown task {body.get('task')!r}; see /v1/info")
        try:
            job = _parse(task, body)
        except (InvalidInput, ValidationError) as error:
            return _error("invalid_request", str(error))
        if not busy.acquire(blocking=False):
            return _error("busy", "another generation is running")
        try:
            # Off the event loop, so /v1/info keeps answering during a long generation.
            import anyio.to_thread

            started = time.monotonic()
            outputs = await anyio.to_thread.run_sync(server.generate, job)
            elapsed = time.monotonic() - started
        except InvalidInput as error:
            return _error("invalid_request", str(error))
        except GenerationError as error:
            return _error("generation_failed", str(error))
        except Exception as error:
            logger.exception("generation failed")
            return _error("internal", f"{type(error).__name__}: {error}")
        finally:
            busy.release()
        return JSONResponse(
            {
                "contract": CONTRACT_VERSION,
                "model": server.model,
                "task": task.task,
                "seed": job.seeds[0],
                "params": job.params.model_dump(mode="json"),
                "outputs": [
                    {"mime": o.mime, "data_b64": base64.b64encode(o.data).decode(), "meta": o.meta}
                    for o in outputs
                ],
                "timing": {"generate_s": elapsed},
            }
        )

    return app


def _parse(task: Task, body: dict[str, Any]) -> Job:
    allowed = {"task", "prompt", "inputs", "params", "count", "seed"}
    if extra := set(body) - allowed:
        raise InvalidInput(f"unknown fields: {sorted(extra)}")
    prompt = body.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise InvalidInput("prompt: expected a string")
    if task.prompt == "required" and not (prompt or "").strip():
        raise InvalidInput(f"{task.task} needs a prompt")
    if task.prompt == "none" and prompt:
        raise InvalidInput(f"{task.task} takes no prompt")

    count = body.get("count", 1)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= task.max_count:
        raise InvalidInput(f"count: between 1 and {task.max_count}")
    seed = body.get("seed")
    if seed is None:
        seed = random.randrange(SEED_MODULUS)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < SEED_MODULUS:
        raise InvalidInput("seed: an integer in [0, 2**32)")

    inputs: dict[str, tuple[str, bytes]] = {}
    specs = {s.role: s for s in task.inputs}
    for item in body.get("inputs") or []:
        if not isinstance(item, dict) or set(item) != {"role", "mime", "data_b64"}:
            raise InvalidInput("inputs: each is {role, mime, data_b64}")
        spec = specs.get(item["role"])
        if spec is None:
            raise InvalidInput(f"inputs: {task.task} has no role {item['role']!r}")
        if item["mime"] not in spec.mime:
            raise InvalidInput(f"inputs.{spec.role}: {item['mime']} is not one of {spec.mime}")
        if spec.role in inputs:
            raise InvalidInput(f"inputs.{spec.role}: given twice")
        try:
            data = base64.b64decode(item["data_b64"], validate=True)
        except ValueError as error:
            raise InvalidInput(f"inputs.{spec.role}: bad base64 ({error})") from None
        inputs[spec.role] = (item["mime"], data)
    if missing := [s.role for s in task.inputs if s.required and s.role not in inputs]:
        raise InvalidInput(f"inputs: missing {missing}")

    params = task.params.model_validate(body.get("params") or {})
    seeds = [(seed + i) % SEED_MODULUS for i in range(count)]
    return Job(task=task, prompt=prompt, inputs=inputs, params=params, seeds=seeds)


def serve(server: ModelServer, *, default_port: int, description: str) -> None:
    """Parse ``--host``/``--port`` (``MODEL_SERVER_HOST``/``MODEL_SERVER_PORT`` override the defaults) and serve."""
    import uvicorn

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default=os.environ.get("MODEL_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MODEL_SERVER_PORT", default_port)))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(server), host=args.host, port=args.port, log_level="info")
