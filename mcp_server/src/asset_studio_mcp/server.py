"""The asset-studio MCP server (stdio).

A thin client of the studio's REST API (``ASSET_STUDIO_URL``, default
``http://127.0.0.1:8090``), so the studio must be running. It adds three
things the REST API does not have:

* ``look``: an image an agent can see. Pictures are returned (downscaled);
  models are rendered by a headless Blender as a 2x2 sheet (front, right,
  back, three-quarter), cached by content hash.
* Bounded waits for jobs: a tool that starts a job waits at most 45 s,
  reporting progress, then returns the job id to poll with ``job_status``.
  If the agent cancels a call that started a job, the job is cancelled too.
* A bridge to the live Blender through the Blender MCP add-on's socket
  (``BLENDER_MCP_HOST``/``BLENDER_MCP_PORT``, default ``localhost:9876``):
  ``open_in_blender`` imports an asset into its own collection,
  ``pull_from_blender`` saves the edited collection as a **new** asset
  derived from the original.

Nothing here deletes or overwrites: edits and generations become new assets,
and ``export_asset`` refuses to overwrite a file.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Any

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image
from mcp.types import ToolAnnotations

STUDIO = os.environ.get("ASSET_STUDIO_URL", "http://127.0.0.1:8090").rstrip("/")
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "assets-studio" / "mcp"
MAX_WAIT_S = 45.0
TERMINAL = {"succeeded", "failed", "cancelled"}
MODEL_EXTENSIONS = {".glb": "mesh"}
IMAGE_EXTENSIONS = {".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image"}


def _ops(name: str) -> Path:
    """A script from blender_ops/ (GPL, run by Blender, never imported here)."""
    for root in (Path(__file__).with_name("blender_ops"), Path(__file__).resolve().parents[2] / "blender_ops"):
        if (root / name).is_file():
            return root / name
    raise ToolError(f"internal: blender_ops/{name} is missing from the installation")


def _hints(read_only: bool = False, idempotent: bool = False, destructive: bool = False) -> ToolAnnotations:
    return ToolAnnotations(
        read_only_hint=read_only,
        destructive_hint=destructive,
        idempotent_hint=read_only or idempotent,
        open_world_hint=False,
    )


READ, ADD, SAME = _hints(read_only=True), _hints(), _hints(idempotent=True)
STOP = _hints(idempotent=True, destructive=True)

INSTRUCTIONS = """\
asset-studio: a local library of pictures and 3D models (GLB), with generation jobs run by
model servers on this machine's GPU. Assets are never overwritten: every generation, import
and Blender edit is a new asset.

Typical uses:
- A 3D model from a sentence: generate_3d_from_text (Z-Image picture -> BiRefNet cut-out ->
  TRELLIS.2 textured GLB, about 3 min cold), then look at the result.
- A model from your own picture: import_file it, then run_job birefnet image-to-image and
  trellis2_3d image-to-3d on the cut-out (inputs can reference a job's output).
- Repaint a mesh: run_job trellis2 3d-paint with a mesh and a cut-out image.
- Edit in Blender: open_in_blender, edit with the Blender MCP tools, pull_from_blender.
Jobs wait at most 45 s per call; continue with job_status. Read asset-studio://guide for
task parameters.
"""

GUIDE = """\
# asset-studio guide

## Model servers and tasks

| server | task | inputs | prompt | main params (defaults) |
|---|---|---|---|---|
| zimage | text-to-image | none | required | width/height 1024 (multiple of 16, 512-2048), steps 9, object_reference false |
| birefnet | image-to-image | image (png/jpeg/webp) | none | variant "dynamic", refine_foreground true, largest_only false |
| trellis2_3d | image-to-3d | image (cut-out RGBA PNG) | none | resolution "1024_cascade" ("512" fast, "1536_cascade" detailed), steps 12, faces 200000, texture_size 2048, height_m (scale + stand on ground) |
| trellis2 | 3d-paint | mesh (GLB), image (cut-out RGBA PNG) | none | texture_size 2048 (1024/2048/4096), steps 12 |

Only one GPU model runs at a time; servers start when a job for them is ready and stop when
idle. A cold start loads the model (Z-Image ~40 s, TRELLIS.2 ~60 s). If another program
(e.g. Ollama) holds GPU memory, jobs wait; job_status says so only through staying queued.

## Inputs

run_job inputs map a role to an asset id (a string) or to another job's output:
{"image": {"job": "<job id>", "output": 0}}. A job whose input job fails, fails too.

## Good prompts for 3D

generate_3d_from_text asks Z-Image for one whole object on a plain background. Describe the
object, its material and style ("a wooden barrel with iron bands, stylised game asset");
leave out scenery and lighting. height_m sets the real-world height.

## Blender

open_in_blender imports a model into the live Blender (Blender MCP add-on running) in a
collection named "AS <title> [<id>]". Edit it there, then pull_from_blender saves the
collection as a new asset tagged derived:<original id>. close_in_blender removes it.
"""

server = MCPServer("asset-studio", instructions=INSTRUCTIONS)


# --- the studio API -------------------------------------------------------------------------


async def api(method: str, path: str, *, json_body: Any = None, content: bytes | None = None,
              params: dict[str, str] | None = None, raw: bool = False, timeout: float = 60) -> Any:
    """Call the studio; an API error becomes a ToolError with its message."""
    try:
        async with httpx.AsyncClient(base_url=STUDIO, timeout=timeout) as client:
            response = await client.request(method, path, json=json_body, content=content, params=params)
    except httpx.TransportError as error:
        raise ToolError(f"unreachable: the studio is not answering at {STUDIO} ({error}); "
                        "start asset-studio-server") from None
    if response.is_error:
        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        raise ToolError(f"{response.status_code}: {message}")
    return response.content if raw else response.json()


def _summary(asset: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "kind", "mime", "title", "size_bytes", "producer", "tags", "rating", "created_at")
    return {k: asset.get(k) for k in keys}


def _job_summary(job: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "server", "task", "state", "prompt", "params", "seeds", "outputs", "error")
    summary = {k: job.get(k) for k in keys}
    summary["inputs"] = {
        role: (ref.get("Asset", {}).get("id") if "Asset" in ref else ref.get("JobOutput"))
        for role, ref in (job.get("inputs") or {}).items()
    }
    return summary


async def _job(job_id: str) -> dict[str, Any]:
    return (await api("GET", f"/jobs/{job_id}"))["job"]


async def _wait(job_ids: list[str], wait_s: float, ctx: Context | None, own: bool) -> list[dict[str, Any]]:
    """Poll jobs until the last one is terminal, a stage fails, or wait_s (<= 45) passes.

    If this call is cancelled while waiting and it started the jobs (``own``),
    the unfinished jobs are cancelled too, so nothing outlives the request.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, min(wait_s, MAX_WAIT_S))
    reported, reported_at = None, 0.0
    try:
        while True:
            jobs = [await _job(i) for i in job_ids]
            last = jobs[-1]
            if last["state"] in TERMINAL or any(j["state"] in {"failed", "cancelled"} for j in jobs):
                return jobs
            done = sum(j["state"] == "succeeded" for j in jobs)
            current = next((j for j in jobs if j["state"] != "succeeded"), last)
            message = f"{current['server']} {current['task']}: {current['state']}"
            # On a change, else every 10 s as a keep-alive.
            if ctx is not None and (message != reported or loop.time() - reported_at >= 10):
                reported, reported_at = message, loop.time()
                with contextlib.suppress(Exception):
                    await ctx.report_progress(done, len(jobs), message)
            if loop.time() >= deadline:
                return jobs
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        if own:
            for job_id in job_ids:
                with contextlib.suppress(Exception):
                    await api("POST", f"/jobs/{job_id}/cancel", timeout=5)
        raise


def _result(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    last = jobs[-1]
    failed = next((j for j in jobs if j["state"] in {"failed", "cancelled"}), None)
    if failed is not None:
        status = failed["state"]
    elif last["state"] == "succeeded":
        status = "done"
    else:
        status = "running"
    result: dict[str, Any] = {"status": status, "jobs": [_job_summary(j) for j in jobs]}
    if status == "done":
        result["output_assets"] = last["outputs"]
        result["next"] = "look at the output_assets, or export_asset / open_in_blender"
    elif status == "running":
        result["next"] = f"call job_status with job_id {last['id']} (wait_s up to 45)"
    else:
        result["error"] = failed.get("error")
    return result


# --- reading ----------------------------------------------------------------------------------


@server.tool(annotations=READ)
async def list_servers() -> dict[str, Any]:
    """Use this to see which model servers and tasks exist (ids, tasks, input roles, whether a
    prompt is used). Parameters are in the resource asset-studio://guide."""
    return await api("GET", "/servers")


@server.tool(annotations=READ)
async def list_assets(kind: str = "", query: str = "", limit: int = 30) -> dict[str, Any]:
    """Use this to find assets: newest first, optionally only one kind (image, mesh) and only those
    whose title, id or tags contain `query`. limit 1-200."""
    params = {"kind": kind} if kind else None
    assets = (await api("GET", "/assets", params=params))["assets"]
    q = query.lower()
    if q:
        assets = [a for a in assets if q in (a.get("title") or "").lower() or q in a["id"]
                  or any(q in t.lower() for t in a.get("tags") or [])]
    limit = max(1, min(limit, 200))
    return {"assets": [_summary(a) for a in assets[:limit]], "total": len(assets)}


@server.tool(annotations=READ)
async def get_asset(asset_id: str) -> dict[str, Any]:
    """Use this for one asset's details and provenance: the job that produced it (server, task,
    prompt, params, seed, inputs) and, for a Blender edit, the asset it was derived from."""
    asset = (await api("GET", f"/assets/{asset_id}"))["asset"]
    result: dict[str, Any] = {"asset": _summary(asset)}
    if asset.get("producer"):
        with contextlib.suppress(ToolError):
            result["produced_by"] = _job_summary(await _job(asset["producer"]))
    derived = [t.split(":", 1)[1] for t in asset.get("tags") or [] if t.startswith("derived:")]
    if derived:
        result["derived_from"] = derived[0]
    return result


async def _render_model(data: bytes, size: int) -> tuple[bytes, dict[str, Any]]:
    digest = hashlib.sha256(data).hexdigest()
    folder = CACHE / "looks"
    folder.mkdir(parents=True, exist_ok=True)
    sheet, info = folder / f"{digest}-{size}.png", folder / f"{digest}-{size}.json"
    if sheet.is_file() and info.is_file():
        return sheet.read_bytes(), json.loads(info.read_text())
    blender = os.environ.get("BLENDER") or shutil.which("blender")
    if not blender:
        raise ToolError("no_blender: set BLENDER to the Blender executable to render models")
    with tempfile.TemporaryDirectory(prefix="asset-studio-look-") as work:
        model = Path(work) / "model.glb"
        model.write_bytes(data)
        out = Path(work) / "sheet.png"
        process = await asyncio.create_subprocess_exec(
            blender, "-b", "--factory-startup", "--python-exit-code", "1",
            "--python", str(_ops("render_sheet.py")), "--", str(model), str(out), str(size),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            log, _ = await asyncio.wait_for(process.communicate(), timeout=300)
        except asyncio.TimeoutError:
            process.kill()
            raise ToolError("render_timeout: Blender took over 5 minutes to render the sheet") from None
        lines = log.decode(errors="replace").splitlines()
        found = [line[7:] for line in lines if line.startswith("RESULT ")]
        if process.returncode != 0 or not found or not out.is_file():
            tail = "\n".join(lines[-8:])
            raise ToolError(f"render_failed: Blender could not render the model:\n{tail}")
        details = json.loads(found[-1])
        shutil.copyfile(out, sheet)
    info.write_text(json.dumps(details))
    return sheet.read_bytes(), details


def _fit_picture(data: bytes, size: int) -> bytes:
    from PIL import Image as Picture

    picture = Picture.open(io.BytesIO(data))
    picture.thumbnail((size, size))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


@server.tool(annotations=READ, structured_output=False)
async def look(asset_id: str, size: int = 768) -> list[Any]:
    """Use this to see an asset: a picture as itself, a 3D model as a 2x2 sheet rendered in Blender
    (front, right, back, three-quarter) with its dimensions in metres, triangle and material
    counts. size 256-1024. Rendering a model takes 5-30 s the first time; it is then cached."""
    size = max(256, min(size, 1024))
    asset = (await api("GET", f"/assets/{asset_id}"))["asset"]
    data = await api("GET", f"/assets/{asset_id}/download", raw=True, timeout=300)
    if asset["mime"].startswith("image/"):
        return [Image(data=_fit_picture(data, size), format="png"), f"{asset['title'] or asset_id}: picture"]
    if asset["mime"] == "model/gltf-binary" or data[:4] == b"glTF":
        png, details = await _render_model(data, size)
        return [Image(data=png, format="png"), json.dumps({"asset": asset_id, "title": asset["title"]} | details)]
    raise ToolError(f"not_viewable: {asset['mime']} assets cannot be shown")


# --- making ----------------------------------------------------------------------------------


@server.tool(annotations=ADD)
async def generate_3d_from_text(prompt: str, seed: int | None = None, height_m: float | None = None,
                                resolution: str = "1024_cascade", texture_size: int = 2048,
                                wait_s: float = 40, ctx: Context = None) -> dict[str, Any]:
    """Use this to make a textured 3D model (GLB) from a description: Z-Image draws the object on a
    plain background, BiRefNet cuts it out, TRELLIS.2 builds the model. Describe one object, its
    material and style. height_m scales it to that height, standing on the ground. resolution
    "512" (fast) | "1024_cascade" | "1536_cascade" (detailed). About 3 min from cold. Waits up to
    wait_s (<= 45) and returns {status: done, output_assets} or {status: running} to continue
    with job_status; the picture and cut-out are kept as assets too."""
    body = {"prompt": prompt, "seed": seed, "height_m": height_m,
            "model_params": {"resolution": resolution, "texture_size": texture_size}}
    submitted = await api("POST", "/pipelines/text-to-3d", json_body=body)
    ids = [j["id"] for j in submitted["jobs"]]
    return _result(await _wait(ids, wait_s, ctx, own=True))


@server.tool(annotations=ADD)
async def run_job(server: str, task: str, inputs: dict[str, Any] | None = None, prompt: str | None = None,
                  params: dict[str, Any] | None = None, count: int = 1, seed: int | None = None,
                  wait_s: float = 40, ctx: Context = None) -> dict[str, Any]:
    """Use this to run one model task (see list_servers and asset-studio://guide), e.g. birefnet
    image-to-image to cut out a picture, trellis2_3d image-to-3d, trellis2 3d-paint, zimage
    text-to-image. inputs map a role to an asset id, or to another job's output as
    {"job": "<id>", "output": 0}, so jobs can be chained before the first finishes. Waits up to
    wait_s (<= 45): {status: done, output_assets} or {status: running} (then job_status)."""
    refs: dict[str, Any] = {}
    for role, ref in (inputs or {}).items():
        refs[role] = {"asset": ref} if isinstance(ref, str) else ref
    body = {"server": server, "task": task, "inputs": refs, "prompt": prompt, "params": params or {},
            "count": count, "seeds": [seed] if seed is not None else None}
    job = (await api("POST", "/jobs", json_body=body))["job"]
    return _result(await _wait([job["id"]], wait_s, ctx, own=True))


@server.tool(annotations=READ)
async def job_status(job_id: str, wait_s: float = 0, ctx: Context = None) -> dict[str, Any]:
    """Use this to follow a job a tool left running. For a job whose inputs come from other jobs,
    those are included (in order). Waits up to wait_s (<= 45) for it to finish."""
    chain, current = [], await _job(job_id)
    while True:
        chain.insert(0, current["id"])
        upstream = [r["JobOutput"]["job_id"] for r in (current.get("inputs") or {}).values() if "JobOutput" in r]
        if not upstream or len(chain) > 16:
            break
        current = await _job(upstream[0])
    return _result(await _wait(chain, wait_s, ctx, own=False))


@server.tool(annotations=STOP)
async def cancel_job(job_id: str) -> dict[str, Any]:
    """Use this to stop a queued or running job; jobs that take its output then fail. Nothing
    already made is removed."""
    return {"job": _job_summary((await api("POST", f"/jobs/{job_id}/cancel"))["job"])}


@server.tool(annotations=ADD)
async def retry_job(job_id: str, wait_s: float = 0, ctx: Context = None) -> dict[str, Any]:
    """Use this to run a failed job again with the same inputs and params (a new job)."""
    job = (await api("POST", f"/jobs/{job_id}/retry"))["job"]
    return _result(await _wait([job["id"]], wait_s, ctx, own=True))


@server.tool(annotations=ADD)
async def import_file(path: str, title: str = "", tags: list[str] | None = None) -> dict[str, Any]:
    """Use this to add a local file to the library: an absolute path to a picture (.png .jpg .jpeg
    .webp) or a model (.glb). Returns the new asset. To make a model from a picture, cut it out
    first: run_job birefnet image-to-image with {"image": <asset id>}."""
    file = Path(path)
    if not file.is_absolute() or not file.is_file():
        raise ToolError(f"bad_path: {path!r} must be the absolute path of an existing file")
    kind = (IMAGE_EXTENSIONS | MODEL_EXTENSIONS).get(file.suffix.lower())
    if kind is None:
        raise ToolError(f"bad_type: {file.suffix!r}; supported: .png .jpg .jpeg .webp .glb")
    if file.stat().st_size > 1 << 30:
        raise ToolError("too_large: files over 1 GB are not imported")
    params = {"kind": kind, "title": title or file.stem}
    if tags:
        params["tags"] = ",".join(tags)
    asset = (await api("POST", "/assets/import", content=file.read_bytes(), params=params, timeout=300))["asset"]
    return {"asset": _summary(asset)}


@server.tool(annotations=ADD)
async def export_asset(asset_id: str, path: str) -> dict[str, Any]:
    """Use this to save an asset to a file: `path` is an absolute path in an existing directory,
    for example /home/me/game/assets/barrel.glb. An existing file is never overwritten."""
    target = Path(path)
    if not target.is_absolute() or not target.parent.is_dir():
        raise ToolError(f"bad_path: {path!r} must be absolute, in an existing directory")
    data = await api("GET", f"/assets/{asset_id}/download", raw=True, timeout=300)
    try:
        with target.open("xb") as file:
            file.write(data)
    except FileExistsError:
        raise ToolError(f"exists: {path} exists; choose another name") from None
    return {"path": str(target), "bytes": len(data)}


# --- Blender ----------------------------------------------------------------------------------


def _blender_address() -> tuple[str, int]:
    return os.environ.get("BLENDER_MCP_HOST", "localhost"), int(os.environ.get("BLENDER_MCP_PORT", "9876"))


def _blender_run(script: str, args: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
    """Run a blender_ops script in the live Blender through the Blender MCP add-on's socket."""
    code = f"ARGS = {json.dumps(args)!s}\n" + _ops(script).read_text()
    request = json.dumps({"type": "execute", "code": code, "strict_json": True}) + "\0"
    host, port = _blender_address()
    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            connection.sendall(request.encode())
            buffer = bytearray()
            while b"\0" not in buffer:
                chunk = connection.recv(65536)
                if not chunk:
                    break
                buffer.extend(chunk)
    except OSError as error:
        raise ToolError(f"no_blender: Blender's MCP add-on is not reachable at {host}:{port} ({error}); "
                        "open Blender with the MCP add-on enabled") from None
    if b"\0" not in buffer:
        raise ToolError("blender_error: Blender closed the connection without a reply")
    response = json.loads(buffer[: buffer.index(b"\0")])
    if response.get("status") != "ok":
        message = str(response.get("message", response))
        raise ToolError(f"blender_error: {message.strip().splitlines()[-1] if message.strip() else message}")
    return response.get("result") or {}


@server.tool(annotations=SAME)
async def open_in_blender(asset_id: str) -> dict[str, Any]:
    """Use this to load a 3D model asset into the running Blender (the Blender MCP add-on must be on)
    so it can be inspected or edited there, e.g. with the Blender MCP tools. It goes into its own
    collection "AS <title> [<id>]"; opening it again reuses that collection."""
    asset = (await api("GET", f"/assets/{asset_id}"))["asset"]
    if asset["mime"] != "model/gltf-binary":
        raise ToolError(f"not_a_model: {asset_id} is {asset['mime']}; only GLB models open in Blender")
    folder = CACHE / "blender"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{asset_id}.glb"
    if not path.is_file():
        path.write_bytes(await api("GET", f"/assets/{asset_id}/download", raw=True, timeout=300))
    args = {"asset_id": asset_id, "title": asset.get("title") or "", "path": str(path)}
    return await asyncio.to_thread(_blender_run, "live_open.py", args)


@server.tool(annotations=ADD)
async def pull_from_blender(asset_id: str, title: str = "") -> dict[str, Any]:
    """Use this after editing an opened asset in Blender: exports its collection as it is now and
    saves it as a NEW asset (the original is unchanged), tagged blender-edit and
    derived:<asset_id>. Returns the new asset."""
    source = (await api("GET", f"/assets/{asset_id}"))["asset"]
    with tempfile.TemporaryDirectory(prefix="asset-studio-pull-") as work:
        path = Path(work) / "edit.glb"
        exported = await asyncio.to_thread(_blender_run, "live_pull.py", {"asset_id": asset_id, "path": str(path)})
        if not path.is_file():
            raise ToolError("blender_error: the export wrote no file")
        data = path.read_bytes()
    name = title or f"{source.get('title') or asset_id} (Blender edit)"
    params = {"kind": "mesh", "title": name, "tags": "blender-edit", "derived_from": asset_id}
    asset = (await api("POST", "/assets/import", content=data, params=params, timeout=300))["asset"]
    return {"asset": _summary(asset), "objects": exported.get("objects", [])}


@server.tool(annotations=STOP)
async def close_in_blender(asset_id: str) -> dict[str, Any]:
    """Use this to remove an opened asset's collection and objects from the Blender scene (unsaved
    edits there are lost; pull_from_blender first to keep them). The library is unchanged."""
    return await asyncio.to_thread(_blender_run, "live_close.py", {"asset_id": asset_id})


@server.resource("asset-studio://guide", mime_type="text/markdown")
def guide() -> str:
    """How to use asset-studio: servers, task parameters, chaining jobs, Blender."""
    return GUIDE


def main() -> None:
    import logging

    # stdio carries the protocol; keep stderr for warnings, not every HTTP request.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    server.run()


if __name__ == "__main__":
    main()
