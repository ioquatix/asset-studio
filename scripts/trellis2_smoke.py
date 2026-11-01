#!/usr/bin/env python3
"""Exercise the Rust studio API against a real TRELLIS.2 paint server.

Inputs must be a valid GLB mesh and a cut-out RGBA PNG. No Python packages
are required by this client; model setup is separate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen


def glb_document(data):
    if len(data) < 20:
        raise ValueError("output is not a GLB")
    magic, version, length, chunk_length, chunk_type = struct.unpack_from("<4sIIII", data)
    if magic != b"glTF" or version != 2 or length != len(data) or chunk_type != 0x4E4F534A:
        raise ValueError("invalid GLB header")
    return json.loads(data[20:20 + chunk_length])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--studio", default="http://127.0.0.1:8090")
    parser.add_argument("--output", type=Path, default=Path("painted.glb"))
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()
    base = args.studio.rstrip("/")

    def request(path, data=None):
        headers = {"Content-Type": "application/json"} if data is not None else {}
        if isinstance(data, dict):
            data = json.dumps(data).encode()
        with urlopen(Request(base + path, data=data, headers=headers), timeout=30) as response:
            return response.read()

    inputs = {}
    for role, kind, path in (("mesh", "mesh", args.mesh), ("image", "image", args.image)):
        payload = json.loads(request("/assets/import?kind=" + kind, path.read_bytes()))
        inputs[role] = {"asset": payload["asset"]["id"]}

    job = json.loads(request("/jobs", {
        "server": "trellis2", "task": "3d-paint", "inputs": inputs,
        "params": {"texture_size": 1024, "steps": args.steps},
        "count": 1, "seeds": [args.seed],
    }))["job"]
    job_id = job["id"]
    deadline = time.monotonic() + args.timeout
    last_state = None
    while True:
        if job["state"] != last_state:
            print(f"{job_id}: {job['state']}", flush=True)
            last_state = job["state"]
        if job["state"] in ("succeeded", "failed", "cancelled"):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"job {job_id} still {job['state']}; it has not been cancelled")
        time.sleep(1)
        job = json.loads(request("/jobs/" + job_id))["job"]
    if job["state"] != "succeeded":
        raise RuntimeError(job.get("error") or job["state"])
    if len(job["outputs"]) != 1:
        raise ValueError(f"expected one output, received {len(job['outputs'])}")

    asset_id = job["outputs"][0]
    asset = json.loads(request("/assets/" + asset_id))["asset"]
    data = request("/assets/" + asset_id + "/download")
    if asset["producer"] != job_id or asset["mime"] != "model/gltf-binary":
        raise ValueError("output asset has incorrect provenance or MIME type")
    if asset["size_bytes"] != len(data):
        raise ValueError("download does not match the library size")
    if hashlib.sha256(data).hexdigest() != asset["blob_sha256"]:
        raise ValueError("download does not match the library digest")
    document = glb_document(data)
    if not document.get("materials") or not document.get("images"):
        raise ValueError("painted GLB has no materials or embedded images")
    if not all("bufferView" in image for image in document["images"]):
        raise ValueError("painted GLB has externally referenced images")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    report = {
        "job": job, "asset": asset, "output": str(args.output),
        "materials": len(document["materials"]), "images": len(document["images"]),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "bytes": len(data), "asset": asset_id}))


if __name__ == "__main__":
    main()
