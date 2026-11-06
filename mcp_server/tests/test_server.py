import asyncio

from asset_studio_mcp import server as s


def job(id, state, server="zimage", outputs=(), error=None, inputs=None):
    return {"id": id, "server": server, "task": "t", "state": state, "outputs": list(outputs),
            "error": error, "inputs": inputs or {}, "params": {}, "seeds": None, "prompt": None}


def test_tools_and_guide_are_registered():
    names = {t.name for t in asyncio.run(s.server.list_tools())}
    assert {"look", "generate_3d_from_text", "run_job", "job_status", "open_in_blender",
            "pull_from_blender", "export_asset", "import_file"} <= names
    tools = {t.name: t for t in asyncio.run(s.server.list_tools())}
    assert tools["look"].annotations.read_only_hint
    assert tools["cancel_job"].annotations.destructive_hint
    assert not tools["pull_from_blender"].annotations.destructive_hint


def test_blender_ops_are_found():
    for name in ("render_sheet.py", "live_open.py", "live_pull.py", "live_close.py"):
        assert s._ops(name).is_file()


def test_result_reports_done_running_and_failed_stages():
    done = s._result([job("a", "succeeded", outputs=["x"]), job("b", "succeeded", outputs=["glb"])])
    assert done["status"] == "done" and done["output_assets"] == ["glb"]
    running = s._result([job("a", "succeeded"), job("b", "queued")])
    assert running["status"] == "running" and "b" in running["next"]
    failed = s._result([job("a", "failed", error="boom"), job("b", "queued")])
    assert failed["status"] == "failed" and failed["error"] == "boom"


def test_job_summary_flattens_inputs():
    summary = s._job_summary(job("b", "queued", inputs={
        "image": {"JobOutput": {"job_id": "a", "output_index": 0}},
        "mesh": {"Asset": {"id": "m1", "role": "mesh"}},
    }))
    assert summary["inputs"] == {"image": {"job_id": "a", "output_index": 0}, "mesh": "m1"}
