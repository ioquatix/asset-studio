# The asset-studio MCP server

`mcp_server/` is an MCP server (stdio) that gives an agent the studio's
library and generation jobs, plus a bridge to the live Blender. It is a thin
client of the studio's REST API, so `asset-studio-server` must be running; the
design follows [Effigy](https://github.com/SifuInTheShell/Effigy)'s MCP server.

```sh
scripts/setup_mcp.sh          # ~/.cache/assets-studio/venvs/mcp, runs the tests
```

Register it with your agent, e.g. Claude Code:

```sh
claude mcp add asset-studio -- ~/.cache/assets-studio/venvs/mcp/bin/asset-studio-mcp
```

or OpenCode (`opencode.json`):

```json
{ "mcp": { "asset-studio": { "type": "local",
  "command": ["/home/<you>/.cache/assets-studio/venvs/mcp/bin/asset-studio-mcp"] } } }
```

Environment: `ASSET_STUDIO_URL` (default `http://127.0.0.1:8090`), `BLENDER`
(the Blender executable for `look`; default `blender` on `PATH`),
`BLENDER_MCP_HOST`/`BLENDER_MCP_PORT` (the Blender MCP add-on; default
`localhost:9876`).

## Tools

| Group | Tools |
|---|---|
| Browse | `list_servers`, `list_assets`, `get_asset` (with the producing job, or the asset a Blender edit came from), `look` |
| Make | `generate_3d_from_text`, `run_job`, `job_status`, `cancel_job`, `retry_job`, `import_file`, `export_asset` |
| Blender | `open_in_blender`, `pull_from_blender`, `close_in_blender` |

The resource `asset-studio://guide` lists every task's parameters.

- **`look`** returns an image the agent can see: a picture as itself, a model
  as a 2×2 sheet (front, right, back, three-quarter) rendered by a headless
  Blender, with its dimensions in metres and triangle count. Sheets are cached
  by content hash under `~/.cache/assets-studio/mcp/looks/`.
- **Jobs** wait at most 45 s per call, reporting progress (on a change, else
  every 10 s), then return `status: running` and a job id for `job_status`.
  If the agent cancels a call that started jobs, those jobs are cancelled.
  `job_status` on the last job of a chain reports every stage.
- **Waiting for the GPU**: while a ready job cannot start (another model, or
  another program such as Ollama, holds the GPU or its memory), the job has a
  `waiting` field with the reason, and the tools pass it on. An agent served
  by a *local* LLM on the same GPU will often see this: unload the LLM (e.g.
  `ollama stop <model>`) or use a smaller one while generating.
- **Nothing is deleted or overwritten**: generations and Blender edits are new
  assets; `export_asset` refuses an existing file. Only `cancel_job` and
  `close_in_blender` are marked destructive.
- **Blender**: `open_in_blender` imports a model into its own collection
  `AS <title> [<id>]` in the running Blender (Blender MCP add-on on);
  edit it there with the Blender MCP tools; `pull_from_blender` exports that
  collection and saves it as a new asset tagged `blender-edit` and
  `derived:<original id>`.

The scripts in `mcp_server/blender_ops/` run inside Blender and are
GPL-3.0-or-later; the server only sends or launches them and never imports
`bpy`, so the rest of `mcp_server/` is MIT.

## Verified (RTX 4090)

Through a real MCP client session over stdio:
`generate_3d_from_text("a wooden barrel with iron bands, stylised game asset",
height_m=1.0)` finished in 2 min 45 s from cold (three model hand-overs);
the three stage assets were titled after the prompt; `look` rendered the
1.0 m, 200k-triangle barrel; `export_asset` wrote the GLB and refused to
overwrite it. A Blender round trip (open, scale ×2 through the add-on socket,
pull) produced a new asset of twice the height with `derived_from` set;
`close_in_blender` removed the collection.
