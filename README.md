# asset-studio

A Rust reimplementation of the `assets_studio` 3D asset pipeline: a Rust core
built on the `base-*` crate family that serves an HTTP API and orchestrates
Python model servers.

The Rust side owns the **asset library**, the **job queue + state machine**,
the **model-server process lifecycle**, and the **REST API**. The models
themselves (TRELLIS.2 Paint, Hunyuan3D, Hi3D, animation tools) stay in
Python — each runs as a **separate process on its own port**, spawned by the
launcher and reached over HTTP. No PyO3, no model port.

## Crate layout

| Crate | Name | Role |
| --- | --- | --- |
| `crates/core` | `asset-studio-core` | Pure domain: `Asset`, `Job`, `JobState` state machine, input resolution. Scheduler-agnostic, synchronous. |
| `crates/library` | `asset-studio-library` | Persistence: SQLite (assets/jobs/servers) + content-addressed `BlobStore`. |
| `crates/launcher` | `asset-studio-launcher` | Model-server process management: spawn/poll/kill, per-server GPU slots. |
| `crates/models` | `asset-studio-models` | Model catalog + typed `ModelClient` over `base-http-client`. |
| `crates/server` | `asset-studio-server` | The REST API (`base-http-server` on `base-scheduler-native`) + the async job orchestrator. |
| `crates/client` | `asset-studio-client` | The `asset-studio` CLI: `servers`, `list`, `import`, `generate`, `wait`, `download`. |

## Build & test

```sh
cargo build --workspace
cargo test --workspace
```

## Run

```sh
# Server
cargo run -p asset-studio-server -- --data ~/.asset-studio --listen 127.0.0.1:8090

# CLI (in another shell)
cargo run -p asset-studio-client --bin asset-studio -- --studio 127.0.0.1:8090 servers
cargo run -p asset-studio-client --bin asset-studio -- import chicken.glb --kind mesh
cargo run -p asset-studio-client --bin asset-studio -- \
    generate --server trellis2 --task 3d-paint \
    --input mesh=chicken.glb --input image=ref.png --wait
```

## API

- `GET /` — service root (server list, status)
- `GET /servers` — model servers + tasks
- `GET /assets[?kind=]` / `GET /assets/{id}` / `GET /assets/{id}/download`
- `POST /assets/import?kind=image|mesh&title=…` (raw body)
- `PUT /assets/{id}/rating` — set a 0–5 rating
- `DELETE /assets/{id}` — soft-delete
- `GET /jobs[?state=]` / `GET /jobs/{id}`
- `POST /jobs` — submit `{server, task, inputs, prompt, params, count, seeds, idempotency_key}`
- `POST /jobs/{id}/cancel` / `POST /jobs/{id}/retry`
- `GET /health`

Client-side errors are returned as JSON with the correct status (404 unknown
asset/server/task, 400 bad input, 409 duplicate idempotency key / illegal
transition). The server only closes the connection on genuine internal
failure.

## Model servers

The `models` crate declares a catalog of model servers. Each entry maps to a
Python process (via `uv`) on a fixed port. To run a real model (e.g. the
TRELLIS.2 paint vertical slice), you need the Python/CUDA environment (see the
`justfile` / Nix flake in the reference repo) — that is independent of the Rust
structure. The server boots the model server on demand when a job for it is
queued.
