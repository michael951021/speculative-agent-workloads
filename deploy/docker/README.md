# Images

Build context is the repo root (`.dockerignore` there excludes data/models/ckpt).
Sandbox has no Docker: lint only (`make check` → hadolint). Build on the host.

| Image | Dockerfile | Base | Notes |
|---|---|---|---|
| `draftlab/gateway` | Dockerfile.gateway | golang:1.26.8-alpine → gcr.io/distroless/static-debian12:nonroot | multi-stage, static binary, uid 65532, :8200, `-config /app/config.yaml` (chart mounts its own) |
| `draftlab/engine` | Dockerfile.engine | python:3.12-slim | CPU; `.[jax,serve]` installed; uid 10001; :8100; **models mounted at /app/models** (not baked in) |
| `draftlab/vllm` | Dockerfile.vllm | vllm/vllm-openai:v0.11.2 | entrypoint adds `--speculative-config` when a drafter adapter (with adapter_config.json) is mounted at `DRAFTER_PATH` (default /drafter, PVC); set `VLLM_MODEL`, optional `DRAFTER_K` |

```sh
docker build -f deploy/docker/Dockerfile.gateway -t draftlab/gateway .
docker build -f deploy/docker/Dockerfile.engine  -t draftlab/engine  .
docker build -f deploy/docker/Dockerfile.vllm    -t draftlab/vllm    .
```

Local smoke:
```sh
docker run -v $PWD/models:/app/models -p 8100:8100 draftlab/engine
docker run -p 8200:8200 draftlab/gateway
```
