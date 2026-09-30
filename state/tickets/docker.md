# Dockerfiles (id: docker)

## Issue
The Helm chart (next ticket) deploys three images the repo does not produce: gateway, engine, vllm. Without Dockerfiles there is nothing to push to Artifact Registry. Sandbox has no Docker, so lint (hadolint) is the gate.

## Question
Can deploy/docker/ hold three Dockerfiles (gateway multi-stage → distroless static non-root; engine python:3.12-slim CPU; vllm on vllm/vllm-openai with drafter-adapter mount) that hadolint passes in `make check`, and whose entrypoints match what the code actually listens on?
Refutation: hadolint flags a Dockerfile, or an ENTRYPOINT/port contradicts gateway/main.go (`-config`, :8200) or draftlab.serve (`--port 8100`).

## Evidence
- tools/check.sh already lints `deploy/docker/Dockerfile*` (hadolint v2.12.0 in .tools/bin).
- gateway/main.go:16 `-config` flag; config.yaml `listen: ":8200"`, backends → engine :8100.
- draftlab/serve/app.py:242 main: `--model/--alias/--port 8100`; models from `<repo>/models` (draftlab/models.py:14).
- Check: `cd deploy/docker && hadolint Dockerfile.*` (exit 0). Output in .agent-evidence/hadolint.log.
- Known-answer checks (no measured numbers here): static-binary claim → `go env` GOOS/CGO note; ports cross-checked against source (observed above).

## Done when
`make check` hadolint section exits 0 for all three Dockerfiles (run first: fails — files absent, section skips; after write: passes). .dockerignore present at build context root. README notes image names + build commands.
No new runtime behavior → no regression tests beyond existing hadolint gate.

## Stop when
Out of scope: Helm chart, Terraform, kind e2e, actually building images (no Docker in sandbox). Two failed hadolint iterations without new evidence → record and stop.

## Plan
1. Dockerfile.gateway: golang:1.26.8-alpine build (CGO_ENABLED=0, -ldflags -s -w) → gcr.io/distroless/static:nonroot, USER 65532, EXPOSE 8200.
2. Dockerfile.engine: python:3.12-slim, pip install draftlab[jax,serve], non-root user, models mounted at /app/models.
3. Dockerfile.vllm: vllm/vllm-openai base + start script honoring DRAFTER_PATH mount (speculative-config), non-root.
4. .dockerignore at repo root (context = repo root, so gateway/ + pyproject are reachable); .hadolint.yaml to pin-tag bases.
5. hadolint until clean; README; NOTES.md; commit.
