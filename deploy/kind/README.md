# kind end-to-end demo (host only)

`e2e.sh` proves the deploy stack fits together on one machine: build images → kind
cluster → load images → `helm install` (cpu profile, 0.5B model) → port-forward →
authed chat completion → `/metrics` shows accepted tokens.

Not run in the sandbox (no Docker; CLAUDE.md rule 2). Run on the host:

```sh
cd <repo>
# once: python -m draftlab.models fetch --name qwen2.5-0.5b-instruct
deploy/kind/e2e.sh
```

## What it checks

1. `docker build` of `draftlab/gateway` and `draftlab/engine` (contexts = repo root).
2. `kind create cluster` + `kind load docker-image`.
3. Model copied into the kind node (`docker cp`) and mounted by the engine via
   `engine.modelVolume.type=hostPath` (`/models/<model>` → container `/app/models`).
4. Chart installed with demo settings: 1 gateway replica, HPA/ServiceMonitor/
   Grafana-dashboard/NetworkPolicy off (none of those exist in a bare kind cluster),
   a random Bearer key (sha256 in the chart Secret).
5. `curl /v1/chat/completions` with the key → HTTP 200.
6. Gateway `/metrics`: `draftlab_accepted_tokens_total` sum > 0 — the engine's default
   drafter is `ngram k=5`, and the prompt asks the model to repeat a phrase, so ngram
   acceptance on the repeated span is ~1.

## Knobs

| Env | Default | Meaning |
|---|---|---|
| `DRAFTLAB_E2E_CLUSTER` | `draftlab-e2e` | kind cluster name (reused if it exists) |
| `DRAFTLAB_E2E_MODEL` | `qwen2.5-0.5b-instruct` | model dir under `models/` |
| `DRAFTLAB_E2E_TAG` | `e2e` | image tag for both images |
| `DRAFTLAB_E2E_PORT` | `18200` | local port for the port-forward |
| `DRAFTLAB_E2E_TEARDOWN` | `0` | `1` = helm uninstall + kind delete at the end |

The cluster and release are kept on success so re-runs skip the slow parts; the final
line prints the API key and a `port-forward` command for poking at it manually.

## Prereqs / costs

docker, kind, helm, kubectl, python3. Expect ~15–30 min on a laptop: image builds
(engine pip install dominates), cluster start, 0.5B CPU load + first JAX compile.
`make check` lints the script with shellcheck but never runs it.
