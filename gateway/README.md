# gateway

OpenAI-compatible reverse proxy in front of the draft backends
(engine `python -m speculative_agent_workloads.serve`, llama-server, vLLM). Go stdlib +
prometheus/client_golang + slog.

## Run

```
cd gateway
go build -o gw .
./gw -config config.yaml        # listens on :8200 by default
```

`ui_dir`/`-config` paths are relative to the working directory; run from
`gateway/` with the repo-root-relative config as checked in.

## Behavior

- `POST /v1/chat/completions` — Bearer auth (keys stored as sha256 in
  `config.yaml`), per-key token-bucket rate limit (429 + `Retry-After`),
  model→backend routing, retries on transport errors, circuit breaker
  (5 consecutive failures → open → half-open probe after `half_open_after`).
  SSE passthrough flushes per read; `usage` (incl. `draft_tokens` /
  `accepted_tokens`) is pulled from the response for metrics.
- `GET /v1/models` — merged model list from all backends.
- `GET /metrics` — `speculative_agent_workloads_requests_total{key,model,status}`,
  `speculative_agent_workloads_ttft_seconds`, `speculative_agent_workloads_request_latency_seconds`,
  `speculative_agent_workloads_output_tokens_per_second`, `speculative_agent_workloads_draft_tokens_total`,
  `speculative_agent_workloads_accepted_tokens_total`, `speculative_agent_workloads_backend_up`,
  `speculative_agent_workloads_breaker_state`, `speculative_agent_workloads_rate_limited_total`.
- `GET /healthz` — per-backend up + breaker state.
- `/ui` — static file server for `web/dist`.

## Test

```
cd gateway && go vet ./... && go test ./...
```

httptest fake backends cover: 401 bad key, 429 + Retry-After, SSE order,
breaker open/half-open/close (with no upstream hit while open), metrics
exact counts (fake usage draft=12/accepted=7 per request → 24/14 after
two), merged /v1/models, 404 unknown model, config load.
