# gateway

## Issue
Engine server (:8100) has no auth, rate limiting, or observability, and one endpoint can't
front multiple backends (engine, llama-server, vLLM). Dashboards (next ticket) need live
request/TTFT/acceptance metrics from a single authenticated OpenAI-compatible endpoint.

## Question
Can one Go gateway proxy N backends correctly (incl. SSE passthrough), rate-limit per key,
isolate failing backends (breaker), and emit the metrics the dashboards consume?
Refuting observation: 401/429/breaker/stream-order wrong in httptest, or counters off vs
the known usage the fake backend sends.

## Evidence
- `draftlab/serve/app.py:231` — usage = {prompt, completion, total, draft, accepted}_tokens;
  streams put usage in the final data chunk before `data: [DONE]`.
- Smallest check: `cd gateway && go test ./...` (httptest fake backends); logs in
  .agent-evidence/gateway-*.log. Observed: nothing yet (fresh dir).

## Done when
`cd gateway && go vet ./... && go test ./...` pass with: bad key 401; rate-limited 429 +
Retry-After; SSE order preserved; breaker opens after 5 failures, half-opens, closes on
success; /metrics has requests_total{key,model,status}, TTFT + latency histograms,
output tok/s gauge, draft/accepted token counters.
Known-answer check: fake backend sends usage draft=12, accepted=7 per request; after 2
requests counters must be exactly 24/14; breaker_state gauge exactly 1 (open) after 5
5xx failures, and no upstream hit while open.
Plausible range: rate=50/s burst=2 → 3rd rapid request is 429 with Retry-After >= 1s.

## Stop when
No TLS, no OAuth2, no body rewriting, no web/ build (serve static dir only).
If client_golang/yaml won't fetch, split the metrics part out. Two attempts without new
evidence → record blocker, stop.

## Plan
1. go.mod + config.go (yaml; sha256 key hashes; backends/models; rate; breaker params).
2. auth.go (Bearer + subtle compare), ratelimit.go (token bucket).
3. breaker.go + backend.go (retries on transport errors, health checks, model routing).
4. proxy.go: SSE tee w/ flush + usage scan; non-stream usage; /v1/models merge; /healthz; /ui.
5. metrics.go (prometheus). 6. handler_test.go per Done-when.
