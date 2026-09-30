
## gateway — Go gateway
- Goal: gateway/ (Go module, stdlib net/http + prometheus/client_golang + slog): OpenAI-compatible reverse proxy for /v1/chat/completions and /v1/models; API keys stored as SHA-256 hashes in config.yaml, Bearer auth; per-key token-bucket rate limit (429 + Retry-After); routing by model name to backends (eng
- Approach: Go gateway at gateway/ — OpenAI-compatible proxy with Bearer auth (sha256 keys in config.yaml), per-key token-bucket rate limit (429+Retry-After), model→backend routing with retries and a 5-failure circuit breaker (half-open probe), SSE passthrough with per-read flushing, usage→Prometheus metrics (requests by key/model/status, TTFT/latency histograms, output tok/s, draft/accepted counters, backend_up, breaker_state), /v1/models merge, /healthz, /ui static; 8/8 httptest acceptance tests pass, make check green, live smoke vs a real fake backend verified 401/429/stream-order and exact draft/accep
- Blockers: none
