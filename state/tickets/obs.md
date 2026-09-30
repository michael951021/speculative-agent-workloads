# obs — Prometheus and Grafana (observability as code)

## Issue
No deployable monitoring: gateway metrics exist (`gateway/metrics.go`) but there is no
Prometheus config, no recording rules/alerts, no dashboard, and `make check` has no promtool
hook. Ops can't see acceptance rate, TTFT or breaker state without hand-writing queries.

## Question
Can promtool validate `deploy/prometheus/{prometheus.yml,rules.yml}` and a unit-test file
(`promtool test rules`) where the rules reproduce hand-computed values on synthetic series?
Refutation: promtool config/rules/test failure, or test asserting a value the rules don't produce.

## Evidence
Observed: gateway metrics at `:8200/metrics` — `draftlab_requests_total{key,model,status}`,
`draftlab_ttft_seconds` hist{model}, `draftlab_request_latency_seconds`, `draftlab_output_tokens_per_second`,
`draftlab_draft_tokens_total`, `draftlab_accepted_tokens_total` (no model label → must add one for
"acceptance by model"), `draftlab_backend_up{backend}`, `draftlab_breaker_state{backend}` (0/1/2),
`draftlab_rate_limited_total{key}`. `tools/check.sh:75-90` has a promtool section but expects
`*.rules`; ticket wants `rules.yml` → check.sh updated.
Check: `promtool test rules deploy/prometheus` (files below); known-answer values hand-computed
before the run: acceptance ratio 50/100 = exactly 0.5; ttft p95 with 15 obs/min (5 ≤2.5s, 10 in
(2.5,5]s) → histogram_quantile linear interp = 2.5+2.5·9.25/10 = 4.8125; p50 = 3.125.
Plausible range: acceptance ∈ [0,1]; alerts fire only after `for` elapses (acceptance at t≥15m,
pending at 5–15m).

## Done when
1. `promtool check config deploy/prometheus/prometheus.yml` → OK
2. `promtool check rules deploy/prometheus/rules.yml` → OK
3. `promtool test rules deploy/prometheus` → "Success: 0 failures" (recording values + alert
   states vs hand-computed above)
4. `deploy/grafana/draftlab.json` parses as JSON (existing check.sh hook) and its panel queries
   use only the recording rules/metrics above
5. `make check` green; gateway `go test ./...` green after adding `model` label to the two
   token counters (test updated: `draftlab_draft_tokens_total{model="m1"} 24`).

## Stop when
Only deploy/prometheus, deploy/grafana, tools/check.sh prometheus section, and the gateway
label addition. No Grafana server, no k8s manifests (Helm ticket's job), no alertmanager config.
Stop/split after two attempts if promtool test-rule semantics (state timeline) fight the design.

## Plan
1. Write this ticket; run acceptance commands (expected to fail: files absent).
2. Gateway: add `model` label to draft/accepted counters + update test.
3. Write prometheus.yml (scrape 127.0.0.1:8200, rule_files), rules.yml (recordings + 2 alerts),
   rules_test.yml; iterate until promtool test passes.
4. Grafana dashboard JSON (5 panels, datasource template var).
5. Update check.sh prometheus section; run make check; NOTES.md; commit.
