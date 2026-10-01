# kind end-to-end script

## Issue
Kevin has Helm chart + Dockerfiles but no one-command proof they fit together. The chart,
images, gateway auth and engine must actually run in a real (small) cluster, or the deploy
section is unverified paperwork.

## Question
Can `deploy/kind/e2e.sh` stand up a kind cluster, install the cpu profile with the tiny
real model, and show an authed chat completion plus `draftlab_accepted_tokens_total > 0`
in gateway /metrics? Refutation: any step failing (image build, helm install, auth 401,
metric absent/zero) with the cluster in front of us.

## Evidence
- deploy/helm/draftlab/{values.yaml,templates/*}: cpu profile, hostPath modelVolume, gateway
  sha256 auth (secret-config.yaml), metric name draftlab_accepted_tokens_total (gateway/metrics.go).
- Engine defaults: bare model name → ngram drafter k=5 (draftlab/serve/app.py:25-26).
- Sandbox has no Docker (rule 2): script is host-only; here we validate with shellcheck +
  bash -n + a dry-run of the arg/env parsing only. No cluster is run in the sandbox.

## Done when
- `shellcheck deploy/kind/e2e.sh` → 0 findings; `bash -n` clean.
- Script does: build 2 images → kind create/load → docker cp model to node (hostPath
  /models) → helm install cpu profile (1 replica, HPA/ServiceMonitor/NP off) → port-forward →
  Bearer-keyed curl /v1/chat/completions (200) → grep /metrics for
  `draftlab_accepted_tokens_total` sum > 0. Known-answer: metric value is the sum of
  usage.accepted_tokens from real completions (gateway test gateway_test.go:254 covers the
  counter wiring); the repetition prompt makes ngram acceptance ~1 on the repeated span.
- deploy/kind/README.md documents the command, prerequisites, teardown.
- Not run in the sandbox (no Docker); Kevin runs it on the host.

## Stop when
Docker/kind unavailable in sandbox ⇒ shellcheck + static review is the ceiling for this run.
Do not modify the Helm chart, gateway, or engine to make e2e pass (separate tasks). Two
failed attempts at any external check ⇒ record blocker and stop.

## Plan
1. Write e2e.sh (env-configurable cluster/model/tag/port; trap cleans port-forward).
2. Get a shellcheck binary (.tools or download), run, fix.
3. README in deploy/kind/.
4. Commit.
