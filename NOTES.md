# draftlab — current state

## Toolchain (id: boot)
- `make bootstrap`: uv, Py3.12 .venv, torch CPU, .tools/; `make check` = ruff + pytest (others SKIP if absent); capture.py: no `VAR=x cmd` prefixes.

## Models (id: models)
- `models.py fetch` → models/{0.5b-instruct,1.5b-instruct,0.5b} (shared vocab 151936); `tiny_config()` = 2L d=64 vocab-512.

## Corpus (id: corpus)
- `python -m draftlab.data build` → data/corpus/{train,val,test}.parquet (code split, dedupe, trunc 4096, runs by sha1); 45257/667/566 blocks.

## N-gram (id: ngram)
- NGramMatcher (n=1..4, first occurrence, no self-match) → results/ngram.json; greedy accept@k=8 0.329 (tool_use 0.548 best, text 0.199 worst).

## JAX model (id: jaxmodel)
- jaxlm/model.py: pure-pytree Qwen2, bf16→f32, padded KV cache, one jitted step; 0.5B parity prefill 1.3e-4, 32 greedy identical.

## Spec decoding (id: spec)
- jaxlm/spec.py: one target call scores m drafts + bonus; accept u·q[t]<p[t], residual max(0,p−q); T=0==greedy; ModelDrafter full-accept rollback advances `_last` by m (regression-tested).

## Engine server (id: engine)
- serve/app.py FastAPI OpenAI-compatible (SSE, /v1/models); `python -m draftlab.serve` (0.5b-instruct, :8100); tiny → ToyTokenizer.

## Alpha metric (id: alpha)
- `python -m draftlab.eval.alpha` (ngram|jax|hf); row i predicts i+1. Known: CE==HF labels <1e-3; top-1 val 42.4%; uniform-q=1-TV; repeat 63/64, shuffle collapses.
- 400 val: ngram α 0.3341 (greedy 0.3681), tpc@8 1.50 (Edit 0.56 > tool_result 0.33 > code 0.26 > text 0.19); tiny α 0.0105; 0.5B control 0.7675 → weak drafter, not metric.

## Distill tiny (JAX, id: distill)
- drafters/distill.py: teacher top-64+tail, 4L d=256 KL AdamW → ckpt/tiny/export; 1200 steps val KL 3.79→2.63 ✅.

## LoRA (PyTorch, id: lora)
- drafters/lora.py: PEFT r=16/α32, 7 targets, SFT on packed blocks; --fsdp ⇒ bit-identical; step_loss==HF <1e-5; 300 steps → ckpt/lora.
- α 0.7306 [0.7302,0.7308] ✅; per-kind Edit 0.82 > Bash 0.77 > tool_result 0.74 > code 0.72 > text 0.66 (tool_use ≥ text ✅); tpc@8 3.49.

## DPO (id: dpo)
- drafters/dpo.py: T=1 spec pairs (1.5b target) → data/pairs/dpo.jsonl; TRL DPOTrainer β=0.1 pre-tokenized.
- 2000 pairs (accept 0.786); margin −3.79→−3.53; α 0.7306→0.7319 (CIs disjoint ✅).

## Kernels (id: triton)
- kernels/verify.py: fused_verify (online softmax, accept p>u·q, residual max(0,p−q)@first-reject), fused_alpha; tests 7/7 (TRITON_INTERPRET=1); alpha._alpha_sum kernel on CUDA only.

## Gateway (id: gateway)
- gateway/ Go 1.26: Bearer auth, 429+Retry-After, backend routing, breaker, SSE tee, /metrics (ttft, tok/s, draft/accepted{model}, 429, breaker_state), /v1/models, /healthz, /ui.
- `cd gateway && go vet && go test ./...` 8/8 (fake usage 12/7→24/14 exact); live smoke ✅.

## Observability (id: obs)
- deploy/prometheus: prometheus.yml + rules.yml (4 recordings, LowDraftAcceptance <0.4/10m, HighTTFTp95 >2s/5m) + rules_test.yml pinned; promtool SUCCESS.
- Gotcha: promtool windows LEFT-CLOSED [T−5m,T] (left-gap extrapolates: ts=4m → factor 1.25/300); recording tests need `__name__`.
- deploy/grafana/draftlab.json: 8 panels (acceptance, TTFT p50/p95, rate, draft/accepted, tok/s, 429, breaker). make check exit 0.

## Deploy docker (id: docker)
- deploy/docker/: gateway (golang:1.26.8-alpine → distroless nonroot, :8200), engine (python:3.12-slim, `.[jax,serve]`, :8100, models mounted /app/models), vllm (vllm-openai:v0.11.2; entrypoint adds --speculative-config from $DRAFTER_PATH if adapter_config.json present). Context = repo root; .dockerignore at root. hadolint 3/3 exit 0 (.agent-evidence/docker-hadolint.log).
- Gotcha: hadolint inline `# hadolint ignore=DLxxxx` must be the only text on its line (trailing note breaks it); DL3002 doesn't fire on third-party bases; DL3006 ignored in .hadolint.yaml.

## Ray sweep (id: sweep)
- `python -m draftlab.sweep` (Ray local, per-cell incremental parquet, resumable by (drafter,k,temp)) → results/sweep.parquet; target 0.5b-instruct, 488 val prompts (100/kind ≤128 tok), n_gen 16.
- Known: base@T0 k=1 accept exactly 1.0 (2-cell Ray test); k≥2 expect ≥0.99 (GEMM tiling ~1e-4).
- Cost: 6.4 tok/s @16 thr → 42 cells ≈ 20 CPU-h → batches: B1 base/lora/dpo k=1 T=0 + none; then ngram/tiny k=1 ×{0,0.7}; then k=8.

## Judge (id: judge)
- `python -m draftlab.eval.judge`: gen {target, spec(k=4, lora_dpo), drafter_alone} T=0.7 (49 prompts, n_gen 16) → blind pairwise vs qwen-agent 27B ($OLLAMA_BASE_URL), cache data/judge_cache/, cap 150 → results/judge.json. Tests 7/7.
- Result (finding, not fix): spec_vs_target n=81: win 0.444/loss 0.543, CIs overlap (binom p=0.434) — no significant loss; T=0.7 gives 0/46 exact matches so judge breaks near-ties ~50/50 = lossless signature (chi-square T=0.7 p>0.01). spec_vs_drafter n=24: gap 25pp ✓, underpowered (cap).
- Corpus gotcha: test.parquet block (contrib-loop-20260926-231240, 77976) appears 8× — code-split sub-blocks keep parent (run,seq) → 4 prompt entries share one output key (9 silent gen skips); no verdict contamination (all its jobs invalid/capped).
