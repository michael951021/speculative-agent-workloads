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
- kernels/verify.py: fused_verify (online softmax, accept p>u·q, residual max(0,p−q)@first-reject), fused_alpha; tests 7/7 (TRITON_INTERPRET=1).
- alpha._alpha_sum: kernel on CUDA only; TRITON_INTERPRET read lazily.

## Gateway (id: gateway)
- gateway/ Go 1.26: Bearer auth, 429 bucket + Retry-After, backend routing, breaker, SSE tee, /metrics (requests, ttft, tok/s, draft/accepted{model}, 429, breaker_state), /v1/models, /healthz, /ui.
- `cd gateway && go vet && go test ./...` 8/8 (fake usage 12/7→24/14 exact); live smoke ✅.

## Observability (id: obs)
- deploy/prometheus: prometheus.yml + rules.yml (4 recordings, alerts LowDraftAcceptance <0.4/10m, HighTTFTp95 >2s/5m) + rules_test.yml pinned values; promtool test rules SUCCESS.
- Gotcha: promtool windows LEFT-CLOSED [T−5m,T] (left-gap extrapolates: ts=4m → factor 1.25/300); recording tests need `__name__`.
- deploy/grafana/draftlab.json: 8 panels (acceptance, TTFT p50/p95, rate, draft/accepted, tok/s, 429, breaker). make check exit 0.

## Ray sweep (id: sweep)
- `python -m draftlab.sweep` (Ray local, per-cell incremental parquet, resumable by (drafter,k,temp)) → results/sweep.parquet; target 0.5b-instruct, 488 val prompts (100/kind ≤128 tok), n_gen 16.
- Known: base@T0 k=1 accept exactly 1.0 (2-cell Ray test: all n_acc==n_draft); k≥2 expect ≥0.99 (T=k vs T=1 GEMM tiling ~1e-4). ModelDrafter bugs fixed: propose double-decoded last committed token; rollback(m) must advance `_last` by m. Regression tests red on old code.
- Cost: 6.4 tok/s @16 thr → 42-cell grid ≈ 20 CPU-h → batches (stop rule): B1 base/lora/dpo k=1 T=0 + none; then ngram/tiny k=1 ×{0,0.7}; then k=8.

## Judge (id: judge)
- `python -m draftlab.eval.judge`: gen {target, spec(k=4, lora_dpo), drafter_alone} at T=0.7 (49 test prompts, n_gen 16, incremental data/judge_outputs.json) → blind pairwise vs qwen-agent 27B ($OLLAMA_BASE_URL), both orders, cache data/judge_cache/, cap 150 → results/judge.json. Tests 7/7 (fake OpenAI server).
- Result, OUT of pre-written range (finding, not fix): spec_vs_target n=81 W/T/L 36/1/44 — win 0.444 vs loss 0.543, CIs overlap, binom p=0.434: no significant loss; "losses ≤25%" presumed tie dominance, but T=0.7 independent sampling gives 0/46 exact text matches, so the judge breaks near-ties ~50/50 — that split IS the lossless signature (distribution-level: test_spec.py chi-square T=0.7 p>0.01). spec_vs_drafter_alone n=24: win 0.625 vs loss 0.375, gap 25pp ≥15pp ✓ but underpowered (cap eaten by 24 invalid-JSON retries → 19 drafter jobs uncalled).
- Corpus gotcha: test.parquet block (contrib-loop-20260926-231240, 77976) appears 8× — code-split sub-blocks keep parent (run,seq) → 4 prompt entries share one output key (9 silent gen skips); no verdict contamination (all its jobs invalid/capped).
