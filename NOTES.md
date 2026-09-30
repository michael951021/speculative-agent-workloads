# draftlab — current state

## Toolchain (id: boot)
- `make bootstrap`: uv, Py3.12 .venv, torch CPU, `uv pip install -e ".[all]"`, .tools/; `make check` = ruff + pytest (others SKIP when dir absent); capture.py: no `VAR=x cmd` prefixes.

## Models (id: models)
- `models.py` `fetch` → models/{0.5b-instruct, 1.5b-instruct, 0.5b}; `info` asserts shared vocab 151936; `tiny_config()` = 2-layer d=64 vocab-512 Qwen2.

## Corpus (id: corpus)
- `python -m draftlab.data build` → data/corpus/{train,val,test}.parquet (code split, dedupe, trunc 4096; runs by sha1 26/3/3); 45257/667/566 blocks; `stats` → results/corpus_stats.json.

## N-gram baseline (id: ngram)
- `drafters/ngram.py`: NGramMatcher.proposals (prompt lookup n=1..4, first occurrence, no self-match) → results/ngram.json. Greedy accept @k=8: overall 0.329 (1.76 tok/call), tool_use 0.548 best, text 0.199 worst.

## JAX model (id: jaxmodel)
- `jaxlm/model.py`: pure-pytree Qwen2, bf16→f32 load, padded KV cache, one jitted step (prefill+decode). Parity 0.5B vs transformers: prefill 1.3e-4, 32 greedy identical.

## Spec decoding (id: spec)
- `jaxlm/spec.py`: Drafter protocol (NGram/Model); one target call scores m drafts + bonus; accept u·q[t]<p[t], residual max(0,p−q); tests T=0==greedy, T=1 chi2 p=0.391.

## Engine server (id: engine)
- `serve/app.py`: FastAPI OpenAI-compatible (SSE, /v1/models); `python -m draftlab.serve` (0.5b-instruct, :8100); `apply_chat_template(tokenize=False)`; tiny → ToyTokenizer.

## Alpha metric (id: alpha, alphafix)
- `python -m draftlab.eval.alpha` (ngram|jax|hf); default target 1.5b-instruct; row i predicts i+1; empty ctx: pos 0 invalid.
- Known answers: CE==HF labels <1e-3; top-1 val 42.4%; uniform-q=1-TV; ngram repeat 63/64, shuffle collapses.
- ngram (400 val): α 0.3341 (greedy 0.3681, corpus 0.3335), tpc@8 1.50; Edit 0.56 > tool_result 0.33 > code 0.26 > text 0.19.
- tiny (43M distilled): α 0.0105 (analytic 4e-9; ngram wins all kinds); 0.5B control 0.7675 → weak drafter, not metric.
- harnessreg: stop hook .claude/hooks/register_result.py parses last RESULT → TODO+TASKLOG+commit.

## Distill tiny drafter (JAX, id: distill)
- `drafters/distill.py`: teacher top-64+tail cache + 4L d=256 train (KL top-64+tail, AdamW; export ckpt/tiny/export). 1200 steps: val KL 3.79 → 2.63 (30.6% ✅); teacher-init control no gain. REPORT: needs more steps (α 0.0105).

## LoRA drafter (PyTorch, id: lora)
- `drafters/lora.py`: PEFT r=16/α32 attn+MLP (7 targets) SFT of 0.5B, packed train blocks; `--fsdp` FSDP2 ⇒ bit-identical; step_loss == HF labels <1e-5; 300 steps → ckpt/lora.
- α 0.7306 [0.7302, 0.7308] (pre-written [0.7, 0.95] ✅; control 0.7675 −0.037; ≫ ngram 0.3341); per-kind Edit 0.82 > Write/Read 0.80 > Bash 0.77 > tool_result 0.74 > code 0.72 > thinking ≈ text 0.66 (tool_use ≥ text ✅); tok/call@8 3.49; tests 5/5.

## DPO drafter (id: dpo)
- `drafters/dpo.py`: T=1 spec run (1.5b target, LoRA drafter; accept u·q<p, resample max(0,p−q)) → pairs cached data/pairs/dpo.jsonl; TRL DPOTrainer β=0.1, pre-tokenized (PreTokDPOTrainer). Gotchas: use_cpu, gradient_checkpointing=False, is_trainable=True.
- 2000 pairs (accept 0.786); margin −3.794 → −3.530; α 0.7306 → 0.7319 (CIs disjoint, small real gain; pre-written [0.70, 0.78] ✅). Tests 3/3, suite 70/70 (logs dpo-default-run3).

## Kernels (id: triton)
- `kernels/verify.py`: fused_verify (1 launch: online-softmax [k+1,V] target + [k,V] draft rows; accept p>u·q = spec.py u·q<p, 0/0 rejects → first reject fr + normalised residual max(0,p−q)@fr, q=0 for bonus); fused_alpha (per-row Σmin(softmax p,softmax q)); torch refs.
- Tests 7/7 (TRITON_INTERPRET=1, analytic q uniform→Σmin(p,1/V), q=p→α=1); bench.py skips CPU; alpha._alpha_sum: kernel on CUDA only. Gotchas: TRITON_INTERPRET read lazily; scalar `if` in _online_upd GPU-untested.

## Gateway (id: gateway)
- `gateway/` Go 1.26 stdlib+client_golang+yaml: Bearer auth (sha256 in config.yaml), token-bucket 429+Retry-After, model→backend routing, retries on connect err, breaker (5 fails→open→half-open probe), SSE tee+flush, usage→metrics (requests{key,model,status}, ttft/latency, tok/s, draft/accepted counters, backend_up, breaker_state), /v1/models merge, /healthz, /ui.
- `cd gateway && go vet && go test ./...` 8/8 PASS (fake usage 12/7→counters 24/14 exact). Live smoke vs python fake: 401/429/stream order/tok/s≈98 ✅. Attempt 2 (2026-09-30) re-verified: 8/8 + live 401/200/429+Retry-After/404/models/draft 24·accepted 14 exact (gateway-*.log).

## Observability (id: obs)
- `deploy/prometheus/`: prometheus.yml (scrape 127.0.0.1:8200, rules.yml) · rules.yml: 4 recordings (acceptance_ratio5m, ttft_p50/p95_seconds5m, requests_rate5m) + 2 alerts (LowDraftAcceptance ratio<0.4 for:10m; HighTTFTp95 >2s for:5m) · rules_test.yml pins hand-computed values.
- Gotchas (promtool 2.53.1): test windows are LEFT-CLOSED [T−5m,T] (6 samples at 1m grid); unaligned left gap extrapolates: ts=4m factor (240+60)/240/300 = 1.25/300; recording-rule expr tests need `__name__` in exp_labels.
- Pinned: healthy @10m ratio 0.5, p50 0.5, p95 0.95 (exact float64); low-acceptance @4m exactly 0.4 (not <0.4), @5m 0.32, @10m 1/30=0.03333333333333333, pending 5–15m, firing 15m; slow-ttft p95 4.8125 / p50 3.125 from ts=1m, firing 6m. `promtool test rules` SUCCESS, make check green.
- `deploy/grafana/draftlab.json`: 8 panels (acceptance, TTFT p50+p95, req rate, draft/accepted tok rate, ttft buckets, output tok/s, 429 rate, breaker state), datasource+model vars, only recording rules + gateway metrics. check.sh promtool section runs check config + check rules + test rules. Attempt 2 (2026-09-30) finished: added tok/s/429/breaker panels, make check exit 0 (log .agent-evidence/obs-make-check.log), RESULT registered.
- Gateway: `model` label added to draftlab_draft/accepted_tokens_total (test: {model="m1"} 24/14).
