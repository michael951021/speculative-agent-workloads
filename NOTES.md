# draftlab — current state

## Toolchain (id: boot)
- `make bootstrap` (idempotent): uv, Py3.12 .venv, torch CPU, `uv pip install -e ".[all]"`, .tools/
  binaries; `make check` = ruff + pytest (others SKIP when dir absent); capture.py: no `VAR=x cmd`
  prefixes, no existing log paths.

## Models (id: models)
- `models.py` `fetch` → models/{0.5b-instruct, 1.5b-instruct, 0.5b}; `info` asserts shared
  vocab 151936; `tiny_config()` = 2-layer d=64 vocab-512 Qwen2.

## Corpus (id: corpus)
- `python -m draftlab.data build` → data/corpus/{train,val,test}.parquet (code split, dedupe,
  trunc 4096; runs by sha1 26/3/3); 45257/667/566 blocks; `stats` → results/corpus_stats.json.

## N-gram baseline (id: ngram)
- `drafters/ngram.py`: NGramMatcher.proposals (prompt lookup n=1..4, first occurrence, no self-match)
  → results/ngram.json. Greedy accept @k=8: overall 0.329 (1.76 tok/call), tool_use 0.548 best,
  text 0.199 worst (per-kind: alpha section).

## JAX model (id: jaxmodel)
- `jaxlm/model.py`: pure-pytree Qwen2, bf16→f32 load, padded KV cache, one jitted step (prefill +
  decode). Parity 0.5B vs transformers: prefill 1.3e-4, 32 greedy identical.

## Spec decoding (id: spec)
- `jaxlm/spec.py`: Drafter protocol (NGram/Model); one target call scores m drafts + bonus (draft j's
  dist = prev row j=0, else scored[j-1]); accept u·q[t]<p[t], residual max(0,p−q);
  `_steps` yields (emitted, StepStats); tests T=0==greedy, T=1 chi2 p=0.391.

## Engine server (id: engine)
- `serve/app.py`: FastAPI OpenAI-compatible (SSE, /v1/models); `python -m draftlab.serve`
  (0.5b-instruct, :8100); `apply_chat_template(tokenize=False)`; tiny → ToyTokenizer.

## Alpha metric (id: alpha, alphafix)
- `python -m draftlab.eval.alpha` (ngram|jax|hf); default target 1.5b-instruct; row i predicts i+1 →
  slice [len(ctx)-1 : len(ctx)+n-1]; empty ctx: pos 0 invalid.
- Known answers: CE==HF labels <1e-3; top-1 val 42.4%; uniform-q=1-TV; ngram repeat
  63/64, shuffle collapses.
- ngram (400 val): α 0.3341 (greedy 0.3681, corpus 0.3335), tpc@8 1.50; Edit 0.56 > tool_result 0.33
  > code 0.26 > text 0.19.
- tiny (43M distilled): α 0.0105 (analytic 4e-9; ngram wins all kinds); 0.5B control 0.7675 → weak
  drafter, not metric.
- harnessreg: host runner never registered RESULT blocks; stop hook .claude/hooks/register_result.py
  parses last RESULT → TODO+TASKLOG+commit (verified 2026-09-30).

## Distilled tiny drafter (JAX) (id: distill, 2026-09-30)
- `drafters/distill.py`: teacher top-64+tail cache (data/teacher) + 4-layer d=256 train (tied embed,
  KL top-64+tail, AdamW; export ckpt/tiny/export). 1200 steps: val KL 3.79 → 2.63 (30.6% ✅);
  teacher-init control no gain. REPORT: needs more steps/capacity (α 0.0105).

## LoRA drafter (PyTorch, id: lora, 2026-09-30)
- `drafters/lora.py`: PEFT r=16/α32 attn+MLP (7 targets) SFT of 0.5B, packed train blocks; `--fsdp`
  FSDP2 fully_shard ⇒ bit-identical; step_loss == HF labels <1e-5 (anchor caught off-by-one:
  echoer α 0.0078); 300 steps → ckpt/lora.
- α 0.7306 [0.7302, 0.7308] (pre-written [0.7, 0.95] ✅; control 0.7675 −0.037; ≫ ngram 0.3341);
  per-kind Edit 0.82 > Write/Read 0.80 > Bash 0.77 > tool_result 0.74 > code 0.72 >
  thinking ≈ text 0.66 (tool_use ≥ text ✅); tok/call@8 3.49; tests 5/5
  (logs lora-train2 / lora-alpha-eval2).

## DPO drafter (id: dpo, 2026-09-30)
- `drafters/dpo.py`: T=1 spec run (1.5b target scores LoRA drafter; accept
  u·q[d]<p[d], resample max(0,p−q)) → pairs (ctx, chosen=accepted/resampled, rejected=draft; no-op
  resamples dropped) cached data/pairs/dpo.jsonl; TRL DPOTrainer β=0.1, ref=adapter disabled,
  prompt=last 48 ctx, completion 1 tok, pre-tokenized rows (PreTokDPOTrainer). Gotchas: use_cpu,
  gradient_checkpointing=False (breaks here), is_trainable=True, bf16 default.
- 2000 pairs (accept 0.786); train-pair margin −3.794 → −3.530 (500 sub ✅ up); α 0.7306 → 0.7319
  (CIs disjoint → small real gain; pre-written [0.70, 0.78] ✅); per-kind unchanged (tool_use ≥ text; WebFetch n=35). Tests 3/3, suite 70/70 (logs dpo-default-run3, dpo-alpha-eval).
