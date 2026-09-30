# draftlab — current state

## Toolchain (id: boot)
- `make bootstrap` (idempotent): uv + Py3.12 .venv, torch CPU first, `uv pip install -e ".[all]"`,
  pinned .tools/ binaries; `make check` = ruff + pytest; others SKIP when dir absent.
- Gotcha: capture.py rejects `VAR=x cmd` prefixes and existing log paths.

## Models (id: models)
- `draftlab/models.py`: `fetch [--name K]` → `models/{0.5b-instruct, 1.5b-instruct, 0.5b}`
  (~4.9 GB), idempotent; `info` asserts shared vocab 151936; `tiny_config()` = 2-layer
  d=64 vocab-512 Qwen2.

## Corpus (id: corpus)
- `python -m draftlab.data build`: raw jsonl → data/corpus/{train,val,test}.parquet
  (fenced code split; drop <8 tok, dedupe, truncate 4096; runs by sha1: 26/3/3).
  45257/667/566 blocks; `stats` → results/corpus_stats.json.

## N-gram baseline (id: ngram)
- `draftlab/drafters/ngram.py`: NGramMatcher.proposals — prompt lookup n=1..4, first
  occurrence, no self-match. `python -m draftlab.drafters.ngram` → results/ngram.json.
  Greedy accept / tok per call @k=8: tool_use 0.548 (3.34), tool_result 0.309 (2.08),
  code 0.286 (1.89), thinking 0.246 (1.73), text 0.199 (1.56); overall 0.329 (1.76).

## JAX model (id: jaxmodel)
- `draftlab/jaxlm/model.py`: pure-pytree Qwen2; `load_safetensors` upcasts bf16→f32;
  padded KV cache; one jitted step does prefill + multi-token decode.
- Parity 0.5B f32 vs transformers: prefill diff 1.3e-4, 32 greedy identical. tiny_config
  `tie_word_embeddings=False` — tie lm_head in probes.

## Spec decoding (id: spec)
- `draftlab/jaxlm/spec.py`: Drafter protocol (NGram/Model). One target call scores m
  drafts + bonus (draft j's dist = prev row (j=0) or scored[j-1]); accept u·q[t]<p[t],
  residual max(0,p−q); rollback truncates to L+A, decodes bonus. `_steps` yields
  (emitted, StepStats); tests T=0==greedy, T=1 chi2 p=0.391.

## Engine server (id: engine)
- `draftlab/serve/app.py`: FastAPI OpenAI-compatible (SSE, /v1/models; usage adds
  draft/accepted; name `<base>[+<drafter>-k<K>]`); `python -m draftlab.serve` (default
  0.5b-instruct, :8100). Gotcha: `apply_chat_template(tokenize=False)` else BatchEncoding;
  tiny → ToyTokenizer.

## Alpha metric (id: alpha, alphafix)
- `python -m draftlab.eval.alpha` (ngram|jax|hf); default target 1.5b-instruct;
  row i predicts token i+1 -> slice [len(ctx)-1 : len(ctx)+n-1]; empty ctx: pos 0 invalid.
- Known answers (real 0.5B): CE==HF labels= <1e-3; top-1 val
  42.4% (band 40-70%); uniform-q=1-TV; ngram repeated 63/64, shuffled null collapses.
- ngram (400 val): alpha 0.3341 (greedy 0.3681, corpus 0.3335, regression OK),
  tok/call@k8 1.50; Edit 0.56 > tool_result 0.33 > code 0.26 > text 0.19.
- Old alpha 0.0096 = off-by-one bug (fixed).
- tiny.json (a20 re-run, worst |d| 3.67e-7): alpha 0.0105, tpc@8 1.01 (analytic to
  4e-9); per-kind 0.004-0.015 vs ngram 0.155-0.56 (ngram wins all); 0.5B control
  0.7675 → weak 43M drafter (KL 2.63), not metric; a24: 11/11 + check_tiny_alpha PASS.
- Registration (harnessreg): host runner registered 0 RESULT blocks ever (total gap). Stop
  hook `.claude/hooks/register_result.py` (wired in .claude/settings.json): parse last RESULT
  block; done → TODO line + TASKLOG + commit; poll ≤20s (HARNESSREG_WAIT/SETTLE); 7 tests;
  verified 2026-09-30: stops 07:48–07:53 registered (line 13 [x]; audit .agent-evidence/harnessreg-stops.log).

## Distilled tiny drafter (JAX) (id: distill, 2026-09-30)
- `draftlab/drafters/distill.py`: cache (teacher top-64+tail → data/teacher/*.npz) +
  train (4-layer d=256, tied embed, KL(T‖S) top-64+renorm tail, AdamW warmup-cosine,
  HF export ckpt/tiny/export). 1200 steps @1e-3, batch 16×256: val KL 3.7879 → 2.6292
  (30.6% ✅); teacher-init control 2.6379 (no gain); lr 3e-3 oscillated.
- REPORT needs more distill steps/capacity (alpha 0.0105, see alpha section).
- a3: all Done-when re-verified (30.59% ✅, 5/5, ruff); a1/a2 failed only host-side
  registration (harnessreg a3 stop-hook, see alpha section).

## LoRA drafter (PyTorch, id: lora, 2026-09-30)
- `draftlab/drafters/lora.py`: PEFT LoRA r=16/α32, attn+MLP (7 targets), f32, packed
  train blocks; `--fsdp` = FSDP2 `fully_shard` (FSDP1 dropped CPU in torch 2.14), all
  ranks same chunks ⇒ bit-identical adapters (`rank{i}_lora.safetensors` + rank-0 dir);
  `--qlora` GPU-only guard. step_loss = HF alignment (== `model(ids, labels=ids).loss`
  <1e-5 test).
- Anchor check caught a step_loss off-by-one (token-echoer: α 0.0078 vs control 0.7675)
  — fixed, retrained. 300 steps ≈ 585 s → ckpt/lora; alpha (400 val) →
  results/alpha/lora.json: overall α 0.7306 [0.7302, 0.7308] — pre-written [0.7, 0.95]
  ✅, control 0.7675 −0.037 (155k-tok budget ≈ no gain over base yet), ≫ ngram 0.3341;
  per-kind Edit 0.82 > Write 0.80 > Read 0.80 > Bash 0.77 > tool_result 0.74 > code 0.72
  > thinking 0.66 ≈ text 0.66 (tool_use ≥ text ✅); tok/call@k8 3.49, speedup 2.12.
  Tests 5/5, suite 67/67 (logs lora-train2 / lora-alpha-eval2).
