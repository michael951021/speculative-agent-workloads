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
- `draftlab/jaxlm/spec.py`: Drafter protocol; NGramDrafter + ModelDrafter. One target
  call scores m drafts + bonus (draft j's dist = prev row (j=0) or scored[j-1]);
  accept u·q[t]<p[t], residual max(0,p−q); rollback: truncate to L+A, decode bonus.
- `SpecDecoder._steps` yields (emitted, StepStats); tests: T=0==greedy, T=1 chi2 p=0.391.

## Engine server (id: engine)
- `draftlab/serve/app.py`: FastAPI OpenAI-compatible (chat/completions SSE, /v1/models;
  usage adds draft/accepted; name `<base>[+<drafter>-k<K>]`).
  `python -m draftlab.serve --model K --alias A --port P` (default 0.5b-instruct, :8100).
  Gotcha: `apply_chat_template(tokenize=False)` else BatchEncoding; tiny → ToyTokenizer.

## Alpha metric (id: alpha, alphafix)
- `python -m draftlab.eval.alpha` (ngram|jax|hf); default target 1.5b-instruct;
  row i predicts token i+1 -> slice [len(ctx)-1 : len(ctx)+n-1]; empty ctx: pos 0 invalid.
- Known answers (real 0.5B): CE==HF labels= <1e-3; top-1 val
  42.4% (band 40-70%); uniform-q=1-TV; ngram repeated 63/64, shuffled null collapses.
- ngram (400 val blocks, 1.5B): alpha 0.3341, greedy 0.3681, greedy_corpus 0.3335
  (regression OK), tok/call@k8 1.50; tool_use:Edit 0.56 highest > tool_result 0.33 >
  code 0.26 > text 0.19.
- Old alpha 0.0096 / "1.2% vs 27B" = off-by-one bug (fixed).
- tiny.json (a20; a23 fresh 400-block re-run reproduces, worst |d| 3.67e-7, tpc@8
  analytic): alpha 0.0105, tpc@8 1.01; per-kind 0.004-0.015 vs ngram 0.155-0.56
  (ngram wins all kinds); 0.5B control 0.7675 → weak 43M drafter (KL 2.63), not metric.
- Registration (harnessreg): host runner registered 0 RESULT blocks ever (prior checks
  were agent self-checks; even bare-block runs skipped → total gap, not matching).
  Stop hook `.claude/hooks/register_result.py`: at stop parse last RESULT block; done:
  TODO line (text/id) + TASKLOG entry + commit, idempotent. Tests: 6, real transcripts.
  a2: 6/6 + ruff green, hook wired in .claude/settings.json; end-to-end stop audit lands
  in .agent-evidence/harnessreg-stops.log (verify next run).

## Distilled tiny drafter (JAX) (id: distill, 2026-09-30)
- `draftlab/drafters/distill.py`: cache (teacher top-64+tail → data/teacher/*.npz) +
  train (4-layer d=256 Qwen2, tied embed, KL(T‖S) top-64 + renormalised tail,
  AdamW warmup-cosine, orbax ckpt/tiny/N, HF export at ckpt/tiny/export).
- 1200 steps (default) @1e-3, batch 16×256: val KL 3.7879 → 2.6292 (30.6% ✅);
  curve results/distill_curve.json; teacher-init control 2.6379 (no gain); lr 3e-3 oscillated.
  Logs distill-train-{1200,tinit1200}.
- REPORT needs more distill steps/capacity (alpha 0.0105, see alpha section).
