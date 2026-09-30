# draftlab — current state

## Toolchain (id: boot)
- `make bootstrap` (idempotent): uv + Py3.12 .venv, torch CPU first, `uv pip install -e
  ".[all]"`, pinned binaries into .tools/ (go at .tools/go/bin). `make check` = ruff +
  pytest; others SKIP when their dir is absent.
- Gotchas: capture.py rejects `VAR=x cmd` (use `sh -c`) and existing log paths (rm first).

## Models (id: models)
- `draftlab/models.py`: `fetch [--name K]` → `models/{0.5b-instruct, 1.5b-instruct, 0.5b}`
  (~4.9 GB), idempotent; `info` asserts shared vocab 151936. `tiny_config()` = 2-layer
  d=64 vocab-512 Qwen2 for fast tests.

## Corpus (id: corpus)
- `python -m draftlab.data build`: raw jsonl → data/corpus/{train,val,test}.parquet
  (fenced code split; drop <8 tok, dedupe, truncate 4096; runs by sha1: 26/3/3).
  45257/667/566 blocks. `stats` → results/corpus_stats.json. p50 tok: tool_result 243,
  thinking 132, code 101, tool_use 85, text 29.

## N-gram baseline (id: ngram)
- `draftlab/drafters/ngram.py`: NGramMatcher.proposals — prompt lookup n=1..4, first
  occurrence, no self-match. `python -m draftlab.drafters.ngram` → results/ngram.json.
  Greedy accept / tok per call @k=8: tool_use 0.548 (3.34), tool_result 0.309 (2.08),
  code 0.286 (1.89), thinking 0.246 (1.73), text 0.199 (1.56); overall 0.329 (1.76).

## JAX model (id: jaxmodel)
- `draftlab/jaxlm/model.py`: pure-pytree Qwen2; `load_safetensors` upcasts bf16→f32;
  padded KV cache; one jitted step does prefill + multi-token decode.
- Parity 0.5B f32 vs transformers: prefill diff 1.3e-4, argmax + greedy 32 identical
  (torch gen: `eos_token_id=-1`). tiny_config: `tie_word_embeddings=False` — tie lm_head in probes.

## Spec decoding (id: spec)
- `draftlab/jaxlm/spec.py`: Drafter protocol; NGramDrafter + ModelDrafter. One target
  call scores m drafts + bonus (draft j's dist is prev row (j=0) or scored[j-1]);
  accept u·q[t]<p[t], residual max(0,p−q); rollback truncates to L+A then decodes bonus.
- `SpecDecoder._steps` yields (emitted, StepStats); tests: T=0==greedy, T=1 chi2 p=0.391.

## Engine server (id: engine)
- `draftlab/serve/app.py`: FastAPI OpenAI-compatible (chat/completions SSE, /v1/models;
  usage adds draft/accepted; name `<base>[+<drafter>-k<K>]`).
  `python -m draftlab.serve --model K --alias A --port P` (default 0.5b-instruct, :8100).
  Gotcha: `apply_chat_template(tokenize=False)` else BatchEncoding; tiny → ToyTokenizer.

## Alpha metric (id: alpha)
- `python -m draftlab.eval.alpha` (ngram|jax|hf adapters `dists(ctx, ids)->(q, valid)`).
  Teacher-forced vs 0.5B target: alpha=Σmin(p,q) + greedy + greedy_corpus;
  ctx = same-run earlier blocks ≤1024 tok; bootstrap CI; c measured → results/alpha/<drafter>.json
- ngram (400 val blocks): alpha 0.0096, greedy 0.0118, greedy_corpus 0.3335 ≈ baseline
  0.3286; tok/call@k8 = 1.01, speedup 1.005 (c = 0.0005).
- Finding: 0.5B argmax matches 27B corpus ~1.2% — bottleneck is target agreement, not
  drafter coverage (trained drafters are the fix).

## Distilled tiny drafter (JAX) (id: distill, 2026-09-30)
- `draftlab/drafters/distill.py`: cache (teacher top-64+tail → data/teacher/*.npz,
  1M train tok built 2592 s) + train (4-layer d=256 Qwen2, tied 151936 embed,
  KL(T‖S) top-64 + renormalised tail, AdamW warmup-cosine, orbax ckpt/tiny/N,
  HF export at ckpt/tiny/export).
- 1200 steps (default) @1e-3, batch 16×256: val KL 3.7879 → 2.6292 (30.6% ✅);
  curve results/distill_curve.json. Teacher-init control (--teacher-init): 2.6379, no gain.
  800 steps 29.6%; lr 3e-3 oscillated. Logs .agent-evidence/distill-train-{1200,tinit1200}.log.
- alpha (results/alpha/tiny.json, 400 val blocks w/ ctx): overall α=0.0142,
  greedy=0.1465 (ngram: 0.0096/0.0118). Student ≈ uniform (H(q)=11.79≈ln V, peak
  3.8e-4 vs teacher 0.36) — weak approx (2.63 nats ≈ 3.8 bits); self-check α=1.0000 →
  metric correct; gap is real (0.5B↔27B mismatch + approx error). Pre-run range 0.4–0.9
  mis-anchored on Leviathan (strong-approx drafters).
