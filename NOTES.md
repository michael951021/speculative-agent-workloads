# draftlab — current state

## Toolchain (boot, done 2026-09-29)
- `make bootstrap` = tools/bootstrap.sh (idempotent): uv + Py3.12 .venv, torch CPU wheels first,
  `uv pip install -e ".[all]"`, pinned binaries per tools/versions.txt into .tools/ (go at
  .tools/go/bin). Re-download skipped when .tools/.versions matches.
- `make check` = tools/check.sh: ruff + pytest, then go/vitest/helm/terraform/hadolint/promtool
  sections print `SKIP <name> (<dir> absent)` until the dir exists. Verified green;
  jax 0.11.2, torch 2.14.0+cpu, triton 3.8.0, ray 2.58.0.
- Gotchas: helm tarball member `linux-amd64/helm`; tflint ships zips not tarballs;
  hadolint asset is `hadolint-Linux-x86_64` (amd64-only naming); terraform zip via
  `python3 -m zipfile`.

## Models (id: models, done 2026-09-29)
- `draftlab/models.py`: `fetch [--name K]` (snapshot_download, safetensors + tokenizer files) →
  `models/{qwen2.5-0.5b-instruct,qwen2.5-1.5b-instruct,qwen2.5-0.5b}` (~4.9 GB), idempotent.
  `info` asserts all three vocabularies equal (151936). `tiny_config()` = 2-layer d=64 Qwen2,
  vocab 512, for fast tests. HF download unauthenticated — fine at this size.

## Corpus (id: corpus, done 2026-09-29)
- `python -m draftlab.data build`: data/raw/agent_turns.jsonl → data/corpus/{train,val,test}.parquet,
  columns run/project/seq/role/kind/tool/text/n_tokens (Qwen 0.5b-instruct tokenizer).
  kind: fenced code split out of `text` blocks; tool = name from tool_use JSON.
  Drops <8 tok, dedupes exact (kind,text) repeats, truncates at 4096 tok.
  Splits = whole runs, ordered by sha1(run): 26/3/3 runs (81/9/9% of 32), disjoint.
  Real build: 45257/667/566 blocks, 24 s. Evidence: .agent-evidence/corpus-build.log.
- `python -m draftlab.data stats`: runs draftlab/data/stats.sql (DuckDB, `-- @query NAME` sections) over the
  3 parquets → results/corpus_stats.json (by_split, by_kind, by_tool, by_project,
  length_percentiles p50/p90/p99/max per kind). p50 tokens: tool_result 243, thinking 132,
  code 101, tool_use 85, text 29.
- Tests: draftlab/tests/test_corpus.py (stub WordTok, no model needed) — kinds, code split,
  dedupe, min/max token filter, split disjointness, assign_splits determinism. make check exit 0.
- Gotchas: capture.py cannot take `VAR=x cmd` (use `sh -c`) and refuses existing log paths
  (rm first). Unanchored `data/` in .gitignore shadowed the draftlab/data/ source package — now `/data/`.

## N-gram baseline (id: ngram, done 2026-09-29)
- `draftlab/drafters/ngram.py`: NGramMatcher.proposals(ctx) — vLLM-style prompt
  lookup, n=1..4 longest-first, first-occurrence continuation, no self-match (an
  n-gram is usable only once its continuation token is in context). Chain
  property: accepted proposals equal the real prefix, so per-position
  proposals precompute the whole k-draft chain (runs of accepts, capped at k).
  `python -m draftlab.drafters.ngram` → results/ngram.json (5 s, val: 667 blocks,
  236,985 pos, context = prev-8 same-run blocks + own prefix, bootstrap B=200
  resampling the 3 val runs, seed 0).
- Results (greedy accept per position; tokens/target call @k=8): tool_use
  0.548 (3.34), tool_result 0.309 (2.08), code 0.286 (1.89), thinking 0.246
  (1.73), text 0.199 (1.56); overall 0.329 (1.76). Tool traffic far more
  predictable than prose — floor confirmed. CIs wide (only 3 val runs).
- Tests: draftlab/tests/test_ngram.py — hand-made sequences (first occurrence,
  longest-n priority, no self-match, accept_runs, mean accepted length by k).
  make check exit 0. Evidence: .agent-evidence/ngram-run.log.
- Gotcha: longest-first lookup must `continue` past n>t (break skips valid
  shorter n's); bootstrap resampling 3 runs makes per-kind CIs coarse.

## JAX model (id: jaxmodel, done 2026-09-29)
- `draftlab/jaxlm/model.py`: pure-function-pytree Qwen2 (RMSNorm, RoPE, GQA, SwiGLU,
  QKV bias; tied or separate lm_head). `load_safetensors` reads HF checkpoints
  (bf16 upcast to f32 first — torch bf16 has no `.numpy()` path). Preallocated
  padded KV cache; one jitted `step` covers prefill (fresh cache) and multi-token
  decode: dynamic-slice cache writes + causal/valid mask, no dynamic shapes.
- Parity (0.5B f32 vs transformers f32, 3 prompts): prefill max abs diff 1.3e-4
  (< 2e-3), argmax identical on all rows, greedy 32 tokens identical (torch
  generate needs `eos_token_id=-1`: EOS 151643 lands within 32 steps). Tiny:
  multi-token decode == single-token chain == prefill (1e-7); prefill 64 + 32
  decode wall < 5 s incl. first jit. Tests: draftlab/tests/test_jaxmodel.py (4).
  `make check` green (21 tests). Evidence: .agent-evidence/jaxmodel.log.
- Gotcha (cost a long debug): a 1.32 full-prefill mismatch on the tiny config was
  probe-side, not model — `tiny_config()` leaves `tie_word_embeddings` at the
  Qwen2Config default (False), so JAX params carries a separate random `lm_head`;
  the probe's torch state_dict had tied it to the embedding. Layers L0/L1 matched
  torch at ~2e-8 all along.

## Next
- Speculative sampling (id: spec): verify loop over `jaxlm.step` logits + ngram
  Drafter (both done). Greedy acceptance per kind, then speedup with draft cost ratio.
