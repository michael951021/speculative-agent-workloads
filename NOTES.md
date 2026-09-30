# draftlab — current state

## Toolchain (id: boot, done 2026-09-29)
- `make bootstrap` = tools/bootstrap.sh (idempotent): uv + Py3.12 .venv, torch CPU first,
  `uv pip install -e ".[all]"`, pinned binaries per tools/versions.txt into .tools/ (go at
  .tools/go/bin). `make check` = tools/check.sh: ruff + pytest, then go/vitest/helm/terraform/
  hadolint/promtool sections `SKIP <name> (<dir> absent)` until present. Verified green;
  jax 0.11.2, torch 2.14.0+cpu, triton 3.8.0, ray 2.58.0.
- Gotchas: helm tarball member `linux-amd64/helm`; tflint ships zips; hadolint asset
  `hadolint-Linux-x86_64`; terraform zip via `python3 -m zipfile`. capture.py rejects
  `VAR=x cmd` (use `sh -c`) and existing log paths (rm first).

## Models (id: models, done 2026-09-29)
- `draftlab/models.py`: `fetch [--name K]` (snapshot_download) → `models/{qwen2.5-0.5b-instruct,
  qwen2.5-1.5b-instruct,qwen2.5-0.5b}` (~4.9 GB), idempotent. `info` asserts shared vocab 151936.
  `tiny_config()` = 2-layer d=64 vocab-512 Qwen2 for fast tests.

## Corpus (id: corpus, done 2026-09-29)
- `python -m draftlab.data build`: data/raw/agent_turns.jsonl → data/corpus/{train,val,test}.parquet
  (run/project/seq/role/kind/tool/text/n_tokens; fenced code split out of text; drop <8 tok,
  dedupe, truncate 4096; whole runs by sha1(run): 26/3/3 runs). 45257/667/566 blocks, 24 s.
- `python -m draftlab.data stats`: stats.sql (DuckDB) over the 3 parquets →
  results/corpus_stats.json (by split/kind/tool/project, length percentiles). p50 tokens:
  tool_result 243, thinking 132, code 101, tool_use 85, text 29.
- Tests: test_corpus.py (stub WordTok). Gotcha: unanchored `data/` in .gitignore shadowed the
  package — now `/data/`.

## N-gram baseline (id: ngram, done 2026-09-29)
- `draftlab/drafters/ngram.py`: NGramMatcher.proposals(ctx) — vLLM prompt lookup n=1..4
  longest-first, first-occurrence continuation, no self-match; accepted proposals = real
  prefix, so per-position pass precomputes the k-draft chain. `python -m draftlab.drafters.ngram`
  → results/ngram.json (5 s; val 667 blocks, context = prev-8 same-run blocks + prefix,
  bootstrap B=200 over 3 val runs, seed 0).
- Greedy accept / tokens per call @k=8: tool_use 0.548 (3.34), tool_result 0.309 (2.08),
  code 0.286 (1.89), thinking 0.246 (1.73), text 0.199 (1.56); overall 0.329 (1.76).
  CIs coarse (3 val runs). Gotcha: lookup must `continue` past n>t, not break.

## JAX model (id: jaxmodel, done 2026-09-29)
- `draftlab/jaxlm/model.py`: pure-pytree Qwen2 (RMSNorm, RoPE, GQA, SwiGLU, QKV bias,
  tied-or-separate lm_head). `load_safetensors` upcasts bf16→f32 before numpy (no bf16 dtype
  in numpy). Preallocated padded KV cache; one jitted `step` does prefill (fresh cache) and
  multi-token decode via dynamic-slice writes, no dynamic shapes.
- Parity (0.5B f32 vs transformers f32, 3 prompts): prefill max abs diff 1.3e-4 (< 2e-3),
  argmax identical all rows, greedy 32 tokens identical (torch generate needs
  `eos_token_id=-1`: EOS lands within 32 steps). Tiny: multi-token == chained single-token ==
  prefill (1e-7); prefill 64 + 32 decode < 5 s incl. first jit. Tests: test_jaxmodel.py (4);
  `make check` green (21). Evidence: .agent-evidence/jaxmodel.log, jaxmodel-verify.log,
  jaxmodel-check.log.
- Debug gotcha: a 1.32 tiny-config prefill mismatch was probe-side — `tiny_config()` leaves
  `tie_word_embeddings=False` (HF default), so JAX had a random separate lm_head while the
  torch probe tied it to embed.

## Next
- Speculative sampling (id: spec): verify loop over `jaxlm.step` logits + ngram Drafter
  (both done). Then greedy acceptance per kind → speedup with draft cost ratio.
