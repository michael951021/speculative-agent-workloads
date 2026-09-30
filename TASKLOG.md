# Task log

<!-- One entry per completed TODO task, appended by the agent. Never edited or condensed — for
     a rolling handoff of current state see NOTES.md instead. -->

## boot — Bootstrap toolchain and make check
- Goal: one-command reproducible toolchain (uv, Python 3.12 venv with jax/torch-cpu/triton/ray extras, pinned Go/helm/kubeconform/terraform/tflint/hadolint/promtool) plus `make check` with per-directory skip logic and a CI mirror.
- Approach: tools/versions.txt pins; tools/bootstrap.sh (idempotent, hash-marked binary installs); pyproject extras (jax/torch/data/serve/viz/dev/all); Makefile bootstrap/check/fmt backed by tools/check.sh; .github/workflows/ci.yml; ruff-cleaned tools/export_traces.py to make the skeleton green.
- Blockers: two download-URL bugs fixed on the fly (helm tarball member path, tflint publishes zips not tarballs).
## models — Model downloads
- Goal: shared Qwen2.5 checkpoints (0.5B-Instruct, 1.5B-Instruct, 0.5B) with one shared tokenizer, plus a tiny random Qwen2 config for fast tests.
- Approach: draftlab/models.py with fetch (snapshot_download, safetensors+tokenizer+config allow_patterns, idempotent), info (params via safetensors headers, dtype, vocab, assert equal get_vocab), tiny_config (2 layers/d64/vocab512); tests in draftlab/tests/test_models.py (tiny_config fields, shared-vocab skipif-unfetched); fetched all three into models/ (~4.9 GB).
- Blockers: none.
## corpus — Corpus to Parquet
- Goal: turn data/raw/agent_turns.jsonl into kind-labeled, run-split Parquet (train/val/test) + DuckDB stats JSON, so every later drafter/metric trains on a leak-free, tokenized corpus.
- Approach: draftlab/data/corpus.py (fenced code split out of text, tool name from tool_use JSON, Qwen tokenizer n_tokens, drop <8 tok, dedupe exact (kind,text), truncate 4096, sha1(run)-ordered 80/10/10 whole-run split) + data/stats.sql (named DuckDB queries) + `python -m draftlab.data {build,stats}` CLI; fixture tests in draftlab/tests/test_corpus.py with a stub word tokenizer; real build 45257/667/566 blocks (24 s), stats → results/corpus_stats.json.
- Blockers: none (minor: capture.py rejects `VAR=x cmd` prefixes and existing log paths; duckdb 1.5 con.execute returns a connection, not a relation).
## ngram — N-gram lookup baseline
- Goal: model-less prompt-lookup drafter (vLLM ngram style) as the floor every trained drafter must beat: per-kind greedy acceptance and mean accepted length for k in {1,2,4,8} on the val split, with bootstrap 95% CIs.
- Approach: draftlab/drafters/ngram.py — NGramMatcher.proposals (match last n=1..4 longest-first against known context, first-occurrence continuation, no self-match) + accept-runs chain (accepted proposals equal the real prefix, so one per-position pass precomputes all k chains); evaluate() tokenizes val blocks (Qwen 0.5b tokenizer), context = prev-8 same-run blocks + own prefix, per-kind/per-k aggregates, bootstrap B=200 resampling the 3 val runs; results/ngram.json (5 s run). Results: greedy accept tool_use 0.548 > tool_result 0.309 > code 0.286 > thinking 0.246 > text 0.199; tokens/call @k=8 from 3.34 (tool_use) to 1.56 (text). Tests: draftlab/tests/test_ngram.py hand-made sequences; make check green.
- Blockers: none (fixed two matcher bugs found by hand-made tests: lookup loop must `continue` past n>t; accept_runs must handle the last element; CIs are coarse because val has only 3 runs).
## jaxmodel — Qwen2 in JAX
- Goal: from-scratch Qwen2 in JAX (RMSNorm, RoPE@theta, GQA, SwiGLU, QKV bias, tied embeddings), HF safetensors load, preallocated KV cache with jitted prefill/decode steps; f32 parity vs transformers (atol 2e-3, 3 prompts), 32 greedy tokens identical, tiny_config < 5 s.
- Approach: attempt 1 (prior run) implemented draftlab/jaxlm/model.py (pure-pytree, one jitted `step` covering prefill + multi-token decode via dynamic-slice cache writes) + test_jaxmodel.py, ticket state/tickets/jaxmodel.md; committed fb3d3cf but the run ended before the TODO checkbox/TASKLOG. Attempt 2 (this run) re-verified from the commit: pytest test_jaxmodel.py 4/4 in 36 s (evidence .agent-evidence/jaxmodel-verify.log) and `make check` green (21 tests, ruff; other sections SKIP, .agent-evidence/jaxmodel-check.log), then checked the TODO line, condensed NOTES.md, logged here.
- Blockers: none (attempt 1's debug note: a 1.32 tiny-config prefill mismatch was probe-side — tiny_config leaves tie_word_embeddings=False, so JAX built a separate random lm_head while the torch probe tied it to embed; also torch bf16 has no .numpy() path, loader upcasts to f32).
