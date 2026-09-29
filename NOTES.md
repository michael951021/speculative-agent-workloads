# draftlab — current state

## Toolchain (boot, done 2026-09-29)
- `make bootstrap` = `bash tools/bootstrap.sh` (idempotent): uv (host or standalone → .tools/bin),
  `uv venv -p 3.12 .venv`, torch from pytorch CPU index first, then `uv pip install -e ".[all]"`,
  pinned binaries from `tools/versions.txt` into `.tools/` (go at .tools/go/bin, rest in .tools/bin).
  Binary re-download skipped when `.tools/.versions` matches versions.txt sha256.
- `make check` = `tools/check.sh`: ruff + pytest (run if draftlab/ exists), go/vitest/helm/terraform/
  hadolint/promtool sections each print `SKIP <name> (<dir> absent)` until the dir exists.
- Verified: `make bootstrap && make check` exit 0; 2nd bootstrap run re-downloads nothing;
  `.venv/bin/python -c "import jax, torch, triton, ray"` → jax 0.11.2, torch 2.14.0+cpu, triton 3.8.0,
  ray 2.58.0. Evidence: .agent-evidence/boot-accept.log, boot-check2.log.
- Gotchas: helm tarball member is `linux-amd64/helm`; tflint ships zips not tarballs; hadolint asset is
  `hadolint-Linux-x86_64` (amd64-only naming); terraform zip extracted via `python3 -m zipfile`.
- pyproject extras: jax, torch (incl. triton), data, serve, viz, dev (pytest, ruff, pyyaml), all (adds
  ray[default]). Smoke test draftlab/tests/test_smoke.py guards heavy imports.
- tools/export_traces.py was ruff-cleaned (behavior unchanged) to get `make check` green on skeleton.

## Models (id: models, done 2026-09-29)
- `draftlab/models.py`: `fetch [--name K]` (snapshot_download, allow_patterns = *.safetensors +
  tokenizer*/vocab*/merges*/config.json) → `models/{qwen2.5-0.5b-instruct,qwen2.5-1.5b-instruct,
  qwen2.5-0.5b}` (~4.9 GB total); idempotent (2nd fetch 0.6 s). `info` prints params via
  safetensors headers + dtype + vocab, asserts all three `get_vocab()` equal. `tiny_config()` =
  Qwen2Config(2 layers, d=64, inter=128, heads 4/2, head_dim 16, vocab 512) for fast tests.
- Verified: `python -m draftlab.models info` exit 0 — 0.494B/1.544B/0.494B params, BF16,
  vocab=151936 (tokenizer n=151665), shared-vocab assert passes.
  Evidence: .agent-evidence/models-{fetch,fetch2,info,check}.log.
- `make check` exit 0; test_shared_vocab active once models present, skipif otherwise.
- Gotcha: HF download is unauthenticated (HF_TOKEN unset) — fine at this size; ~5 GB total.

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
- Gotchas: capture.py cannot take `VAR=x cmd` (use `sh -c`) and refuses existing log paths (rm
  first). Unanchored `data/` in .gitignore shadowed the draftlab/data/ source package — now `/data/`.

## Next
- N-gram lookup baseline (id: ngram, after corpus) over data/corpus splits.
