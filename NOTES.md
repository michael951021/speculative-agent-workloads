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
- `make check` exit 0 (5 passed; test_shared_vocab active once models present, skipif otherwise).
- Gotcha: HF download is unauthenticated (HF_TOKEN unset) — fine at this size; ~5 GB total.

## Next
- Corpus to Parquet (id: corpus): data/corpus.py over data/raw/agent_turns.jsonl.
