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
