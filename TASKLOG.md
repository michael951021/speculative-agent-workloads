# Task log

<!-- One entry per completed TODO task, appended by the agent. Never edited or condensed — for
     a rolling handoff of current state see NOTES.md instead. -->

## boot — Bootstrap toolchain and make check
- Goal: one-command reproducible toolchain (uv, Python 3.12 venv with jax/torch-cpu/triton/ray extras, pinned Go/helm/kubeconform/terraform/tflint/hadolint/promtool) plus `make check` with per-directory skip logic and a CI mirror.
- Approach: tools/versions.txt pins; tools/bootstrap.sh (idempotent, hash-marked binary installs); pyproject extras (jax/torch/data/serve/viz/dev/all); Makefile bootstrap/check/fmt backed by tools/check.sh; .github/workflows/ci.yml; ruff-cleaned tools/export_traces.py to make the skeleton green.
- Blockers: two download-URL bugs fixed on the fly (helm tarball member path, tflint publishes zips not tarballs).
