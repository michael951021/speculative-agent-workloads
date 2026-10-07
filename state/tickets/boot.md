# boot — Bootstrap toolchain and `make check`

## Issue
No reproducible toolchain: every later task needs Python 3.12 venv (jax/torch/triton/ray), pinned
Go/helm/kubeconform/terraform/tflint/hadolint/promtool, and one `make check` that proves the repo is
green. Without it, agents build on divergent environments.

## Question
Does `bash tools/bootstrap.sh` (idempotent) set up everything, and does `make check` pass on the
skeleton, skipping sub-checks whose directory doesn't exist yet? Refutation: any sub-check that fails
or hard-errors (instead of skipping) on an absent directory, or a non-idempotent second bootstrap run
that re-downloads/reinstalls needlessly.

## Evidence
- Start: repo has only `tools/export_traces.py`, no venv, no `.tools/`, no Makefile.
- Repro/check: `make bootstrap && make check`; smoke: `.venv/bin/python -c "import jax, torch, triton, ray"`.
  Output captured to `.agent-evidence/boot-check.log`.
- Observed: uv 0.12.19 + CPython 3.12.14 already present; go.dev/github/hashicorp reachable (302/200).
  Guess: pypi + pytorch cpu index reachable from sandbox (verify on install).

## Done when
- `make bootstrap && make check` exits 0 on this skeleton (python check runs; go/web/helm/tf/prom
  checks skip with a message).
- Second `make bootstrap` run skips downloads (versions.txt hash marker) — idempotent.
- `.venv/bin/python -c "import jax, torch, triton, ray"` exits 0.
- `.github/workflows/ci.yml` mirrors the same steps.
- Regression: a trivial pytest smoke test exists so `pytest` doesn't exit 5.

## Stop when
- Scope: only toolchain + Makefile + CI; no engine/drafter/gateway code, no deploy files.
- Stop/split: if two attempts at a download or install fail without new evidence (e.g. network to
  pypi/pytorch blocked), record the blocker in NOTES.md and stop. Do not vendor wheels.

## Plan
1. `tools/versions.txt` pins; `tools/bootstrap.sh` (uv → 3.12 venv → torch cpu index → `-e .[all]`
   → pinned binaries into `.tools/`, hash-marked for idempotency).
2. `pyproject.toml` (extras: jax, torch, data, serve, viz, dev, all) + `speculative_agent_workloads/__init__.py` + smoke test.
3. `Makefile` (bootstrap/check/fmt) with `tools/check.sh` doing per-directory skip logic.
4. `.github/workflows/ci.yml`.
5. Run `make bootstrap && make check`, import smoke; commit.
