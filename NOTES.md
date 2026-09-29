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

## Next
- Models download (id: models): draftlab/models.py fetch/info, needs network to huggingface.
