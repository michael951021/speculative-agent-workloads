# dl-cifix — tests skip without data

## Issue
A clean clone (no gitignored `data/`, `models/`, `ckpt/`) is what GitHub CI runs. Four tests
crash there instead of skipping, so the published repo can't be green:
`test_export_ui.py::test_replays_near_alpha_reference`, `::test_replays_match_alpha_pipeline`,
`test_sweep.py::test_two_cell_ray_sweep`, `::test_prompt_level_resume_and_sidecar_recovery`.

## Question
Do the 4 tests skip (not error) when their input files are missing, and still pass in /work?
Refutation: any of the 4 FAILs (vs skip) in a fresh clone, or any of the 4 fails in /work.

## Evidence
Reproduced in `/tmp/cc` (fresh clone): 4 failed, 6 passed. Causes:
- 2 sweep tests: `FileNotFoundError data/corpus/heldout.parquet` (load_prompts).
- `test_replays_match_alpha_pipeline`: tokenizer on missing `models/qwen2.5-1.5b-instruct`
  (HFValidationError), then would hit missing corpus/ckpt.
- `test_replays_near_alpha_reference`: `KeyError: 'code'` — committed `results/alpha/dpo.json`
  (heldout, pre-code-split) has no `code` row while committed `web/public/data/replays.json` has
  5 code blocks. Fails in /work too, not just CI. Existing guard style: `test_alpha_known.py`
  module-level `skipif(not HAS_MODEL)`.
- Once the guards landed, `test_prompt_level_resume_and_sidecar_recovery` exposed a PRE-EXISTING
  /work failure: its sidecar came from `load_prompts("val", 2, 128)` but `sweep.main` loads the
  heldout split (default since 5f98414) with max_tokens 64; `i` is the prompt-list position, so
  the val i-set (0..15) no longer covered heldout's (0..20) → cells reran. CI had masked it with
  the FileNotFoundError. Fixed by pinning split='heldout' in `_args` and building the sidecar
  from `load_prompts("heldout", 2, 64)`.

## Done when
`rm -rf /tmp/cc && git clone -q . /tmp/cc && cd /tmp/cc && /work/.venv/bin/python -m pytest -q
draftlab/tests/test_export_ui.py draftlab/tests/test_sweep.py -p no:cacheprovider` → 0 failed
(with skips), and the same 4 tests pass in /work. No measured numbers, so no known-answer run
needed; regression coverage = the skipif guards themselves (they must not skip in /work, proven by
the /work run passing).

## Stop when
Scope: only these 4 tests' guards + the one KeyError. Do not regenerate results/alpha/dpo.json or
replays (that is the Corrections re-measure section). Stop after 2 attempts without new evidence.

## Plan
1. test_sweep.py: module-level skipif when data/corpus/{heldout,val}.parquet or
   models/qwen2.5-0.5b-instruct missing; apply to both tests.
2. test_export_ui.py: skipif on test_replays_match_alpha_pipeline when val.parquet /
   models/qwen2.5-1.5b-instruct / ckpt/dpo missing.
3. test_replays_near_alpha_reference: compare only kinds present in both replays and the reference
   (assert the intersection is non-empty).
4. Run acceptance in /tmp/cc and in /work.
