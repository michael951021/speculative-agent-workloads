# Ticket: Results export (id: export)

## Issue
The web explorer (next ticket) has no data to show. There is no versioned JSON
contract between the Python results (`results/*.json`, `results/sweep.parquet`)
and `web/`, and no token-level replay data for the "which tokens would a drafter
have guessed" page.

## Question
Can one module (`draftlab/export_ui.py`) deterministically derive four small
versioned files — `web/public/data/{summary,alpha,sweep,replays}.json` — from the
existing results, marking not-yet-run sweep cells as `pending` instead of
failing? Refutation: any of the four files missing/malformed, or the export
crashes while batch 3 is still running (21 of 42 cells absent from sweep.parquet).

## Evidence
- `results/alpha/{ngram,lora,dpo,tiny}.json` (target qwen2.5-1.5b-instruct, 400 val blocks); best drafter = argmax `overall.alpha` = dpo (0.7319).
- `results/sweep.parquet`: 17 of 42 cells (T=0 only); `results/corpus_stats.json`; 400-block pool from `load_blocks` has ≥5 blocks per each of the top-6 token-weighted kinds.
- Replays: 30 val blocks (5 per kind, top-6 kinds by n_tokens in the best drafter's alpha file: thinking, tool_use:Bash, tool_use:Edit, tool_result, text, code; round-robin over runs). Per token: target prob p(x), drafter prob q(x) (null when ngram has no proposal), accept = greedy spec-accept (argmax q == argmax p, and proposed), plus per-position alpha. Same target/ctx (1.5b-instruct, ctx 1024) and block pool as the alpha runs so stored alphas reproduce `results/alpha/dpo.json` per-kind values.
- Check: `draftlab/tests/test_export_ui.py` (schema + cross-file consistency + replay-vs-alpha known-answer), output in `.agent-evidence/`.

## Done when
1. `python -m draftlab.export_ui` (default run < 30 min CPU; measured 157s) writes all four files; exit 0 with 21 cells pending.
2. `.venv/bin/python -m pytest draftlab/tests/test_export_ui.py` passes (8/8, 287s): schema checks on all four files; **known answer** = `test_replays_match_alpha_pipeline` re-runs `alpha.evaluate()` (the exact pipeline behind dpo.json) on the same 30 blocks and requires per-position α and greedy accept agreement <1e-5; plus a fast pure-read mean-vs-`results/alpha/dpo.json` sanity (tol 0.15 ≈ 3 SE for 5 blocks/kind); summary done-cells equal sweep.json cells; replays = 30 blocks, exactly 5 per kind, 6 kinds.
3. Rerunning after the sweep adds the new cells without pending for them (code path: cell key present in parquet ⇒ done).

### Post-hoc note (known-answer tolerance)
Initial 0.05 mean-tolerance failed (gaps −0.03…−0.11, all one sign). Root cause was selection, not pipeline: round-robin picks the first/short block per kind per run, and the 400-pool contains duplicate (run,seq) rows (val: 15 blocks share (contrib-loop-w1-…, 28885)) so reference per-kind means weight some blocks up to 11×. The per-position identity test (above) proves the replay pipeline is bit-exact vs the reference; the 0.15 mean check stays as drift guard.

## Stop when
Scope: no web UI code (next ticket), no new result files in `results/`, no GPU. Stop/split after 2 failed attempts with no new evidence (e.g. 1.5B prefill too slow on shared CPU → split replay to a `--replay-blocks` budget).

## Plan
1. Write schema test first; run it (fails: files absent).
2. Implement `export_ui.py`: fast paths (summary/alpha/sweep from JSON+parquet) and replay path (reuse `draftlab.eval.alpha` load/scoring).
3. Time one 1.5B prefill; run full export under capture.py; iterate.
4. Run pytest; update NOTES.md; commit.
