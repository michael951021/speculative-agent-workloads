# Ticket: REPORT and README

## Issue
The project's measured numbers (alpha, sweep speedups, judge) live in scattered JSON under results/ but nothing answers the goal question in prose: "how many agent tokens could a draft model have guessed, per kind of content, and what speedup does that buy?" No REPORT.md, no README.md, no figures.

## Question
Can REPORT.md + README.md state the headline answers (drafter × kind α, measured speedups, judge verdict, failures) with CIs, backed by committed figures and reproducible commands, with every number traceable to a results/ file?

## Evidence
- results/sweep_batch{2,3}_table.json (42 cells, pooled acc/tpc/tok/s/speedup vs none), results/sweep.parquet (16506 rows).
- results/alpha_sweep_*.json (metric α per drafter×kind, target 0.5B-Instruct, val) and results/alpha/*.json (α vs 1.5B target).
- results/judge.json (spec_vs_drafter win 0.625; spec_vs_target 0.444), results/ngram.json, distill_curve.json, dpo.json, corpus_stats.json.
- web/public/data/*.json — reran `python -m draftlab.export_ui --no-replays` (exit 0), 0 pending cells.
- Check: tools/check_report.py (recomputes headline numbers from parquet/JSON, asserts REPORT/README contain them; figures exist). Run before/after.

## Done when
`python tools/check_report.py` exits 0: REPORT.md contains the drafter×kind α table, measured speedups, judge result, failure section; every quoted number matches results/ within tolerance (recomputed); ≥3 PNG figures in results/fig/; README.md has a mermaid diagram, quickstart, and skills-map table covering all 13 skills. Known-answer checks reused: base drafter k8 tpc=8.0 exact (identity draft), ngram T=0.7 k1 acc 0.3265 matches hand-computed E[accept]=P7[d] ref 0.3295 (CI∋0). Plausible ranges: metric α lora∈[0.5,0.9] (Leviathan anchor), ngram α∈[0.1,0.4], speedup∈[0.2,1.2] on CPU (draft cost ≥ gain expected).

## Stop when
Scope = REPORT.md, README.md, results/fig/*, tools/check_report.py only. Stop if results files lack a needed number (record which) — do not rerun sweeps. Two attempts without new evidence → record blocker.

## Plan
1. Write tools/check_report.py, run (must fail).
2. Generate figures (matplotlib): α by kind; tpc vs k; measured speedup vs none; distill KL curve.
3. Write REPORT.md, README.md.
4. Rerun check (pass), make check, commit, merge master.
