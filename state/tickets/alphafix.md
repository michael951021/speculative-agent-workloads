# Fix alpha alignment, add known answers (id: alphafix)

## Issue
`JaxModel.dists`/`HFDrafter.dists` slice logits `[len(ctx): len(ctx)+n]`, but row i
predicts token i+1 — so p[j] is the distribution for token j+1 scored against token j.
Every alpha/greedy number in results/alpha/*.json is off-by-one garbage, and the "0.5B
matches 27B at 1.2%" finding in NOTES.md is the bug, not a real effect. All drafter
comparisons go through this metric.

## Question
After slicing `[len(ctx)-1 : len(ctx)+n-1]` (empty ctx → drop position 0, valid[0]=False)
and switching the default target to Qwen2.5-1.5B-Instruct, does the metric reproduce the
known answers (HF labels= loss, top-1 sanity, analytic alpha cases, planted ngram
effect) and give ngram α in a plausible range? Refutation: CE ≠ HF within 1e-3, or
top-1 < 40%, or repeated-block ngram acceptance < 0.9.

## Evidence
- `draftlab/eval/alpha.py:120,148` (the slices), `:31` TARGET_DIR=0.5b; TODO.md:6 names
  1.5B-Instruct as target T.
- Known-answer tests (real models/qwen2.5-0.5b, skipif absent): per-position CE vs HF
  `labels=` loss < 1e-3; 0.5B top-1 on val blocks in [0.40, 0.70] (anchor 57.5% on 372
  tok); uniform-q α = 1 − ½Σ|p − 1/V| (independent TV identity) + hand 3-vocab case;
  repeated-block ngram acceptance = (n−1)/n, shuffled block collapses (< 0.5).
- Full run: `python -m draftlab.eval.alpha` → results/alpha/ngram.json (1.5B target).
  Pre-run range: overall α 0.10–0.40 (α = P_target(proposal); proposal==actual at
  33% of positions, target top-1 40–70%); greedy_corpus unchanged ≈ 0.3335 ± 0.03
  (q and ids are identical to the old run — regression check); tool_use highest per kind.

## Done when
- `pytest draftlab/tests/test_alpha.py draftlab/tests/test_alpha_known.py` passes (known
  answers on real 0.5B; skip only if models/ is absent, which it is not here).
- `python -m draftlab.eval.alpha` (defaults) writes results/alpha/ngram.json within the
  pre-run range; `make check` green. NOTES.md corrected (1.2% finding = the bug; tiny.json
  numbers flagged stale).

## Stop when
Scope: alpha.py adapters/target default + tests + rerun + NOTES. No spec-engine changes,
no drafter training. Stop/split after 2 attempts without new evidence; if the 1.5B
400-block run exceeds ~30 min, reduce --blocks (record) rather than loop.

## Plan
1. Ticket (this file). 2. Fix both slices + empty-ctx handling + default target 1.5b-instruct.
3. Add test_alpha_known.py (4 known answers) + update empty-ctx test. 4. Run tests, then
   full ngram run, compare vs pre-run range and old greedy_corpus. 5. NOTES, commit.

## Result
Fixed both slices + default target 1.5b-instruct. All 4 known answers pass on real
0.5B (CE max diff < 1e-3 vs HF labels=, top-1 42.4% on 1929 tok, uniform-q TV identity,
repeated-block ngram 63/64 + shuffled null). Full run 1184 s: alpha 0.3341, greedy
0.3681, greedy_corpus 0.3335 (identical to pre-fix run -> regression OK), tok/call@k8
1.50. Pre-run range 0.10-0.40 hit. make check green (55 passed).
