# N-gram lookup baseline

## Issue
No model-less baseline exists yet. Without one we have no floor for the trained
drafters (distill/LoRA/DPO) to beat, and no free first answer to "how predictable
is agent traffic". Tool calls/results repeat context heavily, so prompt-lookup
should already catch a measurable fraction of tokens.

## Question
On the val split, what fraction of each kind's positions does an n-gram
prompt-lookup drafter (match last n∈{1..4} tokens vs the block's preceding
context + previous 8 same-run blocks) accept greedily, and what mean accepted
length does that give for k∈{1,2,4,8}? Refutation: if acceptance is ~0 on
tool_use/tool_result, the "repeats context" hypothesis is wrong and the
baseline is useless as a floor.

## Evidence
- data/corpus/val.parquet (667 blocks, 236,985 tokens, 3 runs, kinds:
  tool_use 210, thinking 186, tool_result 161, text 101, code 9).
- Matcher correctness check: hand-made token sequence in
  draftlab/tests/test_ngram.py (self-match exclusion, first-occurrence rule,
  longest-n priority, chain lengths).
- Run output: results/ngram.json; log .agent-evidence/ngram-run.log.

## Done when
- `python -m draftlab.drafters.ngram` exits 0 and writes results/ngram.json with
  per-kind greedy acceptance, coverage, per-k mean accepted length /
  tokens-per-call, and bootstrap 95% CIs (runs resampled with replacement).
- pytest draftlab/tests/test_ngram.py passes; `make check` green.

## Stop when
Out of scope: any model/drafter, Triton, serve, sweep. If tokenization or
parquet reading is unavailable, stop and record. Two attempts without a
plausible (nonzero, kind-stratified) result → record blocker and stop.

## Plan
1. draftlab/drafters/ngram.py: NGramMatcher.proposals(ctx) + chain/accept
   helpers (pure, token-list in, proposals out) — unit-testable.
2. evaluate(): tokenize val blocks (Qwen 0.5b tokenizer, cached), context =
   prev-8 same-run blocks + own prefix; per-position proposals; per-kind ×
   per-k aggregates; run-level summaries for bootstrap (B=200, seed 0).
3. Write results/ngram.json; verify numbers sane (tool kinds > prose kinds).
