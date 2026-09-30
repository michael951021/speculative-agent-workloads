# Ticket: speculative sampling engine (id: spec)

## Issue
No engine yet turns drafters into generated tokens. jaxmodel gives a jit target
with a padded KV cache; ngram gives proposals. We need the core spec-decoding
loop so acceptance stats and (later) served generation are possible.

## Question
Does draft-then-verify with residual resampling emit tokens whose marginal
distribution exactly equals the target's (lossless, Leviathan 2023), and does
the KV rollback leave caches equal to a fresh prefill? Refutation: T=1
first-token counts disagree with the target multinomial (chi2 p < 0.01), or
post-rollback cache differs from a fresh prefill of the committed sequence.

## Evidence
- Observed: `draftlab/jaxlm/model.py` `step`/`prefill`/`decode` (jitted,
  padded cache, `len` = position pointer); `drafters/ngram.py` `NGramMatcher`
  (offline first-occurrence lookups).
- Check: `draftlab/tests/test_spec.py` — (a) T=0 spec output == target greedy
  for ngram and model drafters; (b) T=1, vocab-16 tiny target, 20k first
  tokens vs target softmax: chi2 p > 0.01; (c) after a run with at least one
  partial acceptance, target cache == fresh prefill of committed tokens.
  Logs: .agent-evidence/spec-chi2.log (chi2=15.9 df=15 p=0.391), spec-check.log (make check).

## Done when
`.venv/bin/python -m pytest draftlab/tests/test_spec.py -q` passes; `make check`
green; per-step stats (drafted, accepted, per-position bools, wall ms) returned
by `SpecDecoder.generate`.

## Stop when
Scope: no server, no distillation, no GPU path. If chi2 fails twice with
different seeds, or cache-equality fails after fixing an index error once,
record the blocker in NOTES.md and stop/split — do not chase.

## Plan
1. `draftlab/jaxlm/spec.py`: `Drafter` protocol (propose(ctx,k) → tokens,
   probs; rollback(n_kept)); `NGramDrafter` (incremental, no self-match) and
   `ModelDrafter` (2nd JAX Qwen2 + own cache).
2. `SpecDecoder.generate`: one target call scores m draft positions + bonus;
   vector-free accept loop; residual max(0,p−q) at first rejection; T=0 =
   argmax compare. Target cache rollback = truncate `len` (positional cache);
   full-accept decodes the bonus; draft cache via `drafter.rollback`.
3. Tests above; run, fix, `make check`.
