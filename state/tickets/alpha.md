# Alpha metric (id: alpha)

## Issue
No cheap, exact way to rank drafters: full speculative generations are slow, and the
ngram baseline only covers one drafter. Need one metric (alpha) computed teacher-forced
that works for any drafter (ngram, JAX model, HF torch model) and converts to speedup.

## Question
For each content kind, does the per-position alpha = sum_x min(p(x), q(x)) (full vocab,
p target 0.5B, q drafter, teacher-forced along val blocks) with bootstrap CIs over runs
separate drafters, and does (1 - a^(k+1))/(1 - a) at a measured cost ratio c give expected
tokens/call and speedup for k in 1..8? Refutation: alpha not 1 for T vs identical T, or
alpha != hand-computed 3-vocab value.

## Evidence
- `speculative_agent_workloads/eval/alpha.py`: adapters with one interface `dists(ctx, ids) -> (q (T,V), valid)`
  (ngram one-hot with a fresh NGramMatcher per call — the matcher's n-gram dict is stateful
  across calls, so a persistent one would leak across blocks/runs); JAX padded prefill (safe:
  causal mask — real rows never attend to padding); HF torch. 6 tests in tests/test_alpha.py
  (hand 3-vocab alpha, valid mask, k tables, kind labels, one-hot + no-state-leak + ctx
  consumption, identical-model alpha=1).
- Default run (631 s, EXIT_CODE=0, .agent-evidence/alpha-run.log): 400 val blocks round-robin
  across 3 runs (333/7/60), 29657 tok, c = 0.00052 (ngram ~free). Overall alpha 0.0096,
  greedy 0.0118, greedy_corpus 0.3335, coverage 0.763; tok/call @k8 = 1.0097, speedup 1.0055.
  Per kind (alpha / greedy_corpus): tool_use:Edit 0.0056/0.5624, :Read 0.0016/0.5249,
  :Bash 0.0077/0.4551, :Write 0.0033/0.3613, tool_result 0.0185/0.3292, code 0.0070/0.2494,
  text 0.0105/0.1835, thinking 0.0085/0.1562.
- Cross-check vs results/ngram.json: greedy_corpus overall 0.3335 vs baseline 0.3286, per-kind
  ordering preserved (tool_use pooled 0.500 <= 0.548 highest; tool_result 0.329, code 0.249,
  text 0.184, thinking 0.156). tool_result 0.3292 is slightly above baseline 0.3088 — my
  selection includes more mid-run blocks; within selection noise.
- Finding: the 0.5B target's argmax agrees with the 27B-agent corpus at only ~1.2% of
  positions (greedy 0.0118 vs greedy_corpus 0.3335). Spec-acceptance is bottlenecked by
  target-drafter disagreement, not drafter coverage — ngram spec speedup on a 0.5B target is
  ~1.00, motivating the distill/LoRA drafter tickets.

## Done when
`python -m speculative_agent_workloads.eval.alpha` (default: ngram drafter, 400 val blocks <=256 tok, ctx
<=1024 tok) writes `results/alpha/ngram.json` with per-kind alpha, alpha_ci95, greedy
(argmax q == argmax p, the real spec-accept), greedy_corpus (argmax q == actual token),
coverage, tokens_per_call and speedup for k=1..8 at measured c, plus overall. Tests:
alpha=1 identical models; alpha matches hand-computed 3-token vocab case. Cross-check:
greedy_corpus <= results/ngram.json greedy_accept overall (mine has less context: only
selected predecessor blocks within a token budget, vs last-8 all blocks) and keeps the
per-kind ordering (tool_use highest).

## Stop when
Scope: metric + aggregation + CLI + tests only; no drafter training (distill/lora tickets),
no spec engine changes. Stop/split after 2 attempts without new evidence if CPU timing of
400-block target pass exceeds budget (>30 min) — then shrink default --blocks and record.

## Plan
1. `speculative_agent_workloads/eval/alpha.py`: pure `alpha_and_greedy(p, q, valid)` (numpy); drafter
   adapters with one interface `dists(ctx, ids) -> (q (T,V), valid (T,))`: ngram one-hot
   (fresh NGramMatcher per call — no cross-block leakage), JAX padded prefill, HF torch.
   Kind labels `tool_use:<Tool>`.
2. Blocks: eligible = n_tokens<=256 (487 val: 420/7/60 per run); selected round-robin
   across runs so 400 spans all 3 (bootstrap unit). Context = same run's earlier selected
   blocks within --ctx-tokens (1024) + block prefix.
3. Per-block teacher-forced p (JAX target 0.5B) and q; per-kind run-level bootstrap
   (resample runs); c = timed draft/target pass ratio; E_k = (1-a^(k+1))/(1-a),
   speedup = E_k/(1+k*c).
4. Tests + default run, cross-check vs results/ngram.json, NOTES/TASKLOG, commit.
