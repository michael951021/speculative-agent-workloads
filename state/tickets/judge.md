# Ticket: LLM-as-judge lossless check (id: judge)

## Issue
The lossless claim ("speculative output is as good as the target's") currently rests on
the engine math only. An independent end-to-end check: a third-party LLM judge compares
generated outputs blind and says whether spec ≈ target and drafter-alone ≠ target.

## Question
At T=0.7 on 50 test-split prompts, does an independent judge treat target-only and
speculative (lora+dpo drafter, k=4) outputs as equivalent (no significant losses for spec),
while drafter-alone outputs are significantly worse? Refutation: spec loses to target at
a rate clearly above its win rate (>25% losses, pre-written), or drafter-alone is judged
indistinguishable from spec (win ≈ loss, <15pp gap) — record as finding, not fix.

## Evidence
- Engine: draftlab/jaxlm/spec.py SpecDecoder (temp sampling verified lossless by design:
  accept u·q<p, residual max(0,p−q)); sweep.py load_prompts + merge_adapter for lora_dpo
  (best drafter, α 0.7319 > lora 0.7306, CIs disjoint per dpo ticket).
- Judge: $OLLAMA_BASE_URL (llama.cpp OpenAI /v1, model "qwen-agent", 27B Q8), probed:
  ~1s/call, returns valid JSON on request.
- Smallest check (no models): fake OpenAI server + synthetic outputs in
  draftlab/tests/test_judge.py; planted verdicts give exact known outcomes.
  Saved: test output + real-run log .agent-evidence/judge-real.log.

## Done when
- `pytest draftlab/tests/test_judge.py` passes (fake server; known answers below).
- `python -m draftlab.eval.judge` runs on real inputs (test split, 0.5b-instruct target,
  lora_dpo drafter, T=0.7, n_gen=16) and writes results/judge.json with win/tie/loss +
  Wilson CIs per comparison, order-bias (cross-order disagreement) per comparison,
  ≤150 API calls, cache in data/judge_cache/.
- Known answers, fake server: (a) always-"A" judge → every pair is a win in one order and
  a loss in the other → disagreement rate exactly 1.0, aggregate wins==losses; (b)
  judge ties on identical texts → 0 wins, 0.0 disagreement; (c) second run hits cache,
  0 new calls; (d) call cap respected exactly.
- Real-run range (pre-written): spec_vs_target losses ≤25%, wins 0–40% (lossless ⇒
  win≈loss, most ties); spec_vs_drafter_alone win−loss ≥15pp, losses ≤30%. Outside range
  ⇒ investigate before reporting.

## Stop when
No GPU; no changes to engine/alpha (judge-only reads). Exclude: report writing, UI.
Stop/split after 2 attempts without new evidence (e.g. judge endpoint down, generation
>45 min → shrink n-gen and note).

## Plan
1. test_judge.py with fake OpenAI server (rules above) — run first, red.
2. judge.py: prompts (5/kind, test split, ≤128 tok) → outputs {target, spec(k=4),
   drafter_alone} at T=0.7 (resume in data/judge_outputs.json) → blind pairwise, both
   orders, rubric in prompt, JSON verdict, cache, cap 150 → results/judge.json.
3. Run real check, compare to pre-written range, NOTES.md, commit.

## Result (2026-09-30) — out of pre-written range; recorded as finding, not fix
- Tests 7/7 (fake OpenAI server: order-bias 1.0, tie, cache, cap, retry, wilson).
  Real run: 49 test prompts, T=0.7, n_gen 16, k=4, lora_dpo drafter, 0.5b-instruct
  target; 150/150 calls, 8 cache hits, 24 invalid JSON (1 retry each), 19 jobs
  cap-starved; 1985 s. results/judge.json, log .agent-evidence/judge-real.log.
- spec_vs_target n=81: W/T/L 36/1/44; win 0.444 [0.341,0.553], loss 0.543 [0.435,0.647],
  order-bias 0.087. Pre-written loss ≤25% → OUT. But win-vs-loss 36/44, binom p=0.434
  (|z|≈1.3): no significant loss.
- spec_vs_drafter_alone n=24: W/T/L 15/0/9; win 0.625 [0.427,0.788] vs loss 0.375
  [0.212,0.573]; gap 25pp ≥ 15pp ✓ (direction right, underpowered); loss 37.5% > 30%.
Findings:
1. "Losses ≤25%" presumes ties dominate, but at T=0.7 the spec and target outputs are
   independent draws of the same distribution: 0/46 exact text matches, so the judge
   breaks near-ties ~50/50 instead of tying. The observed 44/54 non-significant split is
   exactly the lossless signature under this protocol. Distribution-level losslessness is
   separately established by test_spec.py chi-square (T=1 and T=0.7, p>0.01). Tie-dominant
   behavior would require T=0 (spec text ≡ target text by construction).
2. test.parquet: block (contrib-loop-20260926-231240, 77976) appears 8× — code-split
   sub-blocks keep the parent (run,seq). load_prompts returns 49 entries, 4 sharing one
   output key → 9 silent generation skips (138/147 written). No verdict contamination:
   all 8 spec_vs_target jobs of the shared key came back invalid/capped.
3. The 150-call cap was eaten by 24 invalid-JSON retries (48 calls) before the 19
   drafter-alone jobs at the end of the queue ran, underpowering that comparison.
   Future runs: stricter JSON-only rubric and/or higher cap.
