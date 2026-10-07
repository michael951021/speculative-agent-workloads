# dpo — Acceptance-aware DPO

## Issue
SFT (LoRA) already gives α 0.7306 overall, but the drafter has never been trained on the
target verifier's accept/reject signal. DPO with the verifier's own signal as the preference
is the standard RLHF-style step; the payoff question: does it move α per kind at all?

## Question
Does TRL DPO (β=0.1) on (ctx, chosen=accepted/resampled token, rejected=drafter's rejected
token) pairs from T=1 spec runs of the 1.5B target with the LoRA drafter raise corpus α vs
the SFT-only adapter? Refutation: α after ≤ α before − CI margin on val, or margin on the
train pairs doesn't increase (DPO objective not working).

## Evidence
- Baseline: results/alpha/lora.json (α 0.7306 [0.7302, 0.7308], 400 val, target 1.5b-instruct).
- Adapter ckpt/lora; target models/qwen2.5-1.5b-instruct; train prompts from data/corpus/train.parquet.
- First check: tests/test_dpo.py — pair builder on a deterministic fake spec log (exact
  expected pairs) + toy p/q acceptance loop: emitted tokens must match target p (chi2).
  Outputs: .agent-evidence/dpo-*.log; pairs cache data/pairs/dpo.jsonl.

## Done when
1. `.venv/bin/python -m pytest speculative_agent_workloads/tests/test_dpo.py -q` passes (fake-log builder, toy
   marginal chi2, tiny DPO run decreases loss).
2. Default run `python -m speculative_agent_workloads.drafters.dpo` (<30 min): spec run collects 2000 pairs,
   DPOTrainer β=0.1 trains on them, saves ckpt/dpo, writes results/dpo.json with before
   (from lora.json) + after (from results/alpha/dpo.json when present) per-kind α + CIs,
   pair stats, DPO loss curve, train-pair margin before/after.
3. After-eval `python -m speculative_agent_workloads.eval.alpha --drafter hf --draft-model ckpt/dpo --out
   results/alpha/dpo.json`; merged into results/dpo.json.
Known-answer checks: fake log (exact pairs); toy chi2 (spec-sampling marginal == target p);
train-pair margin (logπc−logπr) after > before. Pre-written plausible range: after α ∈
[0.70, 0.78] (before 0.7306); |Δα| > 0.1 ⇒ bug until shown otherwise.

## Stop when
Out of scope: k>1 drafts, GPU paths, full fine-tune, other targets. If spec-run throughput
makes 2000 pairs > 25 min at prompt 192/batch 16, shrink L (decode length) not pair count;
after 2 failed attempts at a step with no new evidence, record blocker and stop.

## Plan
1. Tests first (fake log, toy chi2), run → red.
2. dpo.py: pairs (batched T=1 spec decode, KV cache, bf16 target/drafter) → cache jsonl.
3. dpo.py: DPOTrainer (prompt=last 48 ctx tokens, completion 1 tok, ref=adapter-disabled) → ckpt/dpo.
4. Probe step timings; set defaults for <30 min; run; alpha eval; merge report; make check.
