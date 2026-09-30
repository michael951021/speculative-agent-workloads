# Ticket: Ray sweep (id: sweep)

## Issue
Report needs measured (not just expected) speedups: draftlab/sweep.py runs the JAX
SpecDecoder end-to-end over a grid of drafters/k/temperature on real val prompts,
under Ray, resumable into results/sweep.parquet.

## Question
On a 0.5B-instruct JAX target, how many tokens per target call and wall-clock
tok/s does each drafter (ngram, tiny, base-0.5b, lora, lora+dpo) achieve at
k ∈ {1,2,4,8}, T ∈ {0, 0.7}, vs no drafter? Refutation: any drafter measuring
slower than no-drafter at its best k with no explanation, or measured acceptance
outside the alpha-metric prediction CI.

## Evidence
- Engine: draftlab/jaxlm/spec.py SpecDecoder (StepStats per step: m, A, wall_ms).
- Drafters: ckpt/{tiny/export (JAX HF), lora, dpo (PEFT adapters on 0.5b, r=16 α=32)},
  ngram in spec.NGramDrafter. Merge adapters into JAX params: W += (α/r)·B@A.
- Prompts: data/corpus/val.parquet, 100 blocks/kind (kind_label), n_tokens ≤ 128.
- Smallest check: 2-cell Ray local sweep (ngram + base @ k=1, T=0, tiny budgets)
  → results rows; base-0.5b @T=0 acceptance must be exactly 1.0 (drafter==target,
  greedy). Saved: test output + probe log .agent-evidence/sweep-probe.log.
- Engine bugs found by this check (both in ModelDrafter, spec.py): (1) propose
  re-decoded the last committed token, so draft 0 was conditioned on a duplicated
  token — fixed by using the cache-extension row as draft 0's dist; (2) rollback
  full-accept branch did not advance _last, so the next propose re-decoded
  already-cached drafts (duplicated tail, drafter context diverges) — fixed with
  _last = last + m. Regression tests: test_model_drafter_first_proposal_position,
  test_model_drafter_full_accept_steps_keep_cache_in_sync (KV cache == fresh
  prefill; red on old code, green on fixed).
- base@T=0 == 1.0 exactly holds at k=1 (every decode T=1 on both sides,
  bit-identical). At k≥2 the target scores m drafts in one T=m call while the
  drafter decodes them T=1; XLA GEMM tiling differs (~1e-4 logit noise) so expect
  ≥0.99, not exactly 1.

## Done when
- `pytest draftlab/tests/test_sweep.py` passes: 2-cell Ray local sweep writes
  results/sweep.parquet; base @T=0 accept_rate == 1.0; re-run skips existing cells.
- Default full grid (40 cells + 2 baseline, 100 prompts/kind) resumable in
  results/sweep.parquet; measured numbers in NOTES.md.
- Known answers: (a) base @T=0, k=1 accept 1.0 exact (k≥2: ≥0.99, GEMM tiling
  noise); (b) engine vs metric — measured
  accept (T=0) per drafter within alpha-metric tokens/call CI for same pair/k
  (run draftlab.eval.alpha with target 0.5b-instruct, n_blocks=100); (c) lora+dpo
  accept ≥ lora accept (CI-disjoint as in dpo ticket) at T=0.
- Plausible range (pre-written): T=0 accept — ngram 0.3–0.45, tiny 0.01–0.1,
  base 1.0, lora 0.85–0.99, lora+dpo ≈ lora±0.02; tok/call@8 = accept-formula
  within ±10% of metric; baseline tok/s > 0 and drafter tok/s ≤ ~2× baseline
  (same-size drafters; tiny can beat baseline).

## Stop when
No GPU; JAX/torch CPU only. Exclude: report writing, UI export, Triton kernel.
Stop/split after 2 attempts without new evidence (e.g. Ray+JAX worker crash,
full grid > 90 min → split by drafter group and note in NOTES.md).

## Plan
1. Probe JAX 0.5b tok/s on this CPU; pick default n-gen so full grid < ~30 min/cell-batch.
2. draftlab/sweep.py: prompt build, drafter factory (adapter merge), Ray local
   (num_cpus flag, cpus-per-cell), per-cell rows → parquet, resume-skip.
3. test_sweep.py (2-cell, small budgets) — run first, red→green.
4. Run full grid incrementally; compare with metric CIs; update NOTES.md.
