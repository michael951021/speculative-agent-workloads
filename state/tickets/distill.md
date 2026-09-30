# Ticket: distilled tiny drafter (JAX) (id: distill)

## Issue
No drafter trained on agent traffic exists yet; the alpha run showed a
generic 0.5B agrees with the 27B target on ~1.2% of argmaxes, so coverage,
not the metric, is the bottleneck. We need a very small drafter distilled
from the 0.5B target on agent blocks to (a) prove JAX training works end to
end and (b) test whether tiny + on-domain is enough to beat n-gram.

## Question
Can a 4-layer d=256 Qwen2-shaped student (same 151936 vocab, tied
embedding) trained on the target's top-64 distributions with
KL(T‖student) reduce val KL by ≥ 30% vs its init, and what alpha/greedy
does it then achieve per kind? Refutation: val KL after training > 70% of
step-0 KL (training not working), or the top-64+tail loss has no gradient
signal (constant across inits).

## Evidence
- Observed: `jaxlm/model.py` (batch-less padded jitted prefill; 0.5B ≈ 900
  tok/s at L=1536 on 16 threads); `eval/alpha.py` load_blocks (round-robin
  runs, ctx blocks) + JaxModel (loads config.json+safetensors, pads to
  max_len); corpus train/val splits exist.
- Check: `draftlab/tests/test_distill.py` — (a) KL loss == 0 when student
  logits reproduce the teacher top-64; (b) student_hidden parity vs
  `M.step` (GQA/RoPE/padding); (c) synthetic train reduces val KL
  (memorisation leak on purpose); (d) cached npz sanity. All pass;
  `make check` green (49 passed). Logs: .agent-evidence/distill-*.log.
- Measured: cache 200k tok 596 s (339 tok/s); cache 1M tok 2592 s (394
  tok/s, 26 run files). Train (1M, batch 16 × 256, 1.6 epoch @ 400 steps,
  lr 1e-3): val KL 3.7879 → 2.7796 (26.6%); tail still descending
  (~0.02/75 steps), train loss 2.62 — under-trained, not flat. lr 3e-3
  @ 200k oscillated worse (val 2.95) — abandoned. Overfit diagnostic
  (train on val data, 200k): 2.8874 @ 50 → 2.2487 @ 200, still dropping
  → capacity sufficient; the 400-step floor is an epoch-count problem.
- Measured (2026-09-30): 800 steps @1e-3 → val KL 2.6678 (drop 29.57%, log .agent-evidence/distill-train-800.log) — 0.6% short of 30%. Run D: 1200 steps @1e-3 in progress (log distill-train-1200.log).
- Expected before run (plausible range): tiny student vs 0.5B target alpha overall 0.4-0.9 (Leviathan: ~100x smaller drafter gives 0.5-0.9; student distills the target directly, unlike ngram's 0.0096); greedy ≥ 0.3, must beat ngram alpha 0.0096 on every kind.
- Alpha probe (2026-09-30): self-check α=1.0000/greedy=1.0000 on real blocks
  (known answer ✅ — metric pipeline correct). Student vs 0.5B, train blocks no-ctx:
  α=0.026, greedy=0.295, but H(q)=11.79 ≈ ln V=11.93, student peak=3.8e-4 vs teacher
  peak=0.36, top-64 mass=0.83 → student ≈ uniform; the argmax "match" is shared
  high-freq punctuation (mode `'"'`). Low α is a true distributional gap, not a bug.
  results/alpha/tiny.json (val, ctx): overall α=0.0142, greedy=0.1465 (ngram:
  0.0096/0.0118).
- Teacher-init control (1200 steps, --teacher-init, norm-preserving random
  projection of the 0.5B embed): val KL 3.7707 → 2.6379 (30.0%) ≈ random-init
  2.6292 (30.6%) — no gain; the 2.63 floor is approximation limit, not init.
- Conclusion: tiny drafter is a weak 0.5B approximation (2.63 nats ≈ 3.8 bits vs
  teacher H≈3.5 nats); beats ngram on α (1.5x) and greedy (12x) but is not enough
  alone — acceptance is bottlenecked by the 0.5B↔27B mismatch (~1.2% corpus argmax)
  plus student approximation error. Pre-run α range 0.4–0.9 was mis-anchored on
  Leviathan (whose drafters are strong approximations).

## Status: DONE (2026-09-30)
All Done-when items pass: make check (50 pytest), val KL drop 30.6% ≥ 30%,
results/alpha/tiny.json written. Artifacts: results/distill_curve.json,
results/alpha/tiny.json, ckpt/tiny/export (HF layout), ckpt/tiny_tinit (control).

- Note: teacher hidden is 896, student d=256 — "share T's embeddings vocab"
  = same 151936 vocab, student-own (trainable) embedding; weights cannot be
  shared across the dimension mismatch. Training is block-only (no
  cross-block ctx) to fit the 30-min CPU budget; alpha eval still gives the
  drafter its ctx (model uses the block prefix).

## Done when
`pytest draftlab/tests/test_distill.py` + `make check` pass; val KL at last
step ≤ 0.7 × step-0 val KL (in results/distill_curve.json);
`results/alpha/tiny.json` written by
`python -m draftlab.eval.alpha --drafter jax --draft-model <export> --out results/alpha/tiny.json`.

## Stop when
Scope: no server integration, no GPU, no LoRA/DPO. If val KL does not drop
30% after two distinct runs (different seed/lr), or the top-64 loss stalls
flat for > 200 steps on two lr choices, record the blocker with the curve
paths and stop/split.

## Plan
1. `draftlab/drafters/distill.py`: `cache` (teacher top-64 over train/val
   blocks → data/teacher/{split}_{run}.npz) and `train` (student fwd/loss,
   Optax AdamW + warmup-cosine, orbax CheckpointManager ckpt/tiny/, curve
   → results/distill_curve.json, HF-layout safetensors export for alpha).
2. Tests above; `make check`.
3. 1M cache (ticket default) built. Run A: 400 steps @1e-3 → 26.6% (short).
   Run B: 800 steps @1e-3 (~3.3 epochs). If < 30%, run C: 800 steps @5e-4
   (the two-lr stop check); then alpha → results/alpha/tiny.json.
