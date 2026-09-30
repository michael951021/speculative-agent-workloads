# LoRA drafter with FSDP (id: lora)

## Issue
We have an n-gram baseline and a weak distilled 43M drafter (α≈0.01, tiny.json), but no
production-style drafter: a full 0.5B model fine-tuned with LoRA on the 27B agent's own
traffic. Needed to answer "does style-matching the agent raise acceptance?" with the
standard LoRA/QLoRA + FSDP toolbox.

## Question
Does LoRA (r=16, attention+MLP) SFT of Qwen2.5-0.5B on packed train blocks, trained with a
working 2-rank FSDP (gloo/CPU) path, yield a drafter whose per-kind α against the 1.5B
target is ≥ the untrained 0.5B control (0.7675, alphafix notes) and inside 0.5–0.9?
Refuting observation: overall α < 0.7 (i.e. degraded vs control) or FSDP 2-rank run not
producing identical adapter weights / non-decreasing loss.

## Evidence
- `draftlab/eval/alpha.py` HFDrafter already loads a PEFT adapter dir (adapter_config.json).
- `results/alpha/ngram.json` (α 0.3341 overall, 400 blocks) = baseline to beat; recorded
  untrained 0.5B control 0.7675 (NOTES, alphafix a24) = anchor.
- Smallest check: `draftlab/tests/test_lora.py` — (1) fresh LoRA adapter is exactly
  identity (lora_B zero ⇒ logits == base logits, max diff 0, tiny config); (2)
  `torchrun --nproc_per_node 2` tiny config, 5 steps: loss decreases and both ranks save
  bit-identical adapter weights. Run first, must fail before lora.py exists.
- Measured (not guessed): 0.5B f32 fwd+bwd 512 tok = 2.3 s/step ⇒ 300 steps ≈ 12 min.
- **Bug found by the anchor check (a2):** first 300-step run + alpha (
  `.agent-evidence/lora-alpha-eval.log`): overall α = 0.0078, ~100x below the 0.7675
  anchor → off-by-one in `step_loss` (logits[t] was compared with ids[t]: trained as a
  token-echo machine; train loss → 0.007, val α ~ 0). Fix: HF alignment
  `CE(logits[:, :-1], labels[:, 1:], ignore_index=−100)` + new known-answer test
  `test_step_loss_matches_hf_labels_loss` (== `model(ids, labels=ids).loss` to <1e-5).
  Consequence: the tiny i.i.d. stream became unlearnable ⇒ tiny stream = repeated
  8-token motif (deterministic, learnable). FSDP2 note: FSDP1 dropped CPU in torch 2.14;
  `fully_shard` + DTensor works on gloo/CPU, gather via `full_tensor()`.
- **Final numbers (a3, `.agent-evidence/lora-train2.log`, `lora-alpha-eval2.log`):**
  retrain 300 steps = 583 s, per-chunk loss 2.37→1.97 (real LM range). Alpha (400 val
  blocks, 29657 tok): overall α = 0.7306 [0.7302, 0.7308] — inside pre-written [0.7, 0.95],
  ≥ the 0.7 refutation bar, so no fresh-control re-run triggered. vs untrained 0.5B
  control 0.7675 (recorded, alphafix a24, same fixed metric/400 val blocks): −0.037 —
  a 155k-token/300-step budget does not beat the base yet (honest answer to "does
  style-matching raise acceptance?": not measurable at this budget; adapter ≈ base,
  both ≫ ngram 0.3341). Per-kind: Edit 0.8235, Write 0.8044, Read 0.7998, Bash 0.7669,
  tool_result 0.7442, code 0.7243, thinking 0.6593, text 0.6575, WebFetch 0.5717 —
  tool_use ≥ text ✅ (llama.cpp anchor); tok/call@k8 = 3.49, speedup@k8 = 2.12
  (ngram: 1.50 / ≈1.3). Leviathan band 0.5–0.9: inside.
- Housekeeping note: NOTES.md was 4721 B (>4 KB target) at run start; this run added the
  LoRA section and condensed ~1.5 KB of redundancy across sections (facts preserved),
  net 4996 B. Getting under 4 KB would require deleting other tasks' measured numbers,
  which the "edit only own lines" rule forbids — left for a future run to prune.

## Done when
1. `.venv/bin/python -m pytest draftlab/tests/test_lora.py` — all pass:
   - identity: |logits_peft − logits_base| max == 0 on tiny config;
   - torchrun 2 ranks, tiny, --fsdp, 5 steps: exit 0, last logged loss < first,
     rank0/rank1 saved lora weights allclose (exact).
2. Default run `< 30 min`: `python -m draftlab.drafters.lora train --steps 300` →
   adapter dir `ckpt/lora/` (loadable by HFDrafter).
3. `python -m draftlab.eval.alpha --drafter hf --draft-model ckpt/lora` →
   `results/alpha/lora.json`; overall α in [0.7, 0.95] (plausible range written here
   before the run: untrained 0.5B control = 0.7675, Leviathan anchor 0.5–0.9).
   Known-answer checks: (a) fresh-adapter identity ⇒ α(LoRA@step0) == α(base 0.5B)
   exactly; (b) control anchor 0.7675 — if lora.json overall α < 0.7, re-run
   `--draft-model models/qwen2.5-0.5b` as a fresh control before any conclusion.
4. `--qlora` on CPU: non-zero exit, message says GPU/CUDA required.
5. `make check` green (ruff + pytest + other suites).

## Stop when
No GPU, no bitsandbytes in sandbox — the QLoRA path is guarded code only, verified by the
CPU-exit test, not a 4-bit run. Stop/split after 2 attempts without new evidence on:
FSDP+PEFT weight gathering (fall back: each rank saves raw lora tensors; adapter dir built
by rank 0 from base + tensors), or alpha run > 30 min (drop --blocks to 200, note it).

## Plan
1. Write test_lora.py, run → fails (module missing). ✅
2. `draftlab/drafters/lora.py`: pack train-block token stream into seq-len chunks
   (seeded shuffle; pad mask); LoRA r=16 alpha=32 on q/k/v/o + gate/up/down;
   `train` subcommand: [--tiny --fsdp --qlora --steps --seq-len --lr --out --seed];
   torchrun init gloo(CPU)/nccl(GPU); FSDP2 `fully_shard` (FSDP1 dropped CPU in torch
   2.14 — plan's `use_orig_params=True` FSDP1 not usable); per-rank
   `rank{i}_lora.safetensors`; non-fsdp: PeftModel.save_pretrained(out). ✅
3. Iterate until tests green (5 tests, incl step_loss == HF labels=ids loss <1e-5);
   ruff; full pytest 67/67. ✅
4. Default 300-step run → ckpt/lora; alpha eval → results/alpha/lora.json; compare vs
   ngram.json + 0.7675 control; NOTES.md; commit. ✅ (a3: off-by-one found by the anchor
   check, fixed, re-ran 2+4; α 0.7306)
