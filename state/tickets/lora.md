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
1. Write test_lora.py, run → fails (module missing).
2. `draftlab/drafters/lora.py`: pack train-block token stream into seq-len chunks
   (seeded shuffle; pad mask); LoRA r=16 alpha=32 on q/k/v/o + gate/up/down;
   `train` subcommand: [--tiny --fsdp --qlora --steps --seq-len --lr --out --seed];
   torchrun init gloo(CPU)/nccl(GPU), FSDP(use_orig_params=True); per-rank
   `rank{i}_lora.safetensors`; non-fsdp: PeftModel.save_pretrained(out).
3. Iterate until tests green; ruff; full pytest.
4. Default 300-step run → ckpt/lora; alpha eval → results/alpha/lora.json; compare vs
   ngram.json + 0.7675 control; NOTES.md; commit.
