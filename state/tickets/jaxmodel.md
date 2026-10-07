# jaxmodel — Qwen2 in JAX (target model for the engine)

## Issue
No from-scratch Qwen2 implementation exists; the spec-decoding engine needs target + drafter in one
process with exposed logits and KV-cache control (a black-box server would hide acceptance data).

## Question
Can a ~pure-pytree JAX Qwen2 (RMSNorm, RoPE@theta, GQA, SwiGLU, QKV bias, tied embeddings) loaded
from HF safetensors reproduce the float32 reference? Refutation: last-position prefill logits off
from transformers by > atol 2e-3 on 3 prompts, or greedy sequences diverging within 32 tokens.

## Evidence
- Start: `models/qwen2.5-0.5b/` (safetensors, bf16; q/k/v_proj have bias, no lm_head → tied
  embeddings, GQA 14 heads / 2 kv heads, head_dim 64, rope_theta 1e6, 24 layers, d=896).
- `speculative_agent_workloads/models.py:tiny_config()` = 2-layer d=64 vocab-512 config for fast tests.
- Check: `pytest speculative_agent_workloads/tests/test_jaxmodel.py` (0.5B f32 vs transformers f32, 3 prompts;
  32 greedy tokens; tiny_config wall time). Log: `.agent-evidence/jaxmodel.log`.
- Observed: transformers 5.17.0, jax 0.11.2, flax 0.12.10 (NNX) in .venv; all 3 models share
  vocab 151936. Guess: f32-vs-f32 parity within 2e-3 (bf16 weights upcast, ~24 layers of f32 math).
- Result (2026-09-29): parity holds. 0.5B f32 vs transformers f32, 3 prompts: max abs logits
  diff over the whole prefill 1.3e-4 (< 2e-3); argmax identical on every row; greedy 32 tokens
  identical (torch generate needs `eos_token_id=-1`: 151643=EOS lands inside 32 steps on the
  arithmetic prompt, where generate stops early). Tiny: multi-token decode == chained
  single-token decode == prefill (<=1.2e-7); prefill 64 + 32 decode wall < 5 s incl. first jit.
  `make check` green (21 tests).
- Debug note: a 1.32 full-prefill mismatch on the tiny config was the probe, not the model:
  `tiny_config()` leaves `tie_word_embeddings` at the Qwen2Config default (False), so
  `init_params` builds a separate random `lm_head` while the probe tied it to embed in the
  torch state_dict. Standalone L0/L1 layers matched torch at ~2e-8 all along; the eager
  replicate of `step` matched at 3.6e-7.
- Loader note: torch bf16 tensors have no `.numpy()` path (numpy has no bf16 dtype);
  `load_safetensors` upcasts bf16->f32 before the numpy conversion.

## Done when
- `speculative_agent_workloads/jaxlm/model.py` exposes a Qwen2 model + preallocated KV cache and jit-compiled
  `prefill` / `decode` (multi-token) steps.
- `pytest speculative_agent_workloads/tests/test_jaxmodel.py` green: logits atol 2e-3 (3 prompts, f32 both sides),
  32 greedy tokens identical to transformers, tiny_config prefill+decode wall < 5 s.
- `make check` green (ruff + pytest).

## Stop when
- Scope: model + cache + load + tests only. No sampling/spec logic (id: spec), no training,
  no serve app, no bf16 parity test.
- Two attempts with unexplained > 2e-3 mismatch (e.g. RoPE/reshape detail) → record evidence,
  stop or split.

## Plan
1. Pure-function pytree (NNX turned out to be unnecessary): RMSNorm, RoPE (on-the-fly from
   positions), GQA attention with QKV bias, SwiGLU MLP.
2. Single `step(ids, cache)` core: prefill (len==0) and multi-token decode share it; cache =
   preallocated (n_kv, Lmax, hd) k/v + len, written with dynamic slice.
3. `load_hf(dir, dtype)` maps safetensors names → params; tied embeddings when no lm_head.
4. Tests: f32 parity vs transformers (3 prompts, 0.5B), greedy 32 identical, tiny_config < 5 s,
   multi-token decode == chained single-token (tiny).
