# models — Shared model downloads + tiny test config

## Issue
Every engine, drafter and eval task needs the same Qwen2.5 checkpoints; a shared tokenizer is what
makes cross-model speculative decoding legal. No `speculative_agent_workloads/models.py` exists yet, and `models/` is
empty.

## Question
Does `python -m speculative_agent_workloads.models fetch` download Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct,
Qwen2.5-0.5B (safetensors + tokenizer + config only) into `models/<key>/`, and does
`python -m speculative_agent_workloads.models info` print params/dtype/vocab per model and assert all three tokenizers
share the same vocab? Refutation: any model dir missing safetensors/tokenizer files, or the vocab
assert firing.

## Evidence
- Start: `models/` absent; `huggingface_hub` 1.33.0, `transformers` 5.17.0 in `.venv`;
  huggingface.co returns 200 (curl, this run).
- Check: `python -m speculative_agent_workloads.models info` after fetch; output captured to
  `.agent-evidence/models-info.log`. Observed vs guess: HF network reachable (observed 200);
  download size ~5.2 GB total (guess from HF model cards).

## Done when
- `python -m speculative_agent_workloads.models info` exits 0 and lists all three models with param count, dtype,
  vocab size; vocab-equality assert holds.
- `fetch` is idempotent (re-run downloads nothing new).
- `tiny_config()` returns Qwen2Config(2 layers, d=64, vocab 512); covered by a test.

## Stop when
- Scope: downloads + models.py + tiny_config only; no engine/drafter code.
- Stop/split: two failed fetch attempts with no new evidence (e.g. HF blocked, disk full) → record
  blocker in NOTES.md and stop.

## Plan
1. `speculative_agent_workloads/models.py`: MODELS map, `fetch(key?)` via snapshot_download + allow_patterns, `info()`
   (param count via safetensors headers, dtype, vocab; assert equal `get_vocab()`), `tiny_config()`,
   CLI.
2. `speculative_agent_workloads/tests/test_models.py`: tiny_config fields; vocab-equality (skipif models absent).
3. Run fetch (capture log), `info` (capture log), `make check`; commit.
