# draftlab — current state

## Toolchain (id: boot, done 2026-09-29)
- `make bootstrap` (idempotent): uv + Py3.12 .venv, torch CPU first, `uv pip install -e
  ".[all]"`, pinned binaries into .tools/ (go at .tools/go/bin). `make check` = ruff +
  pytest, then go/vitest/helm/terraform/hadolint/promtool `SKIP <name> (<dir> absent)`.
  jax 0.11.2, torch 2.14.0+cpu, triton 3.8.0, ray 2.58.0.
- Gotchas: capture.py rejects `VAR=x cmd` (use `sh -c`) and existing log paths (rm first).
  unanchored `data/` in .gitignore shadowed the package — now `/data/`.

## Models (id: models, done 2026-09-29)
- `draftlab/models.py`: `fetch [--name K]` → `models/{qwen2.5-0.5b-instruct,
  qwen2.5-1.5b-instruct, qwen2.5-0.5b}` (~4.9 GB), idempotent; `info` asserts shared
  vocab 151936. `tiny_config()` = 2-layer d=64 vocab-512 Qwen2 for fast tests.

## Corpus (id: corpus, done 2026-09-29)
- `python -m draftlab.data build`: raw jsonl → data/corpus/{train,val,test}.parquet
  (fenced code split; drop <8 tok, dedupe, truncate 4096; runs by sha1: 26/3/3).
  45257/667/566 blocks. `stats` → results/corpus_stats.json. p50 tokens: tool_result 243,
  thinking 132, code 101, tool_use 85, text 29.

## N-gram baseline (id: ngram, done 2026-09-29)
- `draftlab/drafters/ngram.py`: NGramMatcher.proposals — prompt lookup n=1..4, first
  occurrence, no self-match. `python -m draftlab.drafters.ngram` → results/ngram.json.
  Greedy accept / tok per call @k=8: tool_use 0.548 (3.34), tool_result 0.309 (2.08),
  code 0.286 (1.89), thinking 0.246 (1.73), text 0.199 (1.56); overall 0.329 (1.76).

## JAX model (id: jaxmodel, done 2026-09-29)
- `draftlab/jaxlm/model.py`: pure-pytree Qwen2; `load_safetensors` upcasts bf16→f32;
  padded KV cache; one jitted step does prefill + multi-token decode.
- Parity 0.5B f32 vs transformers: prefill max abs diff 1.3e-4, argmax identical, greedy
  32 tokens identical (torch generate needs `eos_token_id=-1`). Tests: test_jaxmodel.py.
- Gotcha: tiny_config leaves `tie_word_embeddings=False` (HF default) — JAX gets a
  separate random lm_head unless the probe ties it too.

## Spec decoding (id: spec, done 2026-09-29)
- `draftlab/jaxlm/spec.py`: Drafter protocol (propose/rollback); NGramDrafter +
  ModelDrafter. One target call scores m drafts + bonus (draft j's dist is prev row
  (j=0) or scored[j-1], bonus is scored[m-1]); accept u·q[t]<p[t], residual max(0,p−q).
  Rollback: truncate len to L+A, then decode the bonus.
- `SpecDecoder._steps` generator yields (emitted, StepStats) per step; `generate` and
  `generate_stream` (per-token) consume it. Tests (7): T=0 == greedy; T=1 chi2 p=0.391;
  rollback cache == fresh prefill.

## Engine server (id: engine, done 2026-09-29)
- `draftlab/serve/app.py`: FastAPI, OpenAI-compatible. POST /v1/chat/completions (SSE
  stream + non-stream), GET /v1/models; usage adds draft_tokens/accepted_tokens.
  Name `<base>[+<drafter>-k<K>]` (ngram|model; model = self-draft placeholder).
  HFChatTokenizer (`apply_chat_template(tokenize=False)` — this transformers version
  returns a BatchEncoding otherwise) or ToyTokenizer (vocab-512 stand-in for tiny).
  `python -m draftlab.serve --model K --alias A --port P` (default 0.5b-instruct, :8100).
  400 on bad input (RequestValidationError → 400 handler; pydantic defaults to 422),
  404 unknown model.
- Tests: test_serve.py (11, tiny_config, TestClient): chunk order/determinism vs
  non-stream, usage fields, 400s. Live smoke (0.5B): non-stream + SSE ok.
  `make check` green (39 pytest). Evidence: .agent-evidence/serve-test.log,
  serve-check2.log, serve-live3.log.
