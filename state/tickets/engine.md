# engine: OpenAI-compatible FastAPI server over the spec-decoding engine

## Issue
The gateway, judge and explorer need a real backend that emits draft acceptance
numbers. Today `SpecDecoder` is only reachable in Python; no HTTP surface exists.

## Question
Can `POST /v1/chat/completions` on a tiny_config engine return OpenAI-shaped
responses (stream + non-stream) whose `usage` carries `draft_tokens` /
`accepted_tokens`, matching a direct `SpecDecoder.generate` call?
Refutation: chunk order/usage diverges from the direct call, or streams don't
concatenate to the non-stream text.

## Evidence
- `draftlab/jaxlm/spec.py:SpecDecoder.generate` (tokens + per-step StepStats).
- `draftlab/models.py:tiny_config` for fast CPU tests.
- Check: `pytest draftlab/tests/test_serve.py` (TestClient), then `make check`.
  Observed vs guess: nothing built yet; all behavior is design.

## Done when
- GET /v1/models lists engine names; POST /v1/chat/completions supports
  `stream` (SSE, delta order, final usage chunk, `data: [DONE]`) and non-stream
  (usage incl. draft_tokens/accepted_tokens).
- Model name `KEY+<drafter>-k<K>` (ngram|model) selects drafter and k.
- 400 on bad input (missing model/messages, unknown model, empty messages).
- Tests green on tiny_config; `make check` green.

## Stop when
No GPU work, no gateway/judge integration, no auth/rate-limit (gateway ticket).
Two failed attempts at one test without new evidence → record blocker, stop.

## Plan
1. `spec.py`: factor `generate` loop into a per-step generator; add
   `generate_stream` yielding (token, step) as tokens emit.
2. `draftlab/serve/app.py`: tokenizer protocol (HF chat template wrapper +
   TinyChatTokenizer stand-in), name parser, FastAPI app factory, SSE streaming.
3. `draftlab/tests/test_serve.py`: models list, non-stream usage, stream chunk
   order + final usage, 400s.
4. Run tests + `make check`; commit.
