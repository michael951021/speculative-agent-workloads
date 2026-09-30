"""OpenAI-compatible FastAPI server over the draftlab spec-decoding engine.

Model names select base + drafter + k: ``<base>`` (default ngram, k=5) or
``<base>+<drafter>-k<K>`` (e.g. ``qwen0.5b+ngram-k4``). ``usage`` adds
``draft_tokens`` (drafts proposed) and ``accepted_tokens`` (drafts accepted).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from draftlab.jaxlm import model as M
from draftlab.jaxlm.spec import ModelDrafter, NGramDrafter, SpecDecoder

DEFAULT_K = 5
DEFAULT_DRAFTER = "ngram"
MAX_TOKENS_LIMIT = 1024
MAX_PROMPT_LIMIT = 4096
_SPEC_RE = re.compile(r"^([a-z]+)-k(\d+)$")


class ChatTokenizer(Protocol):
    def apply_chat_template(self, messages: list[dict]) -> str: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class HFChatTokenizer:
    """Wrap a HuggingFace tokenizer; chat template comes from it."""

    def __init__(self, path: str) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(path)

    def apply_chat_template(self, messages: list[dict]) -> str:
        out = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return out if isinstance(out, str) else self.tok.decode(out.input_ids)

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)


class ToyTokenizer:
    """Vocab-512 stand-in tokenizer for tiny-config tests."""

    vocab_size = 512

    def apply_chat_template(self, messages: list[dict]) -> str:
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        parts.append("assistant:")
        return "\n".join(parts)

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(33 + (i % 90)) for i in ids)


@dataclass
class Engine:
    name: str  # base model name
    cfg: M.Qwen2Cfg
    params: dict
    tokenizer: ChatTokenizer


def parse_engine_spec(model: str) -> tuple[str, str, int]:
    """`qwen0.5b+ngram-k4` -> (base, drafter, k); bare `qwen0.5b` -> defaults.

    Raises ValueError on a malformed suffix.
    """
    base, drafter, k = model, DEFAULT_DRAFTER, DEFAULT_K
    if "+" in model:
        base, spec = model.split("+", 1)
        m = _SPEC_RE.match(spec)
        if m:
            drafter, k = m.group(1), int(m.group(2))
        else:
            drafter = spec
    if not base or k < 1 or k > 32:
        raise ValueError(model)
    return base, drafter, k


def make_drafter(kind: str, engine: Engine, max_len: int):
    if kind == "ngram":
        return NGramDrafter(vocab_size=engine.cfg.vocab_size, max_n=4)
    if kind == "model":
        # placeholder: self-drafts with the target's own weights
        return ModelDrafter(engine.cfg, engine.params, max_len)
    raise ValueError(f"unknown drafter {kind!r}")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64, ge=1, le=MAX_TOKENS_LIMIT)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = Field(default=0)
    stream: bool = False


def _err(request: Request, status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "invalid_request_error", "code": code,
                           "param": None, "request_id": request.headers.get("x-request-id")}},
    )


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def create_app(engines: dict[str, Engine]) -> FastAPI:
    app = FastAPI(title="draftlab engine server")

    @app.exception_handler(RequestValidationError)
    async def _bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _err(request, 400, f"invalid request: {exc.errors()}", "invalid_request")

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": e.name, "object": "model", "owned_by": "draftlab",
                 "drafter": DEFAULT_DRAFTER, "k": DEFAULT_K}
                for e in engines.values()
            ],
        }

    @app.post("/v1/chat/completions")
    def completions(request: Request, body: ChatRequest):
        try:
            base, drafter, k = parse_engine_spec(body.model)
        except ValueError:
            return _err(request, 400, f"bad model name {body.model!r}", "invalid_model_name")
        if drafter not in ("ngram", "model"):
            return _err(request, 400, f"unknown drafter {drafter!r}", "invalid_drafter")
        engine = engines.get(base)
        if engine is None:
            return _err(request, 404, f"model {base!r} not found", "model_not_found")
        if not body.messages:
            return _err(request, 400, "messages must be non-empty", "empty_messages")
        for m in body.messages:
            if m.role not in ("system", "user", "assistant"):
                return _err(request, 400, f"bad role {m.role!r}", "bad_role")

        prompt_text = engine.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in body.messages]
        )
        prompt_ids = engine.tokenizer.encode(prompt_text)
        if len(prompt_ids) > MAX_PROMPT_LIMIT:
            return _err(request, 400, "prompt too long", "prompt_too_long")
        max_len = len(prompt_ids) + body.max_tokens + k + 8
        decoder = SpecDecoder(
            engine.cfg, engine.params, make_drafter(drafter, engine, max_len),
            max_len=max_len, k=k, temperature=body.temperature, seed=body.seed,
        )
        cid = _completion_id()
        created = int(time.time())

        if not body.stream:
            toks, stats = decoder.generate(prompt_ids, body.max_tokens)
            content = engine.tokenizer.decode(toks)
            usage = _usage(len(prompt_ids), len(toks),
                           sum(s.drafted for s in stats), sum(s.accepted for s in stats))
            return {
                "id": cid, "object": "chat.completion", "created": created,
                "model": body.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop", "logprobs": None}],
                "usage": usage,
            }

        def gen():
            draft_tokens = accepted_tokens = 0
            completion_tokens = 0
            last_st = None
            for tok, st in decoder.generate_stream(prompt_ids, body.max_tokens):
                if st is not last_st:  # one StepStats per target-call step
                    draft_tokens += st.drafted
                    accepted_tokens += st.accepted
                    last_st = st
                completion_tokens += 1
                delta: dict[str, Any] = {"role": "assistant",
                                         "content": engine.tokenizer.decode([tok])}
                chunk = {
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": body.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            usage = _usage(len(prompt_ids), completion_tokens,
                           draft_tokens, accepted_tokens)
            final = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": body.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _usage(prompt_tokens: int, completion_tokens: int,
           draft_tokens: int | None = None, accepted_tokens: int | None = None) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "draft_tokens": draft_tokens if draft_tokens is not None else 0,
        "accepted_tokens": accepted_tokens if accepted_tokens is not None else 0,
    }


def main(argv: list[str] | None = None) -> None:
    """Serve a real model: `python -m draftlab.serve [--model KEY] [--alias NAME] [--port P]`."""
    import argparse

    import uvicorn

    from draftlab.models import MODEL_DIR

    ap = argparse.ArgumentParser(prog="python -m draftlab.serve")
    ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
    ap.add_argument("--alias", default="qwen0.5b", help="short model name")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args(argv)
    d = MODEL_DIR / args.model
    if not (d / "config.json").exists():
        raise SystemExit(f"missing {d} — run `python -m draftlab.models fetch --name {args.model}`")
    cfg, params = M.load_safetensors(d)
    tok = HFChatTokenizer(str(d))
    engines = {name: Engine(name, cfg, params, tok) for name in {args.alias, args.model}}
    uvicorn.run(create_app(engines), port=args.port)


if __name__ == "__main__":
    main()
