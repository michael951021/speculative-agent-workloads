"""Serve app: OpenAI-compatible endpoints over a tiny_config spec engine."""

import json

import jax
import pytest
from fastapi.testclient import TestClient

from speculative_agent_workloads.jaxlm import model as M
from speculative_agent_workloads.models import tiny_config
from speculative_agent_workloads.serve.app import Engine, ToyTokenizer, create_app

MSGS = [{"role": "user", "content": "say hello"}]


@pytest.fixture(scope="module")
def client():
    cfg = M.Qwen2Cfg.from_hf(tiny_config())
    params = M.init_params(cfg, jax.random.PRNGKey(0))
    app = create_app({"tiny": Engine("tiny", cfg, params, ToyTokenizer())})
    return TestClient(app)


def _body(model="tiny+ngram-k2", stream=False, **kw):
    return {"model": model, "messages": MSGS, "max_tokens": 32, "seed": 0, "stream": stream, **kw}


def _sse_chunks(resp):
    assert resp.status_code == 200
    lines = [line for line in resp.text.split("\n") if line.startswith("data: ")]
    assert lines and lines[-1] == "data: [DONE]"
    return [json.loads(line[len("data: ") :]) for line in lines[:-1]]


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert "tiny" in [m["id"] for m in r.json()["data"]]


def test_nonstream_usage(client):
    j = client.post("/v1/chat/completions", json=_body()).json()
    assert j["object"] == "chat.completion"
    ch = j["choices"][0]
    assert ch["finish_reason"] == "length"  # toy tokenizer has no EOS
    assert ch["message"]["role"] == "assistant"
    assert len(ch["message"]["content"]) == 32
    u = j["usage"]
    assert u["prompt_tokens"] > 0
    assert u["completion_tokens"] == 32
    assert u["total_tokens"] == u["prompt_tokens"] + 32
    assert 1 <= u["accepted_tokens"] <= u["draft_tokens"] <= 64  # k=2 per step


def test_stream_chunks_and_usage(client):
    chunks = _sse_chunks(client.post("/v1/chat/completions", json=_body(stream=True)))
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1
    assert all(c["created"] == chunks[0]["created"] for c in chunks)
    deltas = [c["choices"][0]["delta"] for c in chunks]
    assert deltas[0]["role"] == "assistant"
    content = "".join(d.get("content", "") for d in deltas)
    assert len(content) == 32
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] == "length"
    assert last["choices"][0]["delta"] == {}
    # deterministic (seed 0, T=0): stream must equal the non-stream result
    ns = client.post("/v1/chat/completions", json=_body()).json()
    assert content == ns["choices"][0]["message"]["content"]
    assert last["usage"]["prompt_tokens"] == ns["usage"]["prompt_tokens"]
    assert last["usage"]["draft_tokens"] == ns["usage"]["draft_tokens"]
    assert last["usage"]["accepted_tokens"] == ns["usage"]["accepted_tokens"]


@pytest.mark.parametrize(
    "body",
    [
        {"model": "tiny", "messages": []},  # empty messages
        {"model": "tiny", "messages": [{"role": "bogus", "content": "x"}]},
        {"model": "tiny+bogus-k2", "messages": MSGS},
        {"model": "tiny+ngram-abc", "messages": MSGS},
        {"model": "tiny+ngram-k0", "messages": MSGS},
        {"model": "tiny", "messages": MSGS, "max_tokens": 0},
    ],
)
def test_bad_input_400(client, body):
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 400, body


def test_missing_fields_400(client):
    assert client.post("/v1/chat/completions", json={"messages": MSGS}).status_code == 400
    assert client.post("/v1/chat/completions", json={"model": "tiny"}).status_code == 400


def test_unknown_model_404(client):
    assert client.post("/v1/chat/completions", json=_body(model="nope")).status_code == 404


def test_eos_stops_generation():
    """A token in tokenizer.eos_ids ends the completion (excluded, finish 'stop')."""
    cfg = M.Qwen2Cfg.from_hf(tiny_config())
    params = M.init_params(cfg, jax.random.PRNGKey(0))
    tok = ToyTokenizer()
    c = TestClient(create_app({"tiny": Engine("tiny", cfg, params, tok)}))
    full = c.post("/v1/chat/completions", json=_body(model="tiny+none")).json()
    text = full["choices"][0]["message"]["content"]
    # ToyTokenizer.decode maps id -> chr(33 + id % 90); mark the 4th token's ids as EOS
    stop_ch = text[3]
    tok.eos_ids = frozenset(i for i in range(tok.vocab_size) if chr(33 + i % 90) == stop_ch)
    cut = text.index(stop_ch)
    for stream in (False, True):
        r = c.post("/v1/chat/completions", json=_body(model="tiny+none", stream=stream))
        if stream:
            chunks = _sse_chunks(r)
            content = "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks)
            fin, usage = chunks[-1]["choices"][0]["finish_reason"], chunks[-1]["usage"]
        else:
            j = r.json()
            content, fin, usage = (
                j["choices"][0]["message"]["content"],
                j["choices"][0]["finish_reason"],
                j["usage"],
            )
        assert content == text[:cut] and fin == "stop"
        assert usage["completion_tokens"] == cut


def test_none_drafter_drafts_nothing(client):
    j = client.post("/v1/chat/completions", json=_body(model="tiny+none")).json()
    assert j["usage"]["draft_tokens"] == 0 and j["usage"]["completion_tokens"] == 32
    spec = client.post("/v1/chat/completions", json=_body()).json()
    assert j["choices"][0]["message"]["content"] == spec["choices"][0]["message"]["content"]
