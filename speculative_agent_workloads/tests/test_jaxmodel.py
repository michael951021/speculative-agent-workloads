"""JAX Qwen2 parity vs transformers (f32 both sides) + tiny-config behavior.

0.5B tests need models/qwen2.5-0.5b (skipped if absent); tiny-config tests run
anywhere. f32-vs-f32 with bf16-upcasted weights: ~24 layers of f32 math must
agree within atol 2e-3.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from speculative_agent_workloads.jaxlm import model as M
from speculative_agent_workloads.models import MODEL_DIR, tiny_config

MODEL = MODEL_DIR / "qwen2.5-0.5b"
PROMPTS = [
    "The quick brown fox jumps over the lazy dog. " * 8,
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "user: what is 27*43?\nassistant: let me think. 27*43 = 27*40 + 27*3 = 1080 + 81 = 1161.",
]


@pytest.fixture(scope="module")
def models_05b():
    if not (MODEL / "config.json").exists():
        pytest.skip(
            f"missing {MODEL} — run `python -m speculative_agent_workloads.models fetch "
            "--name qwen2.5-0.5b`"
        )
    cfg, params = M.load_safetensors(MODEL, dtype=jnp.float32)
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    tm = Qwen2ForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    )
    tm.eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    return cfg, params, tm, tok


def _prompt_ids(tok, text):
    return tok(text, return_tensors="pt").input_ids[0].tolist()


def test_prefill_logits_parity_05b(models_05b):
    cfg, params, tm, tok = models_05b
    for text in PROMPTS:
        ids = _prompt_ids(tok, text)
        jl, _ = M.prefill(cfg, params, ids, max_len=len(ids) + 8)
        with torch.no_grad():
            tl = tm(torch.tensor([ids])).logits[0].numpy()
        diff = float(abs(np.asarray(jl) - tl).max())
        assert diff < 2e-3, f"{text[:40]!r}: max abs diff {diff:.2e}"


def test_greedy_32_tokens_05b(models_05b):
    cfg, params, tm, tok = models_05b
    for text in PROMPTS:
        ids = _prompt_ids(tok, text)
        jl, cache = M.prefill(cfg, params, ids, max_len=len(ids) + 40)
        seq, lg = [], jl[-1:]
        for _ in range(32):
            nxt = int(np.argmax(np.asarray(lg)))
            seq.append(nxt)
            lg, cache = M.decode(cfg, params, cache, [nxt])
        out = tm.generate(
            torch.tensor([ids]),
            max_new_tokens=32,
            do_sample=False,
            num_beams=1,
            eos_token_id=-1,  # disable early EOS stop; compare all 32 steps
        )
        assert seq == out[0, len(ids) :].tolist(), f"{text[:40]!r}: greedy sequences diverge"


def _tiny():
    cfg = M.Qwen2Cfg.from_hf(tiny_config())
    return cfg, M.init_params(cfg, jax.random.PRNGKey(0))


def test_tiny_multi_token_decode_equals_chain():
    cfg, params = _tiny()
    toks = [3, 7, 42, 0, 9, 200, 1, 5, 64, 10, 11, 12, 13, 14, 15, 16, 511, 2]
    # fresh-cache multi-token step == prefill
    multi, _ = M.decode(cfg, params, M.empty_cache(cfg, 64), toks)
    pre, _ = M.prefill(cfg, params, toks, 64)
    assert abs(np.asarray(multi) - np.asarray(pre)).max() < 1e-5
    # chained single-token steps reach the same last-position logits
    cache = M.empty_cache(cfg, 64)
    lg = None
    for t in toks:
        lg, cache = M.decode(cfg, params, cache, [t])
    assert abs(np.asarray(lg)[-1] - np.asarray(pre)[-1]).max() < 1e-5


def test_tiny_wall_time():
    cfg, params = _tiny()
    toks = list(range(64))
    t0 = time.time()
    _, cache = M.prefill(cfg, params, toks, 64 + 32)
    lg, cache = M.decode(cfg, params, cache, [toks[-1] - 1])
    for _ in range(31):
        nxt = int(np.argmax(np.asarray(lg)[-1]))
        lg, cache = M.decode(cfg, params, cache, [nxt])
    assert time.time() - t0 < 5.0
