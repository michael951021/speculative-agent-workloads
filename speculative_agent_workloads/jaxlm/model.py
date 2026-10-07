"""Qwen2 in JAX — target model for the spec-decoding engine.

Pure-function pytree model loaded from HF safetensors (f32 or bf16). One
jit-compiled `step` covers prefill (fresh cache) and single/multi-token
decode: new keys/values are written into a preallocated cache and attention
runs over the whole padded window with a causal+valid mask (positions at or
past `len` are masked out), so there are no dynamic shapes under jit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

NEG = -1e9  # masked attention score (finite; rows always have >= 1 valid key)


@dataclass(frozen=True)
class Qwen2Cfg:
    """Static Qwen2 shape constants (passed to jit as a static arg)."""

    vocab_size: int
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv: int
    head_dim: int
    intermediate_size: int
    rope_theta: float
    norm_eps: float
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, cfg) -> Qwen2Cfg:
        d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
        hidden = d["hidden_size"]
        nh = d["num_attention_heads"]
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=hidden,
            n_layers=d["num_hidden_layers"],
            n_heads=nh,
            n_kv=d.get("num_key_value_heads") or nh,
            head_dim=d.get("head_dim") or hidden // nh,
            intermediate_size=d["intermediate_size"],
            rope_theta=float(d.get("rope_theta", 10000.0)),
            norm_eps=float(d.get("rms_norm_eps", 1e-6)),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", True)),
        )


def init_params(cfg: Qwen2Cfg, rng, dtype=jnp.float32) -> dict:
    """Random init (tiny-config tests); same layout as load_safetensors."""
    D, inter, V = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
    nh, nkv, hd = cfg.n_heads, cfg.n_kv, cfg.head_dim
    keys = jax.random.split(rng, 2 + 7 * cfg.n_layers)

    def z(i, *shape):
        return (jax.random.normal(keys[i], shape) * 0.02).astype(dtype)

    # linear weights in HF layout (out, in): y = x @ w.T
    layers = []
    for li in range(cfg.n_layers):
        b = 2 + 7 * li
        layers.append(
            {
                "input_ln": jnp.ones(D, dtype),
                "attn": {
                    "q_w": z(b, nh * hd, D),
                    "q_b": jnp.zeros(nh * hd, dtype),
                    "k_w": z(b + 1, nkv * hd, D),
                    "k_b": jnp.zeros(nkv * hd, dtype),
                    "v_w": z(b + 2, nkv * hd, D),
                    "v_b": jnp.zeros(nkv * hd, dtype),
                    "o_w": z(b + 3, D, nh * hd),
                },
                "post_ln": jnp.ones(D, dtype),
                "mlp": {
                    "gate_w": z(b + 4, inter, D),
                    "up_w": z(b + 5, inter, D),
                    "down_w": z(b + 6, D, inter),
                },
            }
        )
    params = {"embed": z(0, V, D), "layers": layers, "norm": jnp.ones(D, dtype)}
    if not cfg.tie_word_embeddings:
        params["lm_head"] = z(1, V, D)
    return params


def load_safetensors(path: str | Path, dtype=jnp.float32) -> tuple[Qwen2Cfg, dict]:
    """Load a HF Qwen2 checkpoint (config.json + *.safetensors) into a params pytree."""
    import torch
    from safetensors import safe_open

    path = Path(path)
    cfg = Qwen2Cfg.from_hf(json.loads((path / "config.json").read_text()))
    t = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for name in sf.keys():
                x = sf.get_tensor(name)
                if x.dtype == torch.bfloat16:  # numpy has no bf16; upcast, jnp casts to dtype
                    x = x.to(torch.float32)
                t[name] = x.numpy()

    def a(name):
        return jnp.asarray(t[name], dtype=dtype)

    layers = []
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        layers.append(
            {
                "input_ln": a(p + "input_layernorm.weight"),
                "attn": {
                    "q_w": a(p + "self_attn.q_proj.weight"),
                    "q_b": a(p + "self_attn.q_proj.bias"),
                    "k_w": a(p + "self_attn.k_proj.weight"),
                    "k_b": a(p + "self_attn.k_proj.bias"),
                    "v_w": a(p + "self_attn.v_proj.weight"),
                    "v_b": a(p + "self_attn.v_proj.bias"),
                    "o_w": a(p + "self_attn.o_proj.weight"),
                },
                "post_ln": a(p + "post_attention_layernorm.weight"),
                "mlp": {
                    "gate_w": a(p + "mlp.gate_proj.weight"),
                    "up_w": a(p + "mlp.up_proj.weight"),
                    "down_w": a(p + "mlp.down_proj.weight"),
                },
            }
        )
    params = {
        "embed": a("model.embed_tokens.weight"),
        "layers": layers,
        "norm": a("model.norm.weight"),
    }
    if "lm_head.weight" in t:
        if cfg.tie_word_embeddings:
            raise ValueError("config ties embeddings but checkpoint has lm_head.weight")
        params["lm_head"] = a("lm_head.weight")
    return cfg, params


def empty_cache(cfg: Qwen2Cfg, max_len: int, dtype=jnp.float32) -> dict:
    """Preallocated KV cache for one sequence: (n_layers, n_kv, max_len, head_dim)."""
    return {
        "k": jnp.zeros((cfg.n_layers, cfg.n_kv, max_len, cfg.head_dim), dtype),
        "v": jnp.zeros((cfg.n_layers, cfg.n_kv, max_len, cfg.head_dim), dtype),
        "len": jnp.zeros((), jnp.int32),
    }


def _rmsnorm(x, w, eps):
    xf = x.astype(jnp.float32)
    rms = jnp.power(jnp.square(xf).mean(axis=-1, keepdims=True) + eps, -0.5)
    return (xf * rms * w).astype(x.dtype)


def _rope(pos, cfg):
    """Half-size cos/sin tables for absolute positions: each (T, head_dim/2), f32."""
    half = cfg.head_dim // 2
    inv = cfg.rope_theta ** (-(jnp.arange(half, dtype=jnp.float32) / half))
    f = pos[:, None].astype(jnp.float32) * inv[None, :]
    return jnp.cos(f), jnp.sin(f)


def _apply_rope(x, c, s):
    # x: (T, n_heads, head_dim); c, s: (T, head_dim/2)
    h = x.shape[-1] // 2
    c, s = c[:, None, :], s[:, None, :]
    x1, x2 = x[..., :h], x[..., h:]
    return jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def _mlp(h, m):
    return (jax.nn.silu(h @ m["gate_w"].T) * (h @ m["up_w"].T)) @ m["down_w"].T


@partial(jax.jit, static_argnums=(0,))
def step(cfg, params, cache, ids):
    """Forward over `ids` (T,), appended to the cache at position cache['len'].

    ids: (T,) int. Returns (logits (T, V), new_cache). Prefill = fresh cache;
    decode = 1..T new tokens. New keys/values are written into the preallocated
    cache; attention sees keys m with m <= query position and m < len+T.
    """
    T = ids.shape[0]
    nh, nkv, hd = cfg.n_heads, cfg.n_kv, cfg.head_dim
    x = params["embed"][ids]
    start = cache["len"]
    stop = start + T
    pos = start + jnp.arange(T, dtype=jnp.int32)
    c, s = _rope(pos, cfg)
    max_len = cache["k"].shape[2]
    key_pos = jnp.arange(max_len, dtype=jnp.int32)
    mask = (key_pos[None, :] <= pos[:, None]) & (key_pos[None, :] < stop)  # (T, max_len)
    scale = hd**-0.5
    ks, vs = cache["k"], cache["v"]
    for li, layer in enumerate(params["layers"]):
        a = layer["attn"]
        h = _rmsnorm(x, layer["input_ln"], cfg.norm_eps)
        q = (h @ a["q_w"].T + a["q_b"]).reshape(T, nh, hd)
        k = (h @ a["k_w"].T + a["k_b"]).reshape(T, nkv, hd)
        v = (h @ a["v_w"].T + a["v_b"]).reshape(T, nkv, hd)
        q, k = _apply_rope(q, c, s), _apply_rope(k, c, s)
        # write new keys/values (dynamic slice; T is static, start is traced),
        # and fold the update back in so the returned cache carries it
        ks_i = ks[li].at[:, jax.ds(start, T), :].set(k.transpose(1, 0, 2))
        vs_i = vs[li].at[:, jax.ds(start, T), :].set(v.transpose(1, 0, 2))
        ks = ks.at[li].set(ks_i)
        vs = vs.at[li].set(vs_i)
        # GQA: group query heads (nkv, reps) against one kv head, no repeat copy
        reps = nh // nkv
        qg = q.transpose(1, 0, 2).reshape(nkv, reps, T, hd)
        scores = jnp.einsum("grtd,gmd->grtm", qg, ks_i) * scale
        scores = jnp.where(mask[None, None, :, :], scores, NEG)
        p = jax.nn.softmax(scores, axis=-1).astype(ks_i.dtype)
        o = jnp.einsum("grtm,gmd->grtd", p, vs_i).transpose(2, 0, 1, 3).reshape(T, nh * hd)
        x = x + o @ a["o_w"].T
        x = x + _mlp(_rmsnorm(x, layer["post_ln"], cfg.norm_eps), layer["mlp"])
    x = _rmsnorm(x, params["norm"], cfg.norm_eps)
    w = params["lm_head"] if "lm_head" in params else params["embed"]
    return x @ w.T, {"k": ks, "v": vs, "len": stop}


def prefill(cfg: Qwen2Cfg, params: dict, ids, max_len: int, dtype=jnp.float32):
    """Jit-compiled prefill of a fresh cache. Returns (logits (T, V), cache)."""
    return step(cfg, params, empty_cache(cfg, max_len, dtype), jnp.asarray(ids, jnp.int32))


def decode(cfg: Qwen2Cfg, params: dict, cache: dict, token_ids):
    """Jit-compiled single/multi-token step appending to an existing cache."""
    return step(cfg, params, cache, jnp.asarray(token_ids, jnp.int32))
