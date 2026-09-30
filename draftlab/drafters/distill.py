"""Distilled tiny drafter (JAX): top-64 knowledge distillation from the 0.5B target.

cache: teacher-forced 0.5B (JAX, f32) over train/val blocks; per position store
       the top-64 probabilities + indices -> data/teacher/{split}_{run}.npz.
train: 4-layer d=256 Qwen2-shaped student, same 151936 vocab, tied embedding
       (the teacher's hidden is 896, so "sharing T's embeddings vocab" = same
       vocab, student-own trainable embedding; the weight matrices cannot be
       shared across the dimension mismatch). Optax AdamW + warmup-cosine.
       Loss = KL(T||S) on the 65-way distribution (teacher top-64 probs + the
       renormalised tail mass vs the student's softmax over those 64 indices +
       its tail). Block-only conditioning (no cross-block ctx) to fit the CPU
       budget; the alpha eval still hands the drafter its ctx.
       Orbax checkpoints -> ckpt/tiny/; curve -> results/distill_curve.json;
       HF-layout export -> ckpt/tiny/export/ (what the alpha CLI loads).

CLI:
  python -m draftlab.drafters.distill cache [--tokens 1000000] [--max-tokens 256]
  python -m draftlab.drafters.distill train [--steps 300] [--batch 16] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import draftlab.jaxlm.model as M
from draftlab.data.corpus import ROOT
from draftlab.eval.alpha import TARGET_DIR, load_blocks
from draftlab.jaxlm.model import (
    NEG,
    Qwen2Cfg,
    _apply_rope,
    _mlp,
    _rmsnorm,
    _rope,
)

TEACHER_DIR = ROOT / "data" / "teacher"
CKPT_DIR = ROOT / "ckpt" / "tiny"
CURVE_PATH = ROOT / "results" / "distill_curve.json"
TOP_K = 64

TINY_CFG = Qwen2Cfg(
    vocab_size=151936,
    hidden_size=256,
    n_layers=4,
    n_heads=8,
    n_kv=2,
    head_dim=32,
    intermediate_size=1390,  # 0.5B ratio 4864/896 ~ 5.43
    rope_theta=1000000.0,
    norm_eps=1e-6,
    tie_word_embeddings=True,
)


# --- student forward (batched, top-64 head only) ---------------------------


def student_hidden(cfg: Qwen2Cfg, params: dict, ids: jnp.ndarray, lens: jnp.ndarray) -> jnp.ndarray:
    """Batched prefill. ids (B, L) int, lens (B,) valid length -> h (B, L, D)."""
    B, L = ids.shape
    nh, nkv, hd = cfg.n_heads, cfg.n_kv, cfg.head_dim
    x = params["embed"][ids]
    pos = jnp.arange(L, dtype=jnp.int32)
    c, s = _rope(pos, cfg)
    c, s = jnp.repeat(c, B, 0), jnp.repeat(s, B, 0)  # (B*L, hd/2)
    key_pos = jnp.arange(L, dtype=jnp.int32)
    causal = key_pos[None, :] <= pos[:, None]  # (L, L): key m <= query p
    valid = key_pos[None, :] < lens[:, None]  # (B, L): key m < lens[b]
    mask = causal[None, :, :] & valid[:, :, None]  # (B, L, L)
    scale = hd**-0.5
    for layer in params["layers"]:
        a = layer["attn"]
        h = _rmsnorm(x, layer["input_ln"], cfg.norm_eps)
        q = (h @ a["q_w"].T + a["q_b"]).reshape(B * L, nh, hd)
        k = (h @ a["k_w"].T + a["k_b"]).reshape(B * L, nkv, hd)
        v = (h @ a["v_w"].T + a["v_b"]).reshape(B * L, nkv, hd)
        q, k = _apply_rope(q, c, s), _apply_rope(k, c, s)
        reps = nh // nkv
        qg = q.reshape(B, L, nkv, reps, hd)
        scores = jnp.einsum("blgrd,bngd->blgrn", qg, k.reshape(B, L, nkv, hd)) * scale
        scores = jnp.where(mask[:, :, None, None, :], scores, NEG)
        p = jax.nn.softmax(scores, axis=-1).astype(k.dtype)
        o = jnp.einsum("blgrn,bngd->blgrd", p, v.reshape(B, L, nkv, hd))
        x = x + o.reshape(B, L, nh * hd) @ a["o_w"].T
        x = x + _mlp(_rmsnorm(x, layer["post_ln"], cfg.norm_eps), layer["mlp"])
    return _rmsnorm(x, params["norm"], cfg.norm_eps)


def kl_top64(cfg: Qwen2Cfg, params: dict, batch: dict) -> jnp.ndarray:
    """Mean KL(T||S) over loss positions. Batch: ids (B,L) int, lens (B,) int,
    top64_idx (B,L,K) int, top64_p (B,L,K) float (teacher probs on its top-K)."""
    h = student_hidden(cfg, params, batch["ids"], batch["lens"])
    e = params["embed"][batch["top64_idx"]]  # (B, L, K, D)
    z = jnp.einsum("bld,blkd->blk", h, e)
    s = jax.nn.softmax(z, axis=-1)
    s_tail = jnp.clip(1.0 - s.sum(axis=-1), 1e-6, None)
    p = batch["top64_p"]
    p_tail = jnp.clip(1.0 - p.sum(axis=-1), 0.0, None)
    # 0*log(0) = NaN under jnp.where (no short-circuit): safe numerators
    ps = jnp.where(p > 0.0, p, 1.0)
    kl64 = p * jnp.log(ps / (s + 1e-9))
    pts = jnp.where(p_tail > 0.0, p_tail, 1.0)
    kl_tail = p_tail * jnp.log(pts / s_tail)
    mask = jnp.arange(p.shape[1])[None, :] < (batch["lens"] - 1)[:, None]  # (B, L)
    kl = kl64.sum(axis=-1) + kl_tail
    return (kl * mask).sum() / jnp.maximum(mask.sum(), 1)


# --- teacher cache ----------------------------------------------------------


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x, dtype=np.float32)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def cache_teacher(
    split: str = "train",
    max_tokens: int = 256,
    tokens: int = 1_000_000,
    out_dir: Path = TEACHER_DIR,
    target: Path = TARGET_DIR,
) -> dict:
    """Teacher top-K over the split's blocks (round-robin runs, block-only).
    Writes one npz per run: ids (N,) i32, top64_p (N, K) f32, top64_idx (N, K)
    i32, offsets (n_blocks+1,) i32. Stops after `tokens` block tokens."""
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, params = M.load_safetensors(str(target))
    tok = AutoTokenizer.from_pretrained(target)
    blocks = load_blocks(split, max_tokens, 10**9)
    k = min(TOP_K, cfg.vocab_size)
    per_run: dict[str, dict] = {}
    n_tok = 0
    t0 = time.perf_counter()
    for i, b in enumerate(blocks):
        if n_tok >= tokens:
            break
        ids = tok.encode(b["text"], add_special_tokens=False)
        if not ids:
            continue
        L = ((len(ids) + 31) // 32) * 32  # length buckets -> few jit compiles
        full = ids + [0] * (L - len(ids))
        logits, _ = M.step(cfg, params, M.empty_cache(cfg, L), jnp.asarray(full))
        p = _softmax_rows(np.asarray(logits)[: len(ids)])
        vals, idx = jax.lax.top_k(jnp.asarray(p), k)
        acc = per_run.setdefault(b["run"], {"ids": [], "p": [], "x": [], "off": [0]})
        off = acc["off"][-1]
        acc["ids"].append(np.asarray(ids, np.int32))
        acc["p"].append(np.asarray(vals, np.float32))
        acc["x"].append(np.asarray(idx, np.int32))
        acc["off"].append(off + len(ids))
        n_tok += len(ids)
        if (i + 1) % 25 == 0:
            dt = time.perf_counter() - t0
            print(
                f"  cached {i + 1}/{len(blocks)} blocks, {n_tok} tok, {n_tok / dt:.0f} tok/s",
                flush=True,
            )
    for run, acc in per_run.items():
        np.savez_compressed(
            out_dir / f"{split}_{run}.npz",
            ids=np.concatenate(acc["ids"]),
            top64_p=np.concatenate(acc["p"]),
            top64_idx=np.concatenate(acc["x"]),
            offsets=np.asarray(acc["off"], np.int64),
        )
    print(f"cached {n_tok} tokens -> {len(per_run)} run files in {out_dir}", flush=True)
    return {"tokens": n_tok, "runs": len(per_run), "out_dir": str(out_dir)}


def load_teacher(split: str, out_dir: Path = TEACHER_DIR) -> dict:
    """Concatenated blocks of a split: (start, end) spans plus ids/top64 arrays."""
    files = sorted(out_dir.glob(f"{split}_*.npz"))
    if not files:
        raise SystemExit(f"no teacher cache in {out_dir} for split {split!r} — run `cache`")
    ids, p, x, spans = [], [], [], []
    base = 0
    for f in files:
        z = np.load(f)
        off = z["offsets"]
        ids.append(z["ids"])
        p.append(z["top64_p"])
        x.append(z["top64_idx"])
        spans.extend((base + off[j], base + off[j + 1]) for j in range(len(off) - 1))
        base += len(z["ids"])
    return {
        "ids": np.concatenate(ids),
        "top64_p": np.concatenate(p),
        "top64_idx": np.concatenate(x),
        "spans": spans,
    }


# --- training ---------------------------------------------------------------


def _batch_arrays(data, spans: list[tuple[int, int]], batch: int, max_len: int, k: int) -> dict:
    """Pack `batch` blocks into padded arrays (numpy)."""
    B = len(spans)
    ids_p = np.zeros((B, max_len), np.int32)
    idx_p = np.zeros((B, max_len, k), np.int32)
    p_p = np.zeros((B, max_len, k), np.float32)
    lens = np.zeros(B, np.int32)
    for r, (s, e) in enumerate(spans):
        n = e - s
        lens[r] = n
        ids_p[r, :n] = data["ids"][s:e]
        idx_p[r, :n] = data["top64_idx"][s:e]
        p_p[r, :n] = data["top64_p"][s:e]
    return {
        "ids": jnp.asarray(ids_p),
        "lens": jnp.asarray(lens),
        "top64_idx": jnp.asarray(idx_p),
        "top64_p": jnp.asarray(p_p),
    }


def _project_teacher_embed(e: np.ndarray, hidden: int, seed: int) -> np.ndarray:
    """Random projection of the teacher's (V, D) embedding to (V, hidden) that
    preserves row norms in expectation (columns ~ N(0, 1/hidden))."""
    w = np.random.default_rng(seed + 1).normal(0, 1, (e.shape[1], hidden))
    w /= np.sqrt(hidden)
    return e.astype(np.float32) @ w


def train(
    steps: int = 300,
    batch: int = 16,
    max_len: int = 256,
    seed: int = 0,
    val_tokens: int = 25_000,
    log_every: int = 25,
    peak_lr: float = 1e-3,
    ckpt_dir: Path = CKPT_DIR,
    curve_path: Path = CURVE_PATH,
    teacher_dir: Path = TEACHER_DIR,
    teacher_init: bool = False,
) -> dict:
    ckpt_dir, curve_path = Path(ckpt_dir).resolve(), Path(curve_path).resolve()
    cfg = TINY_CFG
    k = min(TOP_K, cfg.vocab_size)
    rng = jax.random.PRNGKey(seed)
    params = M.init_params(cfg, rng)
    if teacher_init:
        # Project the 0.5B's (V, 896) embeddings to (V, 256) with a seeded
        # random map (norm-preserving in expectation): the student starts from
        # meaningful token geometry instead of a random embedding.
        from safetensors import safe_open

        with safe_open(str(TARGET_DIR / "model.safetensors"), framework="np") as f:
            e = f.get_tensor("model.embed_tokens.weight")
        params["embed"] = jnp.asarray(_project_teacher_embed(e, cfg.hidden_size, seed))
    train_data = load_teacher("train", teacher_dir)
    val_data = load_teacher("val", teacher_dir)
    spans = train_data["spans"]

    # val subset: first val_tokens block tokens
    v_spans, n_v = [], 0
    for s, e in val_data["spans"]:
        if n_v >= val_tokens:
            break
        v_spans.append((s, e))
        n_v += e - s

    import optax
    from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer

    sched = optax.warmup_cosine_decay_schedule(
        0.0, peak_lr, max(1, steps // 10), steps, peak_lr / 100
    )
    opt = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=sched, weight_decay=0.01)
    )
    state = opt.init(params)

    @partial(jax.jit, static_argnums=(0,))
    def loss_and_grads(cfg, params, batch):
        # value_and_grad re-traces with cfg dynamic -> must close over cfg/batch
        return jax.value_and_grad(lambda p: kl_top64(cfg, p, batch))(params)

    @partial(jax.jit, static_argnums=(0,))
    def kl_only(cfg, params, batch):
        return kl_top64(cfg, params, batch)

    ckpt = CheckpointManager(
        str(ckpt_dir),
        checkpointers={"params": PyTreeCheckpointer()},
        options=CheckpointManagerOptions(max_to_keep=2),
    )
    np_rng = np.random.default_rng(seed)
    order, ptr = np_rng.permutation(len(spans)), 0
    val_batch = _batch_arrays(val_data, v_spans, len(v_spans), max_len, k)
    curve = [{"step": 0, "val_kl": float(kl_only(cfg, params, val_batch))}]
    print(f"step 0: val_kl={curve[0]['val_kl']:.4f} ({n_v} val tok)", flush=True)
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        parts, need = [], batch
        while need > 0:  # take the batch across an epoch boundary if needed
            take = min(need, len(order) - ptr)
            parts.append(order[ptr : ptr + take])
            ptr += take
            need -= take
            if need > 0:
                order, ptr = np_rng.permutation(len(spans)), 0
        idx = parts[0] if len(parts) == 1 else np.concatenate(parts)
        b = _batch_arrays(train_data, [spans[i] for i in idx], batch, max_len, k)
        loss, grads = loss_and_grads(cfg, params, b)
        updates, state = opt.update(grads, state, params)
        params = jax.tree.map(lambda p, u: p + u, params, updates)
        if step % log_every == 0 or step == steps:
            v = float(kl_only(cfg, params, val_batch))
            curve.append({"step": step, "train_loss": float(loss), "val_kl": v})
            ckpt.save(step, {"params": params})
            dt = time.perf_counter() - t0
            print(
                f"step {step}/{steps}: loss={float(loss):.4f} val_kl={v:.4f} "
                f"({dt / step:.2f} s/step)",
                flush=True,
            )
    v0, vf = curve[0]["val_kl"], curve[-1]["val_kl"]
    n_params = sum(np.asarray(x).size for x in jax.tree.leaves(params))
    summary = {
        "cfg": {
            "n_layers": cfg.n_layers,
            "hidden": cfg.hidden_size,
            "vocab": cfg.vocab_size,
            "n_params_M": int(n_params) / 1e6,
            "tie_word_embeddings": True,
        },
        "steps": steps,
        "batch": batch,
        "seed": seed,
        "peak_lr": peak_lr,
        "teacher_init": teacher_init,
        "val_tokens": int(n_v),
        "val_kl_step0": v0,
        "val_kl_final": vf,
        "kl_drop_frac": 1.0 - vf / v0,
        "curve": curve,
    }
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(json.dumps(summary, indent=1) + "\n")
    export_hf(params, cfg, ckpt_dir / "export")
    print(
        f"val KL {v0:.4f} -> {vf:.4f} (drop {summary['kl_drop_frac']:.1%}); "
        f"curve {curve_path}; export {ckpt_dir / 'export'}"
    )
    return summary


def export_hf(params: dict, cfg: Qwen2Cfg, d: Path) -> Path:
    """HF-layout config.json + model.safetensors (tied embed, no lm_head) so
    draftlab.eval.alpha --drafter jax can load it via load_safetensors."""
    from safetensors.numpy import save_file

    d.mkdir(parents=True, exist_ok=True)
    t = {
        "model.embed_tokens.weight": np.asarray(params["embed"], np.float32),
        "model.norm.weight": np.asarray(params["norm"], np.float32),
    }
    for i, layer in enumerate(params["layers"]):
        p = f"model.layers.{i}."
        a, m = layer["attn"], layer["mlp"]
        t[p + "input_layernorm.weight"] = np.asarray(layer["input_ln"], np.float32)
        t[p + "self_attn.q_proj.weight"] = np.asarray(a["q_w"], np.float32)
        t[p + "self_attn.q_proj.bias"] = np.asarray(a["q_b"], np.float32)
        t[p + "self_attn.k_proj.weight"] = np.asarray(a["k_w"], np.float32)
        t[p + "self_attn.k_proj.bias"] = np.asarray(a["k_b"], np.float32)
        t[p + "self_attn.v_proj.weight"] = np.asarray(a["v_w"], np.float32)
        t[p + "self_attn.v_proj.bias"] = np.asarray(a["v_b"], np.float32)
        t[p + "self_attn.o_proj.weight"] = np.asarray(a["o_w"], np.float32)
        t[p + "post_attention_layernorm.weight"] = np.asarray(layer["post_ln"], np.float32)
        t[p + "mlp.gate_proj.weight"] = np.asarray(m["gate_w"], np.float32)
        t[p + "mlp.up_proj.weight"] = np.asarray(m["up_w"], np.float32)
        t[p + "mlp.down_proj.weight"] = np.asarray(m["down_w"], np.float32)
    save_file(t, str(d / "model.safetensors"))
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "vocab_size": cfg.vocab_size,
                "hidden_size": cfg.hidden_size,
                "intermediate_size": cfg.intermediate_size,
                "num_hidden_layers": cfg.n_layers,
                "num_attention_heads": cfg.n_heads,
                "num_key_value_heads": cfg.n_kv,
                "head_dim": cfg.head_dim,
                "rope_theta": cfg.rope_theta,
                "rms_norm_eps": cfg.norm_eps,
                "tie_word_embeddings": cfg.tie_word_embeddings,
                "max_position_embeddings": 32768,
                "torch_dtype": "float32",
            }
        )
    )
    return d


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m draftlab.drafters.distill")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_cache = sub.add_parser("cache", help="cache teacher top-64 over a split's blocks")
    p_cache.add_argument("--split", default="train")
    p_cache.add_argument(
        "--tokens", type=int, default=1_000_000, help="block-token budget (GPU: 1M)"
    )
    p_cache.add_argument("--max-tokens", type=int, default=256)
    p_train = sub.add_parser("train", help="train the tiny student on the cache")
    p_train.add_argument("--steps", type=int, default=1200)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--max-len", type=int, default=256)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--val-tokens", type=int, default=25_000)
    p_train.add_argument("--log-every", type=int, default=25)
    p_train.add_argument("--peak-lr", type=float, default=1e-3)
    p_train.add_argument(
        "--teacher-init",
        action="store_true",
        help="init student embedding from a random projection of the 0.5B embedding",
    )
    args = ap.parse_args(argv)
    if args.cmd == "cache":
        cache_teacher(args.split, args.max_tokens, args.tokens)
    else:
        train(
            steps=args.steps,
            batch=args.batch,
            max_len=args.max_len,
            seed=args.seed,
            val_tokens=args.val_tokens,
            log_every=args.log_every,
            peak_lr=args.peak_lr,
            teacher_init=args.teacher_init,
        )


if __name__ == "__main__":
    main()
