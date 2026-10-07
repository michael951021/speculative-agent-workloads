"""distill: KL formula, student-vs-step parity, synthetic train, npz sanity."""

import glob

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from speculative_agent_workloads.drafters import distill
from speculative_agent_workloads.jaxlm import model as M

TINY = M.Qwen2Cfg(
    vocab_size=16,
    hidden_size=8,
    n_layers=1,
    n_heads=2,
    n_kv=1,
    head_dim=4,
    intermediate_size=16,
    rope_theta=10000.0,
    norm_eps=1e-6,
    tie_word_embeddings=True,
)


def _batch(ids, lens, p, idx):
    return {
        "ids": jnp.asarray(ids, jnp.int32),
        "lens": jnp.asarray(lens, jnp.int32),
        "top64_idx": jnp.asarray(idx, jnp.int32),
        "top64_p": jnp.asarray(p, "float32"),
    }


def test_kl_zero_when_student_matches_teacher():
    rng = jax.random.PRNGKey(0)
    params = M.init_params(TINY, rng)
    ids, lens = jnp.asarray([[3, 7]]), jnp.asarray([2])  # only t=0 scored (causal)
    p0 = jax.nn.softmax(jax.random.normal(rng, (16,)))  # full-vocab teacher probs
    idx = jnp.repeat(jnp.arange(16)[None, None, :], 2, 1)  # (1, 2, 16); row 1 masked
    p = np.zeros((1, 2, 16), np.float32)
    p[0, 0] = np.asarray(p0)
    batch = _batch(ids, lens, p, idx)
    # pin the input token's embed row v; h0 then depends only on v, so the
    # other rows can be solved for afterwards (causal: t=0 sees no other row)
    params = {**params, "embed": params["embed"].at[3].set(jax.random.normal(rng, (8,)))}
    v = params["embed"][3]
    assert float(distill.kl_top64(TINY, params, batch)) > 1e-3  # random init != teacher
    h0 = distill.student_hidden(TINY, params, ids, lens)[0, 0]
    m, logp = float(jnp.dot(h0, h0)), np.log(np.asarray(p0)).astype(np.float64)
    c = float(np.dot(np.asarray(h0), np.asarray(v))) - logp[3]  # z = log p0 + c
    h64 = h0.astype(np.float64)
    e = np.stack([h64 * (logp[j] + c) / m if j != 3 else np.asarray(v) for j in range(16)])
    out = distill.kl_top64(TINY, {**params, "embed": jnp.asarray(e, jnp.float32)}, batch)
    assert float(out) == pytest.approx(0.0, abs=1e-5)


def test_student_hidden_parity_with_step():
    cfg = M.Qwen2Cfg(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv=2,
        head_dim=8,
        intermediate_size=64,
        rope_theta=10000.0,
        norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    params = M.init_params(cfg, jax.random.PRNGKey(1))
    ids = jnp.asarray([5, 11, 3, 42, 7, 2, 50], jnp.int32)
    Lp = 16
    ref, _ = M.step(cfg, params, M.empty_cache(cfg, Lp), ids)
    pad = jnp.concatenate([ids, jnp.zeros(Lp - ids.shape[0], jnp.int32)])
    h = distill.student_hidden(cfg, params, pad[None, :], jnp.asarray([ids.shape[0]]))
    logits_s = h[0] @ params["embed"].T
    n = ids.shape[0]
    np.testing.assert_allclose(np.asarray(ref)[:n], np.asarray(logits_s)[:n], rtol=1e-4, atol=1e-5)


def _fake_npz(path, n_blocks, k, rng):
    """Blocks with a fixed per-row target (memorisable) for the train loop."""
    ids, p, x, off = [], [], [], [0]
    for _ in range(n_blocks):
        L = int(rng.integers(64, 128))
        ids.append(rng.integers(0, distill.TINY_CFG.vocab_size, L))
        V = distill.TINY_CFG.vocab_size
        idx = np.stack([rng.choice(V, k, replace=False) for _ in range(L)])
        rows = np.zeros((L, k), np.float32)
        rows[:, 0], rows[:, 1] = 0.9, 0.05
        rows *= 0.995 / rows.sum(axis=1, keepdims=True)
        p.append(rows.astype(np.float32))
        x.append(idx.astype(np.int32))
        off.append(off[-1] + L)
    np.savez_compressed(
        str(path),
        ids=np.concatenate(ids).astype(np.int32),
        top64_p=np.concatenate(p),
        top64_idx=np.concatenate(x),
        offsets=np.asarray(off, np.int64),
    )


def test_synthetic_train_reduces_val_kl(tmp_path):
    import shutil

    tdir = tmp_path / "teacher"
    tdir.mkdir()
    src = tmp_path / "fake.npz"  # val reuses train data on purpose (leak = memorisation)
    _fake_npz(src, 8, distill.TOP_K, np.random.default_rng(0))
    shutil.copy(src, tdir / "train_0.npz")
    shutil.copy(src, tdir / "val_0.npz")
    s = distill.train(
        steps=50,
        batch=4,
        max_len=128,
        seed=0,
        val_tokens=10**9,
        log_every=10,
        peak_lr=1e-3,
        ckpt_dir=tmp_path / "ckpt",
        curve_path=tmp_path / "curve.json",
        teacher_dir=tdir,
    )
    assert s["val_kl_final"] < s["val_kl_step0"]
    assert s["kl_drop_frac"] > 0.0
    curve = [c for c in s["curve"] if "train_loss" in c]
    assert curve[-1]["train_loss"] < curve[0]["train_loss"]
    exp = tmp_path / "ckpt" / "export"
    assert (exp / "model.safetensors").exists()
    assert (exp / "config.json").exists()


def test_teacher_npz_sanity():
    files = sorted(glob.glob(str(distill.TEACHER_DIR / "train_*.npz")))
    if not files:
        pytest.skip("teacher cache not built")
    d = np.load(files[0])
    p, x, off = d["top64_p"], d["top64_idx"], d["offsets"]
    assert p.shape == x.shape and p.shape[0] == off[-1] and x.shape[-1] <= distill.TOP_K
    for i in range(len(off) - 1):  # per block
        blk = slice(off[i], off[i + 1])
        assert np.all(np.diff(p[blk], axis=1) <= 1e-6)  # rows sorted desc
        assert np.all(p[blk].sum(axis=1) <= 1.0 + 1e-4)
        assert all(len(np.unique(x[r])) == x.shape[1] for r in range(off[i], off[i + 1]))
    assert np.all(np.diff(off) > 0)


def test_project_teacher_embed():
    e = np.random.default_rng(0).normal(size=(128, 896)).astype(np.float32)
    out = distill._project_teacher_embed(e, 256, seed=0)
    assert out.shape == (128, 256)
    assert np.array_equal(out, distill._project_teacher_embed(e, 256, seed=0))
    assert not np.array_equal(out, distill._project_teacher_embed(e, 256, seed=1))
    rel = np.linalg.norm(out, axis=1) / np.linalg.norm(e, axis=1)
    assert np.all((rel > 0.6) & (rel < 1.4))  # row norms preserved in expectation
