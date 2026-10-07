"""Alpha metric: hand-computed 3-vocab case, identical-model alpha = 1, k tables,
ngram one-hot adapter."""

import jax
import numpy as np
import pytest

from speculative_agent_workloads.eval.alpha import (
    JaxModel,
    NGramDrafter,
    alpha_and_greedy,
    expected_tokens_per_call,
    kind_label,
)


def test_alpha_hand_3vocab():
    p = np.array([[0.5, 0.3, 0.2]])
    q = np.array([[0.3, 0.5, 0.2]])
    a, g = alpha_and_greedy(p, q)
    assert a[0] == pytest.approx(0.8)  # min = (0.3, 0.3, 0.2)
    assert not g[0]  # argmax p = 0, argmax q = 1
    # identical distributions -> alpha 1, greedy match
    a2, g2 = alpha_and_greedy(p, p)
    assert a2[0] == pytest.approx(1.0) and g2[0]
    # uniform q -> alpha = sum_x min(p(x), 1/V) = 1/3 + 0.3 + 0.2 (hand-computed)
    a3, _ = alpha_and_greedy(p, np.full_like(p, 1.0 / 3))
    assert a3[0] == pytest.approx(1.0 / 3 + 0.3 + 0.2)


def test_valid_mask_zeroes_unproposed_positions():
    p = np.array([[0.9, 0.1], [0.5, 0.5]])
    q = np.array([[0.0, 0.0], [0.8, 0.2]])
    a, g = alpha_and_greedy(p, q, valid=np.array([False, True]))
    assert a[0] == 0.0 and not g[0]
    assert a[1] == pytest.approx(0.7) and g[1]  # min = (0.5, 0.2); argmax p = argmax q = 0


def test_expected_tokens_per_call():
    assert expected_tokens_per_call(0.5, 1) == 1.5
    assert expected_tokens_per_call(0.5, 2) == 1.75
    assert expected_tokens_per_call(0.0, 5) == 1.0
    assert expected_tokens_per_call(1.0, 8) == 9.0


def test_kind_label():
    assert kind_label("tool_use", "Bash") == "tool_use:Bash"
    assert kind_label("thinking", None) == "thinking"
    assert kind_label("tool_use", None) == "tool_use"


def test_ngram_drafter_one_hot():
    d = NGramDrafter(vocab_size=8)
    ids = [1, 2, 3, 2, 3, 5]
    q, valid = d.dists([], ids)
    # state does not leak between calls (fresh matcher each call)
    q2, valid2 = d.dists([], ids)
    assert (q == q2).all() and (valid == valid2).all()
    # context rows are consumed, not scored: with a prefix that matches, the
    # first block position can be proposed from the context
    q3, valid3 = d.dists([1, 2, 3, 2], ids)
    assert valid3[0] and q3[0, 3] == 1.0  # bigram (2,3) seen in ctx -> proposes 3
    assert valid.tolist() == [False, False, False, False, True, True]
    assert q[4, 3] == 1.0 and q[5, 2] == 1.0
    assert q.sum() == 2.0  # exactly one hot entry per proposed position
    # alpha of a one-hot draft = p(proposed token), 0 where unproposed
    p = np.full((6, 8), 0.01)
    p[:, 0] = 0.92
    p[4, 3] = 0.7
    p[5, 2] = 0.4
    p = p / p.sum(axis=-1, keepdims=True)
    a, g = alpha_and_greedy(p, q, valid)
    assert a[4] == np.float64(p[4, 3])
    assert a[5] == np.float64(p[5, 2])
    assert (a[:4] == 0).all() and (g[4] == (p.argmax(-1)[4] == 3))


def test_identical_models_alpha_one():
    from speculative_agent_workloads.jaxlm import model as M
    from speculative_agent_workloads.models import tiny_config

    cfg = M.Qwen2Cfg.from_hf(tiny_config())
    params = M.init_params(cfg, jax.random.PRNGKey(0))
    t = JaxModel(cfg, params, max_len=32)
    d = JaxModel(cfg, params, max_len=32)
    ids = list(range(10, 26))
    p, _ = t.dists([], ids)
    q, valid = d.dists([], ids)
    a, g = alpha_and_greedy(p, q, valid)
    # empty ctx: no row predicts ids[0] -> position 0 invalid on both sides
    assert not valid[0] and valid[1:].all()
    assert a[1:].min() >= 1.0 - 1e-6, f"identical models must score alpha 1, got min {a[1:].min()}"
    assert g[1:].all()
    # non-empty ctx: every position scored, still alpha 1
    ctx = list(range(2, 9))
    p2, _ = t.dists(ctx, ids)
    q2, valid2 = d.dists(ctx, ids)
    a2, g2 = alpha_and_greedy(p2, q2, valid2)
    assert valid2.all()
    assert a2.min() >= 1.0 - 1e-6 and g2.all()


def test_bootstrap_resamples_runs():
    """Regression: weights were (n_runs, n_runs) with n_boot draws per row, so
    every 'replicate' was ~equal weights and CIs collapsed to the point
    estimate. Two runs at alpha 0.2 / 0.8: n_boot replicates, CI spans both."""
    from speculative_agent_workloads.eval.alpha import _bootstrap

    def rec(run, a):
        x = np.full(100, a)
        return (run, "text", x, x > 0.5, x > 0.5, np.ones(100, bool))

    out = _bootstrap([rec("r1", 0.2), rec("r2", 0.8)], n_boot=2000, seed=0)
    b_alpha = out["overall"][0]
    assert b_alpha.shape == (2000,)
    lo, hi = np.percentile(b_alpha, [2.5, 97.5])
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.8)
    assert out["overall"][5].shape == (2000,)  # alpha over valid positions, per replicate
    assert np.mean(out["overall"][5]) == pytest.approx(0.5, abs=0.02)
