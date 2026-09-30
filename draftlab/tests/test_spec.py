"""Speculative sampling: losslessness (T=0 greedy identity, T=1 chi-square),
KV rollback == fresh prefill, and the Drafter protocol (ngram + model)."""

import jax
import numpy as np
import pytest
from scipy import stats

from draftlab.drafters.ngram import NGramMatcher
from draftlab.jaxlm import model as M
from draftlab.jaxlm.spec import ModelDrafter, NGramDrafter, SpecDecoder


def _tiny16(seed: int = 0):
    cfg = M.Qwen2Cfg(
        vocab_size=16, hidden_size=64, n_layers=2, n_heads=4, n_kv=2,
        head_dim=16, intermediate_size=128, rope_theta=10000.0, norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    return cfg, M.init_params(cfg, jax.random.PRNGKey(seed))


def _rand_prompt(key, n, vocab):
    return [int(x) for x in np.asarray(jax.random.randint(key, (n,), 0, vocab))]


def _greedy(cfg, params, prompt, n):
    lg, cache = M.prefill(cfg, params, prompt, len(prompt) + n + 8)
    out = []
    for _ in range(n):
        t = int(np.argmax(np.asarray(lg)[-1]))
        out.append(t)
        lg, cache = M.decode(cfg, params, cache, [t])
    return out


def _make_drafter(kind: str, cfg, max_len):
    if kind == "ngram":
        return NGramDrafter(cfg.vocab_size, max_n=4)
    dcfg, dparams = _tiny16(99)
    return ModelDrafter(dcfg, dparams, max_len)


@pytest.mark.parametrize("kind", ["ngram", "model"])
def test_greedy_identical_to_target(kind):
    cfg, params = _tiny16(0)
    prompt = _rand_prompt(jax.random.PRNGKey(7), 40, cfg.vocab_size)
    ref = _greedy(cfg, params, prompt, 40)
    dec = SpecDecoder(
        cfg, params, _make_drafter(kind, cfg, 88), max_len=88, k=4, temperature=0.0, seed=3
    )
    out, st = dec.generate(prompt, 40)
    assert out == ref
    for s in st:
        assert 0 <= s.accepted <= s.drafted
        assert len(s.per_position) == s.drafted


def test_lossless_chi_square():
    """T=1: 20k first tokens vs the target multinomial, chi2 p > 0.01."""
    cfg, params = _tiny16(1)
    # patterned prompt: the ngram drafter fires on every step, so the
    # accept/reject + residual paths (not just the no-draft path) are covered
    prompt = [i % 7 for i in range(32)]
    lg, _ = M.prefill(cfg, params, prompt, len(prompt) + 8)
    p = np.asarray(jax.nn.softmax(lg[-1]), np.float64)
    N = 20000
    obs = np.zeros(cfg.vocab_size, np.int64)
    for i in range(N):
        dec = SpecDecoder(cfg, params, NGramDrafter(cfg.vocab_size, max_n=4),
                          max_len=len(prompt) + 8, k=2, temperature=1.0, seed=i)
        out, _ = dec.generate(prompt, 1)
        obs[out[0]] += 1
    exp = N * p
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    pval = float(stats.chi2.sf(chi2, df=cfg.vocab_size - 1))
    assert pval > 0.01, f"chi2={chi2:.1f}, p={pval:.4g} (distribution shifted)"


@pytest.mark.parametrize("kind", ["ngram", "model"])
def test_rollback_cache_equals_fresh_prefill(kind):
    cfg, params = _tiny16(2)
    # patterned prompt guarantees ngram/model drafts exist and are often rejected
    prompt = [i % 7 for i in range(40)]
    max_len = len(prompt) + 40 + 8
    dec = SpecDecoder(cfg, params, _make_drafter(kind, cfg, max_len), max_len=max_len,
                      k=3, temperature=1.0, seed=5)
    out, st = dec.generate(prompt, 30)
    assert len(out) == 30
    assert any(s.drafted > 0 and s.accepted < s.drafted for s in st), "need a partial acceptance"
    full = prompt + out
    _, ref_cache = M.prefill(cfg, params, full, max_len)
    c = dec.last_cache
    assert int(c["len"]) == len(full)
    n = len(full)
    for key in ("k", "v"):
        diff = float(abs(np.asarray(c[key])[:, :, :n] - np.asarray(ref_cache[key])[:, :, :n]).max())
        assert diff < 1e-5, f"{key}: max abs diff vs fresh prefill {diff:.2e}"


def test_ngram_drafter_matches_offline():
    """Online proposals equal NGramMatcher.proposals at every position."""
    seq = [int(x) for x in np.asarray(jax.random.randint(jax.random.PRNGKey(13), (60,), 0, 8))]
    P = NGramMatcher(max_n=4).proposals(seq)
    d = NGramDrafter(vocab_size=8, max_n=4)
    for t in range(len(seq)):
        toks, _ = d.propose(seq[:t], 1)
        assert (toks[0] if toks else None) == P[t]


def test_model_drafter_protocol():
    cfg, params = _tiny16(3)
    dcfg, dparams = _tiny16(4)
    prompt = _rand_prompt(jax.random.PRNGKey(31), 24, cfg.vocab_size)
    d = ModelDrafter(dcfg, dparams, 64)
    toks, probs = d.propose(prompt, 3)
    assert len(toks) == 3 and all(0 <= t < cfg.vocab_size for t in toks)
    for q in probs:
        assert q.shape == (cfg.vocab_size,) and abs(float(q.sum()) - 1.0) < 1e-4
    # T=1 run is lossless-shape-correct and T=0 matches the target greedy
    dec = SpecDecoder(
        cfg, params, ModelDrafter(dcfg, dparams, 84), max_len=84, k=3, temperature=1.0, seed=8
    )
    out, st = dec.generate(prompt, 12)
    assert len(out) == 12
    assert _greedy(cfg, params, prompt, 12) == SpecDecoder(
        cfg, params, ModelDrafter(dcfg, dparams, 84), max_len=84, k=3, temperature=0.0, seed=8
    ).generate(prompt, 12)[0]


def test_model_drafter_full_accept_steps_keep_cache_in_sync():
    """Regression: drafter==target at T=0 full-accepts every step (A==m).
    rollback(m) must advance _last by m — otherwise the next propose re-decodes
    already-cached draft tokens and the drafter context diverges from the
    target's, so strict n_acc == n_draft fails."""
    cfg, params = _tiny16(5)
    prompt = _rand_prompt(jax.random.PRNGKey(11), 24, cfg.vocab_size)
    dec = SpecDecoder(cfg, params, ModelDrafter(cfg, params, 84), max_len=84,
                      k=1, temperature=0.0, seed=9)
    out, st = dec.generate(prompt, 20)
    assert out == _greedy(cfg, params, prompt, 20)
    assert all(s.drafted == 1 and s.accepted == 1 for s in st)  # strict, no rejections
    # The drafter ends one token behind the final bonus: its cache must equal a
    # fresh prefill of that prefix (a desynced rollback re-decodes already
    # cached drafts and the KV diverges from the first duplicated slot on).
    full = prompt + out
    n = len(full) - 1
    d = dec.drafter
    assert int(d.cache["len"]) == n and d._last == n
    _, ref = M.prefill(d.dcfg, d.dparams, full[:n], 84)
    for key in ("k", "v"):
        diff = float(abs(np.asarray(d.cache[key])[:, :, :n] - np.asarray(ref[key])[:, :, :n]).max())
        assert diff < 1e-5, f"drafter {key} cache diverged from fresh prefill: {diff:.2e}"
    # ...and its next first draft still equals a fresh forward over the full ctx.
    toks, _ = d.propose(full, 1)
    lg, _ = M.step(cfg, params, M.empty_cache(cfg, 84),
                   jax.numpy.asarray(full, jax.numpy.int32))
    assert toks[0] == int(np.argmax(np.asarray(lg[-1])))


def test_model_drafter_first_proposal_position():
    """Known answer: draft j's dist is the forward over ctx+drafts[:j] (no
    duplicated last token); first draft == argmax of the forward over ctx."""
    cfg, params = _tiny16(4)
    prompt = _rand_prompt(jax.random.PRNGKey(7), 10, cfg.vocab_size)
    d = ModelDrafter(cfg, params, 64)
    toks, probs = d.propose(prompt, 3)
    lg, _ = M.step(cfg, params, M.empty_cache(cfg, 64), jax.numpy.asarray(prompt, jax.numpy.int32))
    assert toks[0] == int(np.argmax(np.asarray(lg[-1])))
    # second draft conditions on ctx + toks[0]
    lg2, _ = M.step(cfg, params, M.empty_cache(cfg, 64),
                    jax.numpy.asarray(prompt + [toks[0]], jax.numpy.int32))
    assert toks[1] == int(np.argmax(np.asarray(lg2[-1])))
