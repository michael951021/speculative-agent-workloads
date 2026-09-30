"""Known answers for the alpha metric (CLAUDE.md "Known answers"), on the real
0.5B checkpoint. Skipped only if models/ has not been fetched."""

from pathlib import Path

import numpy as np
import pytest

from draftlab.eval.alpha import JaxModel, NGramDrafter, alpha_and_greedy, load_blocks

MODEL = Path(__file__).resolve().parents[2] / "models" / "qwen2.5-0.5b"
HAS_MODEL = (MODEL / "config.json").exists()
needs_model = pytest.mark.skipif(not HAS_MODEL, reason="models/ not fetched")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def target():
    return JaxModel.from_path(MODEL, max_len=512)


@needs_model
def test_per_position_ce_matches_hf_labels_loss(target, tok):
    """JaxModel rows must be aligned (row i predicts token i+1): per-position CE
    from dists == HF per-token NLL, and their mean == model(ids, labels=ids).loss."""
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    for b in load_blocks("val", max_tokens=256, n_blocks=2):
        ids = torch.tensor([tok.encode(b["text"], add_special_tokens=False)])
        with torch.no_grad():
            out = hf(ids, labels=ids)
            nll = F.cross_entropy(hf(ids).logits[0, :-1], ids[0][1:], reduction="none").numpy()
        p, valid = target.dists([], ids[0].tolist())
        assert not valid[0] and valid[1:].all()
        arr = ids[0].numpy()[1:]  # ce_j = -log p[j, ids[j]] for j >= 1
        ce = -np.log(np.maximum(p.astype(np.float64)[np.arange(1, len(arr) + 1), arr], 1e-30))
        assert np.max(np.abs(ce - nll)) < 1e-3, np.max(np.abs(ce - nll))
        assert abs(float(ce.mean()) - out.loss.item()) < 1e-3


@needs_model
def test_target_top1_on_val_in_sanity_range(target, tok):
    """Qwen2.5-0.5B top-1 next-token accuracy on corpus val blocks is 40-70%
    (57.5% measured on 372 tok); under 20% means an alignment bug."""
    hits = tot = 0
    for b in load_blocks("val", max_tokens=256, n_blocks=40):
        ids = np.asarray(tok.encode(b["text"], add_special_tokens=False))
        p, valid = target.dists([], ids.tolist())
        ok = valid & (p.argmax(axis=-1) == ids)
        hits += int(ok.sum())
        tot += int(valid.sum())
    acc = hits / tot
    assert 0.40 <= acc <= 0.70, f"top-1 {acc:.3f} outside sanity range 0.40-0.70"


@needs_model
def test_uniform_q_alpha_exact(target, tok):
    """q uniform -> alpha = sum_x min(p, 1/V) = 1 - TV(p, u) = 1 - 0.5*sum|p - 1/V|
    (checked via the independent total-variation identity, not the same min-sum)."""
    ids = np.asarray(tok.encode(load_blocks("val", 256, 1)[0]["text"], add_special_tokens=False))
    p, valid = target.dists([], ids.tolist())
    v = p.shape[1]
    a, _ = alpha_and_greedy(p, np.full_like(p, 1.0 / v))
    expected = 1.0 - 0.5 * np.abs(p.astype(np.float64) - 1.0 / v).sum(axis=-1)
    assert np.max(np.abs(a - expected)) < 1e-6
    assert (a >= 1.0 / v - 1e-6).all() and (a <= 1.0).all()  # TV bounds


def test_ngram_repeated_block_planted_effect():
    """Planted effect: a block repeated verbatim after itself -> ngram acceptance
    on the copy ~= 1 (all but the first position, which has no prior context)."""
    x = list(range(1, 65))  # no internal n-gram repeats, so every proposal is exact
    d = NGramDrafter(vocab_size=1000)
    q, valid = d.dists(x, x)
    match = valid & (q.argmax(axis=-1) == np.asarray(x))
    assert match.sum() == len(x) - 1 and match[1:].all()


def test_ngram_shuffled_block_null_control():
    """Null control: the same tokens shuffled within the block -> ngram acceptance
    collapses (proposals were learned on the original order)."""
    x = list(range(1, 65))
    rng = np.random.default_rng(0)
    y = [x[i] for i in rng.permutation(len(x))]
    d = NGramDrafter(vocab_size=1000)
    q, valid = d.dists(x, y)
    acc = float((valid & (q.argmax(axis=-1) == np.asarray(y))).mean())
    assert acc < 0.5, f"shuffled acceptance {acc:.3f} did not collapse"
