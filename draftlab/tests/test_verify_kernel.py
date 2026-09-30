"""Triton verify/alpha kernels vs torch reference.

CPU sandbox: TRITON_INTERPRET=1 (read lazily by triton knobs at launch), V=1000,
random fp32 logits, atol 1e-5. Known answers: identical rows -> all accepted and
alpha == 1; uniform q -> alpha == sum(min(p, 1/V)).
"""

import os

os.environ["TRITON_INTERPRET"] = "1"

import numpy as np  # noqa: F401
import pytest
import torch

pytest.importorskip("triton")
from draftlab.kernels import verify  # noqa: E402

V = 1000
ATOL = 1e-5


def _rand(rows: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, V, generator=g) * scale


def _tokens(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (8,), generator=g, dtype=torch.int32)


def _uniforms(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(8, generator=g)


def test_verify_matches_torch():
    p, q, t, u = _rand(9, 0), _rand(8, 1), _tokens(2), _uniforms(3)
    fr, res = verify.fused_verify(p, q, t, u)
    fr_ref, res_ref = verify.verify_torch(p, q, t, u)
    assert fr == fr_ref
    assert res.shape == (V,)
    torch.testing.assert_close(res, res_ref, atol=ATOL, rtol=0)
    assert abs(res.sum().item() - 1.0) < 1e-4


def test_verify_identical_rows_all_accepted():
    base = _rand(4, 0)
    p = torch.cat([base, _rand(1, 9)])
    t, u = _tokens(2)[:4], torch.full((4,), 0.99)
    fr, res = verify.fused_verify(p, base, t, u)
    assert fr == 4  # p == q: p[t] > 0.99 * q[t]
    ref = torch.softmax(p[4], -1)  # bonus row
    torch.testing.assert_close(res, ref, atol=ATOL, rtol=0)


def test_verify_forced_reject_at_zero():
    p, q, t, u = _rand(7, 0), _rand(6, 1), _tokens(2)[:6], torch.full((6,), 0.5)
    p[0, t[0]] = -20.0
    q[0, t[0]] = 20.0
    fr, res = verify.fused_verify(p, q, t, u)
    assert fr == 0
    fr_ref, res_ref = verify.verify_torch(p, q, t, u)
    assert fr_ref == 0
    torch.testing.assert_close(res, res_ref, atol=ATOL, rtol=0)


def test_verify_u_zero_matches_torch():
    p, q, t = _rand(4, 0), _rand(3, 1), _tokens(2)[:3]
    u = torch.zeros(3)
    fr, res = verify.fused_verify(p, q, t, u)
    fr_ref, res_ref = verify.verify_torch(p, q, t, u)
    assert fr == fr_ref
    torch.testing.assert_close(res, res_ref, atol=ATOL, rtol=0)


def test_alpha_matches_torch():
    p, q = _rand(5, 0), _rand(5, 1)
    torch.testing.assert_close(
        verify.fused_alpha(p, q), verify.alpha_torch(p, q), atol=ATOL, rtol=0
    )


def test_alpha_identical_is_one():
    p = _rand(3, 0)
    torch.testing.assert_close(verify.fused_alpha(p, p), torch.full((3,), 1.0), atol=ATOL, rtol=0)


def test_alpha_uniform_q_analytic():
    p = _rand(4, 0, scale=2.0)
    q = torch.zeros(4, V)  # uniform draft
    a = verify.fused_alpha(p, q)
    ref = torch.minimum(torch.softmax(p, -1), torch.full_like(p, 1.0 / V)).sum(-1)
    torch.testing.assert_close(a, ref, atol=ATOL, rtol=0)
