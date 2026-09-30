"""Fused Triton kernels for spec-decode verify and alpha, plus torch references.

verify: one launch, one program. Online-softmax of the target rows [k+1, V] and
draft rows [k, V]; drafted token j is accepted iff p_j[t_j] > u_j * q_j[t_j]
(equivalent to spec.py's u*q < p; 0/0 rejects); outputs the first-reject index
fr and the normalised residual max(0, p - q) of row fr (q = 0 when fr == k,
i.e. the bonus row).
alpha: one launch, one program per row; per-row sum min(softmax p, softmax q).

CPU: TRITON_INTERPRET=1 (tests/test_verify_kernel.py, V=1000 fp32, atol 1e-5).
GPU: bench.py times both against the torch references at V=151936, k=16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _online_upd(mp, sp, x):
    """One online-softmax tile update; skips tiles that are all -inf."""
    m = tl.maximum(mp, tl.max(x, 0))
    if m > float("-inf"):
        sp = sp * tl.exp(mp - m) + tl.sum(tl.exp(x - m))
        mp = m
    return mp, sp


@triton.jit
def _norm(x, m, s):
    """exp(x - m) / s with -inf rows (fully masked) -> 0 without NaN."""
    arg = tl.where(m > float("-inf"), x, float("-inf")) - tl.where(m > float("-inf"), m, 0.0)
    return tl.where(m > float("-inf"), tl.exp(arg) / tl.where(s > 0.0, s, 1.0), 0.0)


@triton.jit
def _verify_kernel(
    P, Q, T, U, FR_OUT, RES_OUT, V: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    fr = K
    for r in range(K):
        mp, sp = float("-inf"), 0.0
        mq, sq = float("-inf"), 0.0
        for i in range(0, V, BLOCK):
            o = i + tl.arange(0, BLOCK)
            msk = o < V
            pt = tl.load(P + r * V + o, mask=msk, other=float("-inf"))
            qt = tl.load(Q + r * V + o, mask=msk, other=float("-inf"))
            mp, sp = _online_upd(mp, sp, pt)
            mq, sq = _online_upd(mq, sq, qt)
        t_r = tl.load(T + r)
        oi = (t_r // BLOCK) * BLOCK + tl.arange(0, BLOCK)
        msk2 = oi < V
        pt = tl.load(P + r * V + oi, mask=msk2, other=float("-inf"))
        qt = tl.load(Q + r * V + oi, mask=msk2, other=float("-inf"))
        u_r = tl.load(U + r)
        pv = tl.sum(tl.where(oi == t_r, _norm(pt, mp, sp), 0.0))
        qv = tl.sum(tl.where(oi == t_r, _norm(qt, mq, sq), 0.0))
        fr = tl.minimum(fr, tl.where(pv > u_r * qv, K, r))
    # Residual: normalised max(0, p - q) at row fr (draft row fr, or 0 for bonus).
    mp, sp = float("-inf"), 0.0
    mq, sq = float("-inf"), 0.0
    for i in range(0, V, BLOCK):
        o = i + tl.arange(0, BLOCK)
        msk = o < V
        pt = tl.load(P + fr * V + o, mask=msk, other=float("-inf"))
        qt = tl.load(Q + fr * V + o, mask=msk & (fr < K), other=float("-inf"))
        mp, sp = _online_upd(mp, sp, pt)
        mq, sq = _online_upd(mq, sq, qt)
    tot = 0.0
    for i in range(0, V, BLOCK):
        o = i + tl.arange(0, BLOCK)
        msk = o < V
        pt = tl.load(P + fr * V + o, mask=msk, other=float("-inf"))
        qt = tl.load(Q + fr * V + o, mask=msk & (fr < K), other=float("-inf"))
        pv, qv = _norm(pt, mp, sp), _norm(qt, mq, sq)
        tot += tl.sum(tl.maximum(0.0, pv - qv))
    inv = 1.0 / tl.where(tot > 1e-30, tot, 1.0)
    for i in range(0, V, BLOCK):
        o = i + tl.arange(0, BLOCK)
        msk = o < V
        pt = tl.load(P + fr * V + o, mask=msk, other=float("-inf"))
        qt = tl.load(Q + fr * V + o, mask=msk & (fr < K), other=float("-inf"))
        pv, qv = _norm(pt, mp, sp), _norm(qt, mq, sq)
        tl.store(RES_OUT + o, tl.maximum(0.0, pv - qv) * inv, mask=msk)
    tl.store(FR_OUT, fr)


@triton.jit
def _alpha_kernel(P, Q, OUT, V: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    mp, sp = float("-inf"), 0.0
    mq, sq = float("-inf"), 0.0
    for i in range(0, V, BLOCK):
        o = i + tl.arange(0, BLOCK)
        msk = o < V
        pt = tl.load(P + r * V + o, mask=msk, other=float("-inf"))
        qt = tl.load(Q + r * V + o, mask=msk, other=float("-inf"))
        mp, sp = _online_upd(mp, sp, pt)
        mq, sq = _online_upd(mq, sq, qt)
    a = 0.0
    for i in range(0, V, BLOCK):
        o = i + tl.arange(0, BLOCK)
        msk = o < V
        pt = tl.load(P + r * V + o, mask=msk, other=float("-inf"))
        qt = tl.load(Q + r * V + o, mask=msk, other=float("-inf"))
        pv, qv = _norm(pt, mp, sp), _norm(qt, mq, sq)
        a += tl.sum(tl.minimum(pv, qv))
    tl.store(OUT + r, a)


def fused_verify(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    tokens: torch.Tensor,
    u: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Fused verify (one launch). p_logits [k+1, V], q_logits [k, V] fp32,
    tokens [k] int, u [k] in [0, 1). Returns (first_reject, residual [V]):
    first_reject == k when all drafts accepted; residual is the normalised
    max(0, p - q) at that row (p_k for the bonus)."""
    p = p_logits.contiguous()
    q = q_logits.contiguous()
    k, v = q.shape
    if p.shape != (k + 1, v) or tokens.numel() != k or u.numel() != k:
        raise ValueError("need p [k+1, V], q [k, V], tokens/u [k]")
    fr_out = torch.empty(1, dtype=torch.int32, device=p.device)
    res = torch.empty(v, dtype=torch.float32, device=p.device)
    _verify_kernel[(1,)](
        p,
        q,
        tokens.to(torch.int32).contiguous(),
        u.float().contiguous(),
        fr_out,
        res,
        V=v,
        K=k,
        BLOCK=_BLOCK,
    )
    return int(fr_out.item()), res


def fused_alpha(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Fused alpha (one launch, one program per row). p, q [N, V] fp32 ->
    per-row sum min(softmax p, softmax q)."""
    p = p_logits.contiguous()
    q = q_logits.contiguous()
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError("need p, q: [N, V]")
    out = torch.empty(p.shape[0], dtype=torch.float32, device=p.device)
    _alpha_kernel[(p.shape[0],)](p, q, out, V=p.shape[1], BLOCK=_BLOCK)
    return out


# ---------------------------------------------------------------- torch refs


def _softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def verify_torch(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    tokens: torch.Tensor,
    u: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Reference verify; same contract as fused_verify."""
    k = q_logits.shape[0]
    p, q = _softmax(p_logits), _softmax(q_logits)
    idx = torch.arange(k, device=p.device)
    acc = p[idx, tokens] > u * q[idx, tokens]
    rej = (~acc).nonzero(as_tuple=True)[0]
    fr = int(rej[0]) if rej.numel() else k
    qr = q[fr] if fr < k else torch.zeros_like(p[fr])
    res = (p[fr] - qr).clamp_min(0.0)
    return fr, res / res.sum().clamp_min(1e-30)


def alpha_torch(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Reference alpha: per-row sum min(softmax p, softmax q)."""
    return torch.minimum(_softmax(p_logits), _softmax(q_logits)).sum(-1)
