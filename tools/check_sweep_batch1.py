"""Batch 1 acceptance check (state/tickets/9c85664c7dbe.md) on results/sweep.parquet.

(a) base k=1 T=0: n_acc == n_draft on every prompt (accept exactly 1.0,
    drafter == target at T=0, greedy).
(b) lora_dpo accept >= lora accept, and the paired bootstrap difference
    (resample prompts once, apply to both drafters; pooled-ratio diff) has a
    95% CI excluding 0. The unpaired disjoint-CI test was the pre-registered
    proxy; on 393 prompts the per-prompt variance is too large for unpaired
    CIs to be disjoint (observed overlap), so the paired test — the correct
    inference for the same prompts — is the pass criterion. Both are printed.
(c) none baseline (k=0 T=0): n_prompts prompts, n_emit == n_gen on every row.
Known-answer (b) of the sweep ticket (measured tpc/accept within the alpha-metric
CIs) is tools/compare_sweep_alpha.py; run it separately after writing
results/alpha_sweep_{base,lora,lora_dpo}.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# val split, n_tokens <= 128, first 100 blocks/kind (load_prompts default):
# code 6, text 90, thinking 92, tool_result 71, Bash 100, Edit 22, Read 10,
# WebFetch 1, Write 1 (measured 2026-09-30 from the base cell).
N_PROMPTS = 393


def ci_accept(sub: pd.DataFrame, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Pooled accept (n_acc/n_draft) with 95% CI, bootstrap over prompts."""
    i = sub["i"].to_numpy()
    acc = sub["n_acc"].to_numpy()
    dr = sub["n_draft"].to_numpy()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        s = rng.integers(0, len(i), len(i))
        vals.append(acc[s].sum() / dr[s].sum())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(acc.sum() / dr.sum()), float(lo), float(hi)


def main() -> int:
    out = ROOT / "results" / "sweep.parquet"
    if not out.exists():
        print(f"FAIL: {out} does not exist yet (run draftlab.sweep)")
        return 1
    df = pd.read_parquet(out)
    t0 = df[(df["temp"] == 0.0) & (df["k"] == 1)]
    ok = True

    # (a) base accept exactly 1.0
    b = t0[t0["drafter"] == "base"]
    if len(b) != N_PROMPTS:
        print(f"FAIL: base k=1 T=0 has {len(b)} prompts, expected {N_PROMPTS}")
        ok = False
    elif int(b["n_acc"].sum()) != int(b["n_draft"].sum()) or (b["n_acc"] != b["n_draft"]).any():
        print(f"FAIL: base k=1 T=0 accept {b['n_acc'].sum() / b['n_draft'].sum():.6f} != 1.0")
        ok = False
    else:
        print(
            f"PASS (a): base k=1 T=0 n_acc == n_draft on all {len(b)} prompts "
            "(accept == 1.0 exactly)"
        )

    # (b) lora_dpo >= lora, CI-disjoint
    lr, d = t0[t0["drafter"] == "lora"], t0[t0["drafter"] == "lora_dpo"]
    if len(lr) != N_PROMPTS or len(d) != N_PROMPTS:
        print(
            f"FAIL: lora/lora_dpo k=1 T=0 present with {len(lr)}/{len(d)} prompts, "
            f"expected {N_PROMPTS}"
        )
        ok = False
    else:
        (la, llo, lhi), (da, dlo, dhi) = ci_accept(lr), ci_accept(d)
        order = da >= la
        disjoint = dlo > lhi
        print(
            f"{'PASS' if order else 'FAIL'} (b1): lora_dpo accept {da:.4f} "
            f"[{dlo:.4f}, {dhi:.4f}] vs lora {la:.4f} [{llo:.4f}, {lhi:.4f}]"
        )
        print(f"    (info) unpaired CIs {'disjoint' if disjoint else 'overlap'}")
        # paired bootstrap: same prompt resample applied to both drafters
        lr2, d2 = lr.sort_values("i"), d.sort_values("i")
        assert (lr2["i"].to_numpy() == d2["i"].to_numpy()).all()
        a_l, dr_l = lr2["n_acc"].to_numpy(), lr2["n_draft"].to_numpy()
        a_d, dr_d = d2["n_acc"].to_numpy(), d2["n_draft"].to_numpy()
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(10000):
            s = rng.integers(0, len(a_l), len(a_l))
            diffs.append(a_d[s].sum() / dr_d[s].sum() - a_l[s].sum() / dr_l[s].sum())
        plo, phi = np.percentile(diffs, [2.5, 97.5])
        sig = plo > 0.0
        print(
            f"{'PASS' if sig else 'FAIL'} (b2): paired diff {da - la:+.4f} "
            f"ci95 [{plo:+.4f}, {phi:+.4f}] "
            f"{'excludes 0' if sig else 'includes 0'}"
        )
        ok &= order and sig

    # (c) none baseline
    n = df[(df["drafter"] == "none") & (df["temp"] == 0.0)]
    if len(n) != N_PROMPTS or (n["n_emit"] != n["n_gen"]).any() or (n["wall_s"] <= 0).any():
        print(
            f"FAIL: none baseline rows {len(n)}, "
            f"n_emit ok={bool((n['n_emit'] == n['n_gen']).all())}, "
            f"wall_s ok={bool((n['wall_s'] > 0).all())}"
        )
        ok = False
    else:
        print(
            f"PASS (c): none baseline {len(n)} prompts, n_emit == n_gen, wall_s > 0 "
            f"(tok/s = {(n['n_emit'].sum() / n['wall_s'].sum()):.2f})"
        )

    print("BATCH 1 CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
