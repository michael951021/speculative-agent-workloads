"""Batch 2 acceptance check (state/tickets/bf03e6c66b37.md) on results/sweep.parquet.

(a) completeness: T=0 cells ngram k=1 and tiny/base/lora/lora_dpo k in {1,2,4,8},
    393 prompts each.
(b) pooled tpc(k) = sum(n_emit)/sum(n_steps) non-decreasing in k per model drafter.
(c) known answer, same text + same trajectory: at T=0 every emitted token is a
    target-greedy token, so for ANY k the spec trajectory equals the cached none
    trajectory (spec.py: accept iff draft == argmax p).
    - ngram k=1: replay the engine loop without the target (NGramDrafter
      proposals vs cached trajectory); bootstrap CI over prompts, per-prompt
      accept rate and per-prompt tpc.
    - lora k=8: teacher-force HFDrafter(ckpt/lora) on prompt+trajectory; with
      empty ctx, dists row L+s is the drafter's distribution given
      prompt+traj[:s], so argmax runs over rows reproduce the engine's greedy
      accept runs; bootstrap CI over per-prompt tpc.
    Measured overall accept and tpc (ngram k=1) and tpc (lora k=8) must fall
    inside the CIs.
(d) per-kind tok/s vs none table -> results/sweep_batch2_table.json (report
    table; always written, not gating).

Pre-registered ranges (before the run): lora k=8 tpc in [2.2, 3.0] (corpus
metric alpha_sweep_lora.json tpc_valid_ci95 k=8 = [2.626, 2.648]); ngram k=1
tpc in [1.15, 1.7] (corpus ngram greedy_valid 0.3681 -> tpc@1 ~ 1.37).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "ckpt"
RESULTS = ROOT / "results"
N_GEN = 16
N_PROMPTS = 393
KS = [1, 2, 4, 8]
MODELS = ["tiny", "base", "lora", "lora_dpo"]
VOCAB = 151936


def boot_ci(vals: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    b = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return tuple(float(x) for x in np.percentile(b, [2.5, 97.5]))


def sim_ngram(prompts: list[dict], traj: dict[int, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-prompt (accept_rate, tpc) replaying the T=0 k=1 engine loop."""
    from draftlab.jaxlm.spec import NGramDrafter

    accs: list[float] = []
    tpc: list[float] = []
    for p in prompts:
        t = np.asarray(traj[p["i"]])
        assert len(t) == N_GEN, f"traj for i={p['i']} is {len(t)} tokens"
        drafter = NGramDrafter(VOCAB)
        ctx, s, steps, acc, drafts = list(p["ids"]), 0, 0, 0, 0
        while s < N_GEN:
            dtoks, _ = drafter.propose(ctx, 1)
            steps += 1
            a = int(bool(dtoks) and dtoks[0] == t[s])
            drafts += bool(dtoks)
            acc += a
            s += a + 1
            ctx.extend(t[s - a - 1 : s])
        accs.append(acc / drafts if drafts else np.nan)
        tpc.append(N_GEN / steps)
    return np.asarray(accs, float), np.asarray(tpc, float)


def sim_lora_k8(prompts: list[dict], traj: dict[int, list[int]]) -> np.ndarray:
    """Per-prompt tpc for k=8 greedy acceptance, from HFDrafter argmax rows."""
    from draftlab.eval.alpha import HFDrafter

    drafter = HFDrafter(CKPT / "lora")
    tpc: list[float] = []
    for n, p in enumerate(prompts):
        t = np.asarray(traj[p["i"]])
        q, valid = drafter.dists([], p["ids"] + t.tolist())
        L = len(p["ids"])
        v = (q[L : L + N_GEN].argmax(axis=-1) == t) & valid[L : L + N_GEN]
        s = steps = 0
        while s < N_GEN:
            a = 0
            while a < 8 and s + a < N_GEN and v[s + a]:
                a += 1
            steps += 1
            s += a + 1
        tpc.append(N_GEN / steps)
        if (n + 1) % 50 == 0:
            print(f"  lora sim {n + 1}/{len(prompts)}", flush=True)
    return np.asarray(tpc, float)


def main() -> int:
    out = RESULTS / "sweep.parquet"
    if not out.exists():
        print(f"FAIL: {out} missing (run draftlab.sweep)")
        return 1
    df = pd.read_parquet(out)
    ok = True

    # (a) completeness
    missing = []
    for d, ks in [("ngram", [1])] + [(d, KS) for d in MODELS]:
        for k in ks:
            s = df[(df["drafter"] == d) & (df["k"] == k) & (df["temp"] == 0.0)]
            if len(s) != N_PROMPTS:
                missing.append(f"{d} k={k}: {len(s)}/{N_PROMPTS}")
    if missing:
        ok = False
        print("FAIL (a) missing/partial cells: " + "; ".join(missing))
    else:
        print(f"PASS (a): all 16 T=0 cells present, {N_PROMPTS} prompts each")

    # (b) tpc monotone in k
    print(f"{'drafter':10s} " + " ".join(f"{'k=' + str(k):>8s}" for k in KS))
    for d in MODELS:
        vals = []
        for k in KS:
            s = df[(df["drafter"] == d) & (df["k"] == k) & (df["temp"] == 0.0)]
            vals.append(float(s["n_emit"].sum() / s["n_steps"].sum()) if len(s) else np.nan)
        mono = all(not (np.isnan(vals[i + 1]) or np.isnan(vals[i]) or vals[i + 1] < vals[i])
                   for i in range(len(KS) - 1))
        ok &= mono
        print(f"{d:10s} " + " ".join(f"{v:8.3f}" for v in vals)
              + ("  PASS" if mono else "  FAIL (decreasing)"))

    # (c) known answer on the engine trajectory
    from draftlab.sweep import load_prompts

    prompts = load_prompts()
    cache = RESULTS / "engine_traj_none.json"
    traj = {int(k): v for k, v in json.loads(cache.read_text()).items()}
    assert [p["i"] for p in prompts] == sorted(traj), "prompt set != trajectory set"

    print("\nknown answer (c): measured vs same-trajectory metric CI")
    # ngram k=1
    ng = df[(df["drafter"] == "ngram") & (df["k"] == 1) & (df["temp"] == 0.0)]
    if len(ng) == N_PROMPTS:
        acc_p, tpc_p = sim_ngram(prompts, traj)
        m_acc = float(ng["n_acc"].sum() / ng["n_draft"].sum())
        m_tpc = float(ng["n_emit"].sum() / ng["n_steps"].sum())
        a_lo, a_hi = boot_ci(acc_p[~np.isnan(acc_p)])
        t_lo, t_hi = boot_ci(tpc_p)
        in_a = a_lo <= m_acc <= a_hi
        in_t = t_lo <= m_tpc <= t_hi
        ok &= in_a and in_t
        print(f"  ngram k=1: meas acc {m_acc:.4f} vs traj acc ci95 [{a_lo:.4f}, {a_hi:.4f}] "
              f"{'PASS' if in_a else 'FAIL'}; meas tpc {m_tpc:.4f} vs [{t_lo:.4f}, {t_hi:.4f}] "
              f"{'PASS' if in_t else 'FAIL'} (pre-registered tpc range [1.15, 1.7])")
    else:
        ok = False
        print(f"  ngram k=1: MISSING ({len(ng)}/{N_PROMPTS})")
    # lora k=8
    lr = df[(df["drafter"] == "lora") & (df["k"] == 8) & (df["temp"] == 0.0)]
    if len(lr) == N_PROMPTS:
        tpc_p = sim_lora_k8(prompts, traj)
        m_tpc = float(lr["n_emit"].sum() / lr["n_steps"].sum())
        t_lo, t_hi = boot_ci(tpc_p)
        in_t = t_lo <= m_tpc <= t_hi
        ok &= in_t
        corpus = json.loads((RESULTS / "alpha_sweep_lora.json").read_text())
        c_lo, c_hi = corpus["overall"]["tpc_valid_ci95"]["8"]
        print(f"  lora k=8: meas tpc {m_tpc:.4f} vs traj tpc ci95 [{t_lo:.4f}, {t_hi:.4f}] "
              f"{'PASS' if in_t else 'FAIL'}; corpus-metric ci95 [{c_lo:.3f}, {c_hi:.3f}] "
              f"(info); pre-registered range [2.2, 3.0]")
    else:
        ok = False
        print(f"  lora k=8: MISSING ({len(lr)}/{N_PROMPTS})")

    # (d) report table: tok/s per kind vs none
    none = df[(df["drafter"] == "none") & (df["temp"] == 0.0)]
    none_kind = none.groupby("kind").agg(e=("n_emit", "sum"), w=("wall_s", "sum"))
    none_kind["tok_s"] = none_kind["e"] / none_kind["w"]
    none_tok_s = float(none["n_emit"].sum() / none["wall_s"].sum())
    table = {
        "metric": "tok/s = sum(n_emit)/sum(wall_s); tpc = sum(n_emit)/sum(n_steps); "
        "speedup = tok_s / none tok_s (same kind, T=0, n_gen=16, 393 val prompts)",
        "target": str(df["target"].iloc[0]),
        "none_tok_s": none_tok_s,
        "cells": {},
    }
    print(f"\nreport table (pooled): none = {none_tok_s:.2f} tok/s")
    print(f"{'cell':16s} {'tpc':>6s} {'tok/s':>7s} {'x none':>7s}")
    for d in ["ngram"] + MODELS:
        for k in [1] if d == "ngram" else KS:
            s = df[(df["drafter"] == d) & (df["k"] == k) & (df["temp"] == 0.0)]
            if s.empty:
                continue
            tok_s = float(s["n_emit"].sum() / s["wall_s"].sum())
            table["cells"][f"{d} k={k}"] = {
                "tpc": float(s["n_emit"].sum() / s["n_steps"].sum()),
                "tok_s": tok_s,
                "speedup_vs_none": tok_s / none_tok_s,
                "by_kind": {},
            }
            for kind, g in s.groupby("kind"):
                kt = float(g["n_emit"].sum() / g["wall_s"].sum())
                table["cells"][f"{d} k={k}"]["by_kind"][kind] = {
                    "tok_s": kt,
                    "speedup_vs_none": kt / float(none_kind.loc[kind, "tok_s"]),
                }
            print(f"{d + ' k=' + str(k):16s} "
                  f"{float(s['n_emit'].sum() / s['n_steps'].sum()):6.3f} "
                  f"{tok_s:7.2f} {tok_s / none_tok_s:7.2f}")
    out_t = RESULTS / "sweep_batch2_table.json"
    out_t.write_text(json.dumps(table, indent=1))
    print(f"wrote {out_t}")

    print("BATCH 2 CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
