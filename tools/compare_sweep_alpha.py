"""Known-answer check (b): sweep-measured T=0 acceptance vs alpha-metric CIs.

Usage: python tools/compare_sweep_alpha.py [--drafters base lora lora_dpo]
Reads results/sweep.parquet (k=1, T=0 rows) and results/alpha_sweep_<drafter>.json
(written by draftlab.eval.alpha, target 0.5b-instruct, ctx-tokens 0). At k=1, T=0
the measured accept rate is the empirical mean of (argmax q == argmax p) over
drafted positions, i.e. the metric's `greedy_valid` (proposed positions only;
the diluted `greedy` also counts positions the drafter did not propose), so
measured overall accept must fall in greedy_valid_ci95, and measured tpc
(n_emit/n_steps) in tpc_valid_ci95 at k=1. Prints per-kind detail; exits 1 if
any overall comparison is outside its CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DRAFTERS = ["ngram", "base", "lora", "lora_dpo"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafters", nargs="+", default=DRAFTERS, choices=DRAFTERS)
    args = ap.parse_args()
    df = pd.read_parquet(ROOT / "results" / "sweep.parquet")
    sub = df[(df["k"] == 1) & (df["temp"] == 0.0)]
    ok = True
    print(
        f"{'drafter':10s} {'meas acc':>9s} {'metric greedy':>13s} {'greedy ci95':>22s} "
        f"{'meas tpc':>9s} {'tpc ci95':>22s}  verdict"
    )
    for d in args.drafters:
        s = sub[sub["drafter"] == d]
        if s.empty:
            print(f"{d:10s}  MISSING in sweep.parquet")
            ok = False
            continue
        meas = s["n_acc"].sum() / s["n_draft"].sum()
        tpc = s["n_emit"].sum() / s["n_steps"].sum()
        j = json.loads((ROOT / "results" / f"alpha_sweep_{d}.json").read_text())
        a, by_kind = j["overall"], j["by_kind"]
        lo, hi = a["greedy_valid_ci95"]
        in_ci = lo <= meas <= hi
        ok &= in_ci
        tlo, thi = a["tpc_valid_ci95"]["1"]
        in_tpc = tlo <= tpc <= thi
        ok &= in_tpc
        print(
            f"{d:10s} {meas:9.4f} {a['greedy_valid']:13.4f} [{lo:.4f}, {hi:.4f}] "
            f"{tpc:9.4f} [{tlo:.4f}, {thi:.4f}]  "
            f"{'PASS' if in_ci and in_tpc else 'FAIL'}"
        )
        print("    by kind (meas acc / metric greedy_valid [ci95]):")
        g = s.groupby("kind").agg(acc=("n_acc", "sum"), dr=("n_draft", "sum"))
        for kind, row in g.iterrows():
            m = row["acc"] / row["dr"]
            k = by_kind.get(kind)
            ref = (
                f"{k['greedy_valid']:.4f} [{k['greedy_valid_ci95'][0]:.4f}, "
                f"{k['greedy_valid_ci95'][1]:.4f}]"
                if k
                else "n/a"
            )
            print(f"      {kind:16s} {m:.4f}  {ref}")
    print("KNOWN ANSWER (b):", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
