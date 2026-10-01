"""Acceptance check for REPORT.md / README.md.

Recomputes headline numbers from results/ (sweep parquet + alpha JSONs + judge)
and asserts they are present in REPORT.md, figures exist, and README has the
architecture diagram + skills map. Exit 0 = pass.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def fmt(v: float, n: int = 4) -> str:
    return f"{v:.{n}f}"


def main() -> int:
    report = (ROOT / "REPORT.md").read_text() if (ROOT / "REPORT.md").exists() else ""
    readme = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""

    # --- sweep: recompute pooled metrics from the parquet, compare to tables
    df = pl.read_parquet(ROOT / "results/sweep.parquet")
    agg = df.group_by(["drafter", "k", "temp"]).agg(
        acc=pl.sum("n_acc") / pl.sum("n_draft"),
        tpc=pl.sum("n_emit") / pl.sum("n_steps"),
        tok_s=pl.sum("n_emit") / pl.sum("wall_s"),
    )

    def cell(drafter: str, k: int, temp: float) -> dict:
        r = agg.filter(
            (pl.col("drafter") == drafter) & (pl.col("k") == k) & (pl.col("temp") == temp)
        )
        return {c: float(r[c][0]) for c in ("acc", "tpc", "tok_s")} if len(r) else {}

    none_ts = {
        0.0: float(
            df.filter((pl.col("drafter") == "none") & (pl.col("temp") == 0.0)).select(
                pl.sum("n_emit") / pl.sum("wall_s")
            )[0, 0]
        ),
        0.7: float(
            df.filter((pl.col("drafter") == "none") & (pl.col("temp") == 0.7)).select(
                pl.sum("n_emit") / pl.sum("wall_s")
            )[0, 0]
        ),
    }

    # known-answer: identity drafter (base = target weights) must give acc 1.0 and
    # tpc = n_gen / steps with the engine's k=min(k,rem) tail rule (Leviathan a=1, capped):
    # each prompt emits exactly n_gen tokens, e.g. n_gen=16: k1->16/8, k2->16/6, k4->16/4, k8->16/2
    n_gen = int(df.filter(pl.col("drafter") == "base")["n_gen"][0])

    def perfect_tpc(k: int, n: int) -> float:
        rem, steps = n, 0
        while rem > 0:
            rem -= min(rem, min(k, rem) + 1)
            steps += 1
        return n / steps

    for k in (1, 2, 4, 8):
        c = cell("base", k, 0.0)
        check(f"known-answer base k={k} acc=1.0", abs(c.get("acc", -1) - 1.0) < 1e-9)
        exp = perfect_tpc(k, n_gen)
        check(
            f"known-answer base k={k} tpc={exp:.4f}",
            abs(c.get("tpc", -1) - exp) < 1e-6,
            f"got {c.get('tpc')}",
        )

    # ngram T=0.7 k1 acc must match the hand-computed reference
    # (tools/check_ngram_t07_ref.py: 0.3295, CI contains the difference)
    ng = cell("ngram", 1, 0.7)["acc"]
    ref = 0.3295
    check(
        "ngram T=0.7 k1 acc in hand-computed ref range",
        abs(ng - ref) <= 0.01,
        f"got {ng:.4f} ref {ref}",
    )

    headline = {  # (drafter, k, temp): number that must appear in REPORT.md
        ("ngram", 1, 0.7): ng,
        ("ngram", 8, 0.0): cell("ngram", 8, 0.0)["acc"],
        ("lora", 8, 0.0): cell("lora", 8, 0.0)["acc"],
        ("lora", 8, 0.7): cell("lora", 8, 0.7)["acc"],
        ("lora_dpo", 8, 0.0): cell("lora_dpo", 8, 0.0)["acc"],
        ("lora_dpo", 8, 0.7): cell("lora_dpo", 8, 0.7)["acc"],
        ("tiny", 1, 0.0): cell("tiny", 1, 0.0)["acc"],
    }
    for (d, k, t), v in headline.items():
        check(f"REPORT has sweep acc {d} k{k} T{t}", fmt(v) in report)
    for t, v in none_ts.items():
        check(f"REPORT has none tok/s T{t}", fmt(v, 2) in report)

    # measured speedups: recompute tok_s/none, must appear for the best ngram cell per temp
    for t in (0.0, 0.7):
        c = cell("ngram", 8, t)
        sp = c["tok_s"] / none_ts[t]
        check(f"REPORT has ngram k8 speedup T{t}", fmt(sp) in report)

    # --- alpha metric: headline per-drafter overall alpha (both targets) in REPORT
    for f in [
        "results/alpha_sweep_ngram.json",
        "results/alpha_sweep_base.json",
        "results/alpha_sweep_lora.json",
        "results/alpha_sweep_lora_dpo.json",
        "results/alpha/ngram.json",
        "results/alpha/tiny.json",
        "results/alpha/lora.json",
        "results/alpha/dpo.json",
    ]:
        d = json.loads((ROOT / f).read_text())
        a = d["overall"]["alpha"]
        check(f"REPORT has alpha {Path(f).name}", fmt(a) in report)

    # --- judge
    j = json.loads((ROOT / "results/judge.json").read_text())
    for key in ("spec_vs_drafter_alone", "spec_vs_target"):
        wr = j[key]["win_rate"]
        check(f"REPORT has judge {key} win_rate", fmt(wr, 3) in report)

    # --- figures
    figs = sorted((ROOT / "results/fig").glob("*.png")) if (ROOT / "results/fig").exists() else []
    check(">=3 figures in results/fig/", len(figs) >= 3, f"found {len(figs)}")
    for f in figs[:6]:
        check(f"REPORT references {f.name}", f.name in report)

    # --- REPORT sections
    for sec in ("question", "corpus", "method", "judge", "fail", "gpu"):
        check(f"REPORT mentions '{sec}'", re.search(sec, report, re.I) is not None)

    # --- README
    check("README has mermaid diagram", "```mermaid" in readme)
    check("README has quickstart", re.search(r"quickstart", readme, re.I) is not None)
    for skill in [
        "JAX",
        "PyTorch",
        "FSDP",
        "LoRA",
        "DPO",
        "Triton",
        "Ray",
        "Go",
        "Prometheus",
        "Grafana",
        "Kubernetes",
        "Helm",
        "Terraform",
        "React",
        "LLM-as-judge",
        "OpenAI",
    ]:
        check(
            f"README skills map has {skill}", re.search(re.escape(skill), readme, re.I) is not None
        )

    print(f"\n{len(FAIL)} failure(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
