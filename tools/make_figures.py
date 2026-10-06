"""Regenerate REPORT.md figures from results/ JSONs (matplotlib PNGs).

Usage: python tools/make_figures.py   (writes results/fig/*.png)
Palette: draftlab categorical slots 1-5 (validated, light mode).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "fig"
FIG.mkdir(parents=True, exist_ok=True)

# drafter -> color (validated categorical order, fixed across figures)
COLOR = {
    "ngram": "#2a78d6",
    "tiny": "#eb6834",
    "base": "#1baf7a",
    "lora": "#eda100",
    "lora_dpo": "#e87ba4",
}
LABEL = {
    "ngram": "ngram",
    "tiny": "tiny (43M)",
    "base": "base 0.5B (=target)",
    "lora": "LoRA 0.5B",
    "lora_dpo": "LoRA+DPO 0.5B",
}
ORDER = ["ngram", "base", "lora", "lora_dpo"]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": "#999",
        "axes.linewidth": 0.8,
        "xtick.color": "#333",
        "ytick.color": "#333",
        "text.color": "#1a1a1a",
    }
)


ALPHA_ORDER = ["ngram", "tiny", "lora", "dpo"]  # results/alpha/<name>.json
ALPHA_COLOR = {"ngram": COLOR["ngram"], "tiny": COLOR["tiny"], "lora": COLOR["lora"],
               "dpo": COLOR["lora_dpo"]}
ALPHA_LABEL = {"ngram": "ngram", "tiny": "tiny (43M)", "lora": "LoRA 0.5B", "dpo": "LoRA+DPO 0.5B"}
SWEEP_15 = RES / "sweep_1.5b.parquet"


def alpha_metric(name: str) -> dict:
    return json.loads((RES / "alpha" / f"{name}.json").read_text())


def sweep_agg(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .filter(pl.col("drafter") != "none")
        .group_by(["drafter", "k", "temp"])
        .agg(acc=pl.sum("n_acc") / pl.sum("n_draft"), tpc=pl.sum("n_emit") / pl.sum("n_steps"))
    )


def fig_alpha_by_kind() -> None:
    """Grouped bars + run-bootstrap 95% CIs: alpha vs the 1.5B target, held-out runs."""
    data = {d: alpha_metric(d) for d in ALPHA_ORDER}
    kinds = [k for k in sorted(data["ngram"]["by_kind"])
             if all(k in data[d]["by_kind"] and data[d]["by_kind"][k]["n_tokens"] >= 500
                    for d in ALPHA_ORDER)][:10]
    x = range(len(kinds))
    w = 0.8 / len(ALPHA_ORDER)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for j, d in enumerate(ALPHA_ORDER):
        bk = data[d]["by_kind"]
        ys = [bk[k]["alpha"] for k in kinds]
        err = [[y - bk[k]["alpha_ci95"][0] for k, y in zip(kinds, ys, strict=True)],
               [bk[k]["alpha_ci95"][1] - y for k, y in zip(kinds, ys, strict=True)]]
        ax.bar([i + (j - 1.5) * w for i in x], ys, width=w * 0.92, color=ALPHA_COLOR[d],
               label=ALPHA_LABEL[d], zorder=3, yerr=err, error_kw={"lw": 0.8, "ecolor": "#555"})
    ax.set_xticks(list(x), [k.replace("tool_use:", "tu:") for k in kinds], rotation=30, ha="right")
    ax.set_ylabel("alpha (expected acceptance)")
    n_runs = data["ngram"]["n_runs"]
    ax.set_title(f"Drafter quality by content kind: alpha vs Qwen2.5-1.5B-Instruct "
                 f"({n_runs} held-out runs, 95% run-bootstrap CI)")
    ax.grid(axis="y", color="#ddd", lw=0.6, zorder=0)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "alpha_by_kind.png", dpi=150)
    plt.close(fig)


def fig_tpc_vs_k() -> None:
    """Tokens per target call vs k, 1.5B target: metric (left), measured engine (right)."""
    ks = [1, 2, 3, 4, 5, 6, 7, 8]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for d in ALPHA_ORDER:
        a1.plot(ks, [alpha_metric(d)["overall"]["tokens_per_call"][str(k)] for k in ks],
                color=ALPHA_COLOR[d], lw=2, marker="o", ms=4, label=ALPHA_LABEL[d])
    a1.set_xlabel("draft length k")
    a1.set_ylabel("tokens per target call")
    a1.set_title("Metric (teacher-forced, held-out)")
    a1.grid(color="#ddd", lw=0.6)
    a1.legend(frameon=False, fontsize=8)
    agg = sweep_agg(SWEEP_15)
    for d in ["ngram", "lora", "lora_dpo"]:
        for t, ls in ((0.0, "-"), (0.7, "--")):
            r = agg.filter((pl.col("drafter") == d) & (pl.col("temp") == t)).sort("k")
            a2.plot(r["k"], r["tpc"], color=COLOR[d], lw=2, ls=ls,
                    label=LABEL[d] if t == 0.0 else None, marker="o" if t == 0.0 else None, ms=4)
    a2.axhline(1.0, color="#888", lw=1, ls=":")
    a2.set_xlabel("draft length k")
    a2.set_title("Measured engine (generated text; solid T=0, dashed T=0.7)")
    a2.grid(color="#ddd", lw=0.6)
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "tpc_vs_k.png", dpi=150)
    plt.close(fig)


def fig_speedup() -> None:
    """Head-to-head CPU speedup vs no drafting (tools/bench_speedup.py), rep range as bars."""
    b = json.loads((RES / "speedup_bench.json").read_text())
    labels, ys, lo, hi, cols = [], [], [], [], []
    for target, t in b["targets"].items():
        for c in t["configs"].values():
            if c["drafter"] == "none":
                continue
            short = target.replace("qwen2.5-", "").replace("-instruct", "")
            labels.append(f"{short} target\n{c['drafter']} k={c['k']} T={c['temp']}")
            ys.append(c["speedup"])
            lo.append(c["speedup"] - c["speedup_range"][0])
            hi.append(c["speedup_range"][1] - c["speedup"])
            cols.append(COLOR["ngram" if c["drafter"] == "ngram" else "lora"])
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.bar(range(len(ys)), ys, color=cols, yerr=[lo, hi], zorder=3,
           error_kw={"lw": 0.8, "ecolor": "#555"})
    for i, y in enumerate(ys):
        ax.annotate(f"{y:.2f}x", (i, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    ax.axhline(1.0, color="#888", lw=1, ls=":")
    ax.set_xticks(range(len(ys)), labels, fontsize=8)
    ax.set_ylabel("speedup vs no drafting")
    ax.set_title(f"Measured CPU speedup, head to head ({b['reps']} interleaved reps, "
                 f"{b['n_gen']} tokens/prompt; bars = rep range)")
    ax.grid(axis="y", color="#ddd", lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(FIG / "speedup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_distill_kl() -> None:
    d = json.loads((RES / "distill_curve.json").read_text())
    curve = d["curve"]
    val = [p["val_kl"] for p in curve if "val_kl" in p]
    vs = [p["step"] for p in curve if "val_kl" in p]
    tr = [p["train_loss"] for p in curve if "train_loss" in p]
    ts = [p["step"] for p in curve if "train_loss" in p]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(ts, tr, color="#94a3b8", lw=1.5, ls="--", label="train loss")
    ax.plot(vs, val, color=COLOR["lora"], lw=2, label="val KL (target 1.5B -> 43M)")
    ax.set_xlabel("training steps")
    ax.set_ylabel("loss / KL (nats)")
    ax.set_title(
        f"Distillation: val KL {d['val_kl_step0']:.2f} -> {d['val_kl_final']:.2f} "
        f"(-{d['kl_drop_frac'] * 100:.0f}%) over {d['steps']} steps"
    )
    ax.grid(color="#ddd", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "distill_kl.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_alpha_by_kind()
    fig_tpc_vs_k()
    fig_speedup()
    fig_distill_kl()
    print("wrote", ", ".join(f.name for f in sorted(FIG.glob("*.png"))))
