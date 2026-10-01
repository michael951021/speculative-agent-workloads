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


def alpha_metric(name: str) -> dict:
    return json.loads((RES / f"alpha_sweep_{name}.json").read_text())


def sweep_frame() -> pl.DataFrame:
    df = pl.read_parquet(RES / "sweep.parquet")
    none_ts = {
        t: float(
            df.filter((pl.col("drafter") == "none") & (pl.col("temp") == t)).select(
                (pl.sum("n_emit") / pl.sum("wall_s")).alias("v")
            )[0, 0]
        )
        for t in (0.0, 0.7)
    }
    agg = (
        df.filter(pl.col("drafter") != "none")
        .group_by(["drafter", "k", "temp"])
        .agg(
            acc=pl.sum("n_acc") / pl.sum("n_draft"),
            tpc=pl.sum("n_emit") / pl.sum("n_steps"),
            tok_s=pl.sum("n_emit") / pl.sum("wall_s"),
        )
    )
    return agg, none_ts


def fig_alpha_by_kind() -> None:
    """Grouped bars: metric alpha (target 0.5B-Instruct, val) per drafter x kind."""
    kinds = [
        "code",
        "text",
        "thinking",
        "tool_result",
        "tool_use:Bash",
        "tool_use:Edit",
        "tool_use:Read",
        "tool_use:WebFetch",
        "tool_use:Write",
    ]
    data = {d: [alpha_metric(d)["by_kind"][k]["alpha"] for k in kinds] for d in ORDER}
    x = range(len(kinds))
    w = 0.8 / len(ORDER)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for j, d in enumerate(ORDER):
        ax.bar(
            [i + (j - 1.5) * w for i in x],
            data[d],
            width=w * 0.92,
            color=COLOR[d],
            label=LABEL[d],
            zorder=3,
        )
    ax.set_xticks(list(x), [k.replace("tool_use:", "tu:") for k in kinds], rotation=30, ha="right")
    ax.set_ylabel("alpha (expected acceptance)")
    ax.set_title(
        "Drafter quality: alpha by content kind (metric, target Qwen2.5-0.5B-Instruct, val split)"
    )
    ax.grid(axis="y", color="#ddd", lw=0.6, zorder=0)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "alpha_by_kind.png", dpi=150)
    plt.close(fig)


def fig_tpc_vs_k() -> None:
    """Tokens per target call vs k: metric (left) and measured engine (right)."""
    ks = [1, 2, 3, 4, 5, 6, 7, 8]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for d in ORDER:
        a1.plot(
            ks,
            [alpha_metric(d)["overall"]["tokens_per_call"][str(k)] for k in ks],
            color=COLOR[d],
            lw=2,
            marker="o",
            ms=4,
            label=LABEL[d],
        )
    a1.set_xlabel("draft length k")
    a1.set_ylabel("tokens per target call")
    a1.set_title("Metric (teacher-forced, val)")
    a1.grid(color="#ddd", lw=0.6)
    a1.legend(frameon=False, fontsize=8)
    agg, _ = sweep_frame()
    for d in ORDER:
        for t, ls in ((0.0, "-"), (0.7, "--")):
            r = agg.filter((pl.col("drafter") == d) & (pl.col("temp") == t)).sort("k")
            a2.plot(
                r["k"],
                r["tpc"],
                color=COLOR[d],
                lw=2,
                ls=ls,
                label=LABEL[d] if t == 0.0 else None,
                marker="o" if t == 0.0 else None,
                ms=4,
            )
    a2.axhline(1.0, color="#888", lw=1, ls=":")
    a2.set_xlabel("draft length k")
    a2.set_title("Measured engine (target 0.5B-Instruct; solid T=0, dashed T=0.7)")
    a2.grid(color="#ddd", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "tpc_vs_k.png", dpi=150)
    plt.close(fig)


def fig_speedup() -> None:
    """Measured speedup vs no-drafting, by k, both temperatures."""
    agg, none_ts = sweep_frame()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, t in zip(axes, (0.0, 0.7), strict=True):
        for d in ORDER:
            r = agg.filter((pl.col("drafter") == d) & (pl.col("temp") == t)).sort("k")
            sp = [v / none_ts[t] for v in r["tok_s"]]
            ax.plot(r["k"], sp, color=COLOR[d], lw=2, marker="o", ms=4, label=LABEL[d])
            if d == "ngram":  # direct labels on the best drafter (relief for low-contrast colors)
                for x, y in zip(r["k"], sp, strict=True):
                    ax.annotate(
                        f"{y:.2f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center",
                        fontsize=7,
                    )
        ax.axhline(1.0, color="#888", lw=1, ls=":")
        ax.set_xlabel("draft length k")
        ax.set_title(f"Temperature T={t}")
        ax.grid(color="#ddd", lw=0.6)
    axes[0].set_ylabel("tok/s vs no drafting (speedup)")
    axes[0].legend(frameon=False, fontsize=8, loc="center left")
    fig.suptitle(
        "Measured speedup vs no drafting, Qwen2.5-0.5B-Instruct on CPU (393 val prompts x 16 gen)",
        fontsize=10,
        y=1.02,
    )
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
