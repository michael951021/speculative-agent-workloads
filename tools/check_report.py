"""REPORT.md / README.md acceptance check: every number traces to results/.

Recomputes the report's values from results/ (alpha/*.json, sweep.parquet,
sweep_1.5b.parquet, speedup_bench.json, judge.json, engine_traj_metric.json,
corpus_stats.json), requires each in REPORT.md, then sweeps every number in
REPORT.md prose and tables (code spans and fences excluded) and fails on any
that is neither a recomputed value nor a listed structural constant.

    python tools/check_report.py           # exit 1 on any mismatch
    python tools/check_report.py --print   # dump the recomputed values
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FAIL: list[str] = []

DRAFTERS = ("ngram", "lora", "dpo", "tiny")
KINDS = ("text", "thinking", "tool_result", "tool_use:Bash", "tool_use:Edit", "tool_use:Read")

# Numbers that are design constants, not measurements: draft lengths, temperatures,
# model sizes, CI level, citation years, figure/section counts.
STRUCTURAL = {"0", "0.0", "0.7", "1", "2", "4", "8", "0.5", "1.5", "7", "27", "95", "2023", "100"}

# Corrections: the "before" values are what the October 5 report stated; the "after"
# values not recomputable from results/ come from the audit commits named in REPORT.md.
CORRECTIONS = {
    "0.3336",
    "0.3343",  # ngram CI before the bootstrap fix (0158d68)
    "0.249",
    "0.362",
    "3",  # ngram CI on the 3 val runs after it
    "2.3",
    "4.0",  # identity drafter distinct outputs per 4 seeds (1e9f5d3)
    "0.3042",  # lora k8 T=0.7 acceptance before the engine fix
    "0.7306",
    "0.7319",
    "0.7308",
    "0.7316",  # LoRA/DPO alphas and the 'disjoint' CI ends claimed before
    "0.0013",  # the DPO gain claimed before
    "0.10",  # the task note's expected lora k8 T=0.7 acceptance (not measured)
}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def f4(v: float) -> str:
    return f"{v:.4f}"


def f2(v: float) -> str:
    return f"{v:.2f}"


def thousands(v: int) -> str:
    return f"{v:,}"


def values() -> dict[str, str]:
    out: dict[str, str] = {}
    for d in DRAFTERS:
        a = json.loads((RES / "alpha" / f"{d}.json").read_text())
        o = a["overall"]
        out[f"alpha {d}"] = f4(o["alpha"])
        out[f"alpha {d} lo"], out[f"alpha {d} hi"] = map(f4, o["alpha_ci95"])
        out[f"greedy {d}"] = f4(o["greedy"])
        for kind in KINDS:
            out[f"alpha {d} {kind}"] = f4(a["by_kind"][kind]["alpha"])
        if d == "lora":
            out["heldout runs"] = str(a["n_runs"])
            out["heldout blocks"] = str(a["n_blocks"])
            out["heldout tokens"] = thousands(o["n_tokens"])
    out["coverage ngram"] = (
        f"{json.loads((RES / 'alpha/ngram.json').read_text())['overall']['coverage']:.1%}"
    )

    for name, path in (("0.5b", "sweep.parquet"), ("1.5b", "sweep_1.5b.parquet")):
        df = pl.read_parquet(RES / path)
        agg = df.group_by(["drafter", "k", "temp"]).agg(
            acc=pl.sum("n_acc") / pl.sum("n_draft"), tpc=pl.sum("n_emit") / pl.sum("n_steps")
        )
        for row in agg.iter_rows(named=True):
            if row["drafter"] == "none":
                continue
            key = f"sweep {name} {row['drafter']} k{row['k']} T{row['temp']}"
            out[f"{key} tpc"] = f4(row["tpc"])
            out[f"{key} acc"] = f4(row["acc"])
        out[f"sweep {name} prompts"] = str(df["i"].n_unique())
        out[f"sweep {name} gen"] = str(int(df["n_gen"][0]))

    bench = json.loads((RES / "speedup_bench.json").read_text())
    out["bench reps"] = str(bench["reps"])
    out["bench gen"] = str(bench["n_gen"])
    for target, t in bench["targets"].items():
        size = "0.5b" if "0.5b" in target else "1.5b"
        out[f"bench {size} prompts"] = str(t["n_prompts"])
        for cfg, r in t["configs"].items():
            out[f"bench {size} {cfg} tok_s"] = f2(r["tok_s"])
            if r["drafter"] != "none":
                out[f"bench {size} {cfg} speedup"] = f2(r["speedup"])
                lo, hi = r["speedup_range"]
                out[f"bench {size} {cfg} lo"], out[f"bench {size} {cfg} hi"] = f2(lo), f2(hi)
                out[f"bench {size} {cfg} accept"] = f4(r["accept"])
                out[f"bench {size} {cfg} tps"] = f2(r["tokens_per_step"])

    judge = json.loads((RES / "judge.json").read_text())
    for key in ("spec_vs_drafter_alone", "spec_vs_target"):
        j = judge[key]
        out[f"judge {key} win"] = f"{j['win_rate']:.3f}"
        out[f"judge {key} lo"], out[f"judge {key} hi"] = (f"{v:.3f}" for v in j["win_ci95"])
        out[f"judge {key} verdicts"] = str(j["n_verdicts"])
    out["judge calls"] = str(judge["meta"]["calls_used"])
    out["judge invalid"] = str(judge["meta"]["invalid"])

    eng = json.loads((RES / "engine_traj_metric.json").read_text())
    for d in ("lora", "lora_dpo"):
        out[f"engine {d} meas"] = f4(eng[d]["meas_acc"])
        out[f"engine {d} lo"], out[f"engine {d} hi"] = map(f4, eng[d]["traj_ci95"])

    corpus = json.loads((RES / "corpus_stats.json").read_text())
    train = next(s for s in corpus["by_split"] if s["split"] == "train")
    out["corpus train blocks"] = thousands(train["blocks"])
    out["corpus train runs"] = str(train["runs"])
    out["corpus train tokens M"] = f"{train['tokens'] / 1e6:.1f}"
    return out


def prose(text: str) -> str:
    """REPORT.md without fenced blocks and inline code (paths, commits, commands)."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    return re.sub(r"`[^`]*`", " ", text)


def main() -> int:
    vals = values()
    if "--print" in sys.argv:
        for k, v in vals.items():
            print(f"{k:45s} {v}")
        return 0
    report = (ROOT / "REPORT.md").read_text()
    readme = (ROOT / "README.md").read_text()

    required = [k for k in vals if not k.startswith("sweep 0.5b")]
    required += [f"sweep 0.5b base k{k} T0.0 tpc" for k in (1, 2, 4, 8)]
    required += ["sweep 0.5b lora k8 T0.7 acc"]
    for key in required:
        check(f"REPORT states {key} = {vals[key]}", vals[key] in report)

    allowed = {v.rstrip("%") for v in vals.values()} | STRUCTURAL | CORRECTIONS
    stray = []
    for token in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", prose(report)):
        token = token.rstrip(",")
        if token not in allowed and token.rstrip("0").rstrip(".") not in allowed:
            stray.append(token)
    check(
        "every number in REPORT.md traces to results/ or is structural",
        not stray,
        f"untraced: {sorted(set(stray))[:20]}",
    )

    for stale in ("check_sweep_batch", "compare_sweep_alpha", "alpha_sweep_", "0.3295"):
        check(f"no reference to removed {stale}", stale not in report and stale not in readme)

    figs = sorted((RES / "fig").glob("*.png"))
    check(">=3 figures in results/fig/", len(figs) >= 3, f"found {len(figs)}")
    for f in figs:
        check(f"REPORT references {f.name}", f.name in report)

    check("README has mermaid diagram", "```mermaid" in readme)
    check("README has quickstart", re.search(r"quickstart", readme, re.I) is not None)
    for skill in (
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
    ):
        check(
            f"README skills map has {skill}", re.search(re.escape(skill), readme, re.I) is not None
        )

    print(f"\n{len(FAIL)} failure(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
