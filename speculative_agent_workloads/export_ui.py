"""Export results -> web/public/data/{summary,alpha,sweep,replays}.json (schema v1).

summary.json  corpus stats, per-drafter overall alpha, sweep grid status. The
              expected 42-cell grid (5 drafters x k 1/2/4/8 x T 0/0.7 + none per
              temp) is always listed; cells missing from results/sweep.parquet
              (e.g. batch 3 still running) are marked {"status": "pending"}
              instead of failing. Rerun after the sweep to fill them in.
alpha.json    per drafter: per-kind alpha/greedy/coverage + tpc/speedup tables,
              copied from results/alpha/*.json.
sweep.json    one entry per cell (drafter, k, temp) present in sweep.parquet:
              tpc = sum n_emit/sum n_steps, tok_s = sum n_emit/sum wall_s,
              speedup_vs_none = tok_s / none tok_s (same temp), by_kind split.
replays.json  30 val blocks (5 per kind; top-6 token-weighted kinds from the
              best drafter's alpha file; round-robin over runs within the same
              400-block pool/ctx as the alpha runs). Per token: p = target prob
              of the next corpus token, q = drafter prob (null when the drafter
              has no proposal, e.g. ngram miss; also null for the unscorable
              first token of a block with empty context), accept = greedy
              spec-accept (proposed and argmax q == argmax p), alpha = per-
              position sum_x min(p(x), q(x)) (0 when no proposal). Drafters:
              the best (argmax overall alpha in results/alpha) + ngram.

CLI: python -m speculative_agent_workloads.export_ui [--out-dir web/public/data] [--no-replays]
     [--max-blocks N]   (replays need the 1.5B target JAX + drafter forwards)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ALPHA_DIR = RESULTS / "alpha"
SWEEP_PARQUET = RESULTS / "sweep.parquet"
CORPUS_STATS = RESULTS / "corpus_stats.json"
OUT_DIR = ROOT / "web" / "public" / "data"

SCHEMA_VERSION = 1
GRID_DRAFTERS = ("ngram", "tiny", "base", "lora", "lora_dpo")
GRID_KS = (1, 2, 4, 8)
GRID_TEMPS = (0.0, 0.7)
REPLAY_BLOCKS_PER_KIND = 5
REPLAY_N_KINDS = 6


def cell_key(drafter: str, k: int, temp: float) -> str:
    return f"{drafter} k={k} T={temp}"


def load_alpha_results() -> dict[str, dict]:
    out = {}
    for p in sorted(ALPHA_DIR.glob("*.json")):
        out[p.stem] = json.loads(p.read_text())
    if not out:
        raise SystemExit(f"no results in {ALPHA_DIR} — run the alpha ticket first")
    return out


def best_drafter(alpha: dict[str, dict]) -> str:
    return max(alpha, key=lambda d: alpha[d]["overall"]["alpha"])


def build_alpha(alpha: dict[str, dict]) -> dict:
    """alpha.json: per-drafter summary + by_kind tables from results/alpha/."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": alpha[best_drafter(alpha)]["target"],
        "split": alpha[best_drafter(alpha)]["split"],
        "metric": "see results/alpha/<drafter>.json 'metric' field",
        "best": best_drafter(alpha),
        "drafters": {
            d: {
                "type": v["drafter_type"],
                "n_blocks": v["n_blocks"],
                "n_tokens": v["n_tokens"],
                "overall": v["overall"],
                "by_kind": v["by_kind"],
            }
            for d, v in sorted(alpha.items())
        },
    }


def _aggregate(group, none_tok_s: dict[float, float], none_by_kind: dict[float, dict[str, float]]):
    """((drafter, k, temp), group) -> sweep.json cell dict."""
    (drafter, k, temp), g = group
    cols = ("n_emit", "n_steps", "n_acc", "n_draft")
    n_emit, n_steps, n_acc, n_draft = (int(g[c].sum()) for c in cols)
    wall = float(g["wall_s"].sum())
    base = none_tok_s.get(temp, none_tok_s.get(0.0))
    by_kind = {}
    for kind, kg in g.groupby("kind"):
        e, w = int(kg["n_emit"].sum()), float(kg["wall_s"].sum())
        spd = 1.0 if drafter == "none" else (e / w) / none_by_kind[temp].get(kind, base)
        by_kind[kind] = {"tok_s": e / w, "speedup_vs_none": spd}
    tok_s = n_emit / wall
    return {
        "status": "done",
        "n_prompts": int(g[["run", "seq"]].drop_duplicates().shape[0]),
        "n_steps": n_steps,
        "n_draft": n_draft,
        "n_acc": n_acc,
        "n_emit": n_emit,
        "wall_s": wall,
        "alpha": None if n_draft == 0 else n_acc / n_draft,
        "tpc": n_emit / n_steps,
        "tok_s": tok_s,
        "speedup_vs_none": 1.0 if drafter == "none" else tok_s / base,
        "by_kind": by_kind,
    }


def build_sweep() -> tuple[dict, dict, set[str]]:
    """sweep.json + summary.sweep section + set of done cell keys."""
    import pandas as pd

    df = pd.read_parquet(SWEEP_PARQUET)
    none = df[df["drafter"] == "none"]
    none_tok_s = {
        float(t): int(g["n_emit"].sum()) / float(g["wall_s"].sum())
        for (t), g in none.groupby("temp")
    }
    none_by_kind = {}
    for t, g in none.groupby("temp"):
        none_by_kind[float(t)] = {
            k_: int(kg["n_emit"].sum()) / float(kg["wall_s"].sum()) for k_, kg in g.groupby("kind")
        }
    cells: dict[str, dict] = {}
    for (d, k, t), g in df.groupby(["drafter", "k", "temp"]):
        key = cell_key(d, int(k), float(t))
        cells[key] = _aggregate(((d, int(k), float(t)), g), none_tok_s, none_by_kind)
    summary_cells = {}
    for d in GRID_DRAFTERS:
        for k in GRID_KS:
            for t in GRID_TEMPS:
                key = cell_key(d, k, t)
                c = cells.get(key)
                summary_cells[key] = (
                    {
                        "status": "done",
                        "alpha": c["alpha"],
                        "tpc": c["tpc"],
                        "tok_s": c["tok_s"],
                        "speedup_vs_none": c["speedup_vs_none"],
                    }
                    if c
                    else {"status": "pending"}
                )
    for t in GRID_TEMPS:
        key = f"none k=0 T={t}"
        c = cells.get(key)
        summary_cells[key] = (
            {
                "status": "done",
                "alpha": c["alpha"],
                "tpc": c["tpc"],
                "tok_s": c["tok_s"],
                "speedup_vs_none": c["speedup_vs_none"],
            }
            if c
            else {"status": "pending"}
        )
    # any cells present in the parquet but outside the expected grid
    for key, c in cells.items():
        summary_cells.setdefault(
            key,
            {
                "status": "done",
                "alpha": c["alpha"],
                "tpc": c["tpc"],
                "tok_s": c["tok_s"],
                "speedup_vs_none": c["speedup_vs_none"],
            },
        )
    out = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metric": "tpc = sum(n_emit)/sum(n_steps); tok_s = sum(n_emit)/sum(wall_s); "
        "speedup_vs_none = tok_s / none tok_s (same temp); alpha = n_acc/n_draft",
        "target": str(df["target"].iloc[0]),
        "n_gen": int(df["n_gen"].iloc[0]),
        "max_tokens": int(df["max_tokens"].iloc[0]),
        "none_tok_s": {str(t): v for t, v in sorted(none_tok_s.items())},
        "cells": cells,
    }
    summary = {
        "target": out["target"],
        "n_cells_done": sum(1 for c in summary_cells.values() if c["status"] == "done"),
        "n_cells_expected": len(summary_cells),
        "cells": summary_cells,
    }
    return out, summary, set(cells)


def build_summary(alpha: dict[str, dict], sweep_summary: dict, replays_path: Path) -> dict:
    corpus = json.loads(CORPUS_STATS.read_text())
    al = {
        d: {
            "target": v["target"],
            "alpha": v["overall"]["alpha"],
            "alpha_ci95": v["overall"]["alpha_ci95"],
            "greedy": v["overall"]["greedy"],
            "greedy_ci95": v["overall"]["greedy_ci95"],
            "coverage": v["overall"]["coverage"],
            "tokens_per_call": v["overall"]["tokens_per_call"],
            "speedup": v["overall"]["speedup"],
        }
        for d, v in sorted(alpha.items())
    }
    if replays_path.exists():
        r = json.loads(replays_path.read_text())
        kinds = Counter(b["kind"] for b in r["blocks"])
        replays = {
            "status": "done",
            "n_blocks": len(r["blocks"]),
            "kinds": dict(sorted(kinds.items())),
            "drafters": sorted(r["drafters"]),
            "target": r["target"],
        }
    else:
        replays = {"status": "pending", "note": "replays not (re)generated this run"}
    inputs = ["results/sweep.parquet", "results/corpus_stats.json"] + sorted(
        str(p.relative_to(ROOT)) for p in ALPHA_DIR.glob("*.json")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": inputs,
        "corpus": {"splits": corpus["by_split"], "kinds": corpus["by_kind"]},
        "alpha": {**al, "best": best_drafter(alpha)},
        "sweep": sweep_summary,
        "replays": replays,
    }


# ---------------------------------------------------------------- replays


def select_replay_blocks(
    alpha_results: dict[str, dict],
    n_kinds: int = REPLAY_N_KINDS,
    per_kind: int = REPLAY_BLOCKS_PER_KIND,
) -> list[dict]:
    """Top-n_kinds kinds by n_tokens in the best drafter's alpha file; from the
    val split's 400-block pool (always val: replays publish block text, and
    val text was reviewed for release), round-robin over runs, `per_kind` each."""
    from speculative_agent_workloads.eval.alpha import load_blocks

    best = alpha_results[best_drafter(alpha_results)]
    kinds = sorted(best["by_kind"], key=lambda k: -best["by_kind"][k]["n_tokens"])[:n_kinds]
    blocks = load_blocks("val", 256, 400)
    by_kind: dict[str, list[dict]] = {k: [] for k in kinds}
    for b in blocks:
        if b["kind"] in by_kind:
            by_kind[b["kind"]].append(b)
    chosen: list[dict] = []
    for k in kinds:
        by_run: dict[str, list[dict]] = {}
        for b in sorted(by_kind[k], key=lambda b: (b["run"], b["seq"])):
            by_run.setdefault(b["run"], []).append(b)
        runs = sorted(by_run)
        pos = {r: 0 for r in runs}
        n_k = 0
        while n_k < per_kind:
            picked = False
            for r in runs:
                if pos[r] < len(by_run[r]):
                    chosen.append(by_run[r][pos[r]])
                    pos[r] += 1
                    n_k += 1
                    picked = True
                    if n_k >= per_kind:
                        break
            if not picked:
                break
    return chosen


def build_replays(alpha_results: dict[str, dict], max_blocks: int) -> dict:
    """Score selected val blocks with the alpha-run target/ctx; per-token p, q,
    accept (greedy spec-accept), alpha for the best drafter + ngram."""
    from transformers import AutoTokenizer

    from speculative_agent_workloads.eval.alpha import (
        HFDrafter,
        JaxModel,
        NGramDrafter,
        _context,
        alpha_and_greedy,
    )

    best_name = best_drafter(alpha_results)
    best = alpha_results[best_name]
    if best["drafter_type"] != "hf":
        raise SystemExit(
            f"best drafter {best_name} is type {best['drafter_type']!r}; "
            "replays support hf adapters"
        )
    t = Path(best["target"])
    target_dir = t if t.is_absolute() else ROOT / "models" / best["target"]
    tok = AutoTokenizer.from_pretrained(target_dir)

    def ids_of(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    ctx_budget = best.get("ctx_budget", 1024)
    target = JaxModel.from_path(target_dir, ctx_budget + 256)
    d_best = HFDrafter(ROOT / "ckpt" / best_name)
    d_ngram = NGramDrafter(target.cfg.vocab_size)
    drafters = {"ngram": d_ngram, best_name: d_best}

    blocks = select_replay_blocks(alpha_results)[:max_blocks]
    out_blocks = []
    for i, b in enumerate(blocks):
        ctx = _context(b["ctx_blocks"], ids_of, ctx_budget)
        ids = ids_of(b["text"])
        p, valid_p = target.dists(ctx, ids)
        tok_text = [tok.decode([int(x)]) for x in ids]
        entry = {
            "run": b["run"],
            "seq": int(b["seq"]),
            "kind": b["kind"],
            "n_tokens": len(ids),
            "tokens": [{"id": int(x), "text": t} for x, t in zip(ids, tok_text, strict=True)],
        }
        for name, d in drafters.items():
            q, valid_q = d.dists(ctx, ids)
            valid = valid_p & valid_q
            a, g = alpha_and_greedy(p, q, valid)
            p_t, q_t = p[np.arange(len(ids)), ids], q[np.arange(len(ids)), ids]

            def r6(x):
                return None if x is None else round(float(x), 6)

            entry[name] = {
                "p": [None if not v else r6(p_t[j]) for j, v in enumerate(valid)],
                "q": [None if not v else r6(q_t[j]) for j, v in enumerate(valid)],
                "accept": [bool(g[j]) for j in range(len(ids))],
                "alpha": [None if not valid[j] else r6(a[j]) for j in range(len(ids))],
            }
        out_blocks.append(entry)
        if (i + 1) % 5 == 0:
            print(f"  replay scored {i + 1}/{len(blocks)} blocks", flush=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": best["target"],
        "split": "val",
        "ctx_budget": ctx_budget,
        "accept_rule": "greedy spec-accept: proposed and argmax q == argmax p (target)",
        "metric": "p/q = model prob of the next corpus token; alpha = sum_x min(p(x), q(x)) "
        "per position (0 where the drafter has no proposal); null where the position is "
        "unscorable (first token of an empty-context block)",
        "drafters": {
            best_name: {"type": "hf", "path": f"ckpt/{best_name}"},
            "ngram": {"type": "ngram"},
        },
        "blocks": out_blocks,
    }


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"wrote {shown} ({path.stat().st_size / 1e6:.2f} MB)", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m speculative_agent_workloads.export_ui")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-replays", action="store_true", help="skip the heavy replay pass")
    ap.add_argument("--max-blocks", type=int, default=REPLAY_BLOCKS_PER_KIND * REPLAY_N_KINDS)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    alpha = load_alpha_results()
    sweep, sweep_summary, _ = build_sweep()
    if not args.no_replays:
        replays = build_replays(alpha, args.max_blocks)
        write_json(out_dir / "replays.json", replays)
    else:
        replays = None
    write_json(out_dir / "alpha.json", build_alpha(alpha))
    write_json(out_dir / "sweep.json", sweep)
    summary = build_summary(alpha, sweep_summary, out_dir / "replays.json")
    write_json(out_dir / "summary.json", summary)
    print(f"export_ui done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
