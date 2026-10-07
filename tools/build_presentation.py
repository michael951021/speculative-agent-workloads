"""Build a portable report for Speculative Decoding for Agent Workloads.

Uses saved evidence without running experiments.

python tools/build_presentation.py
The output embeds its styles, charts, scripts and aggregate data. No corpus text
or prompt identifiers are exported. Requires pyarrow to read recorded Parquet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "ngram": "N-gram",
    "tiny": "Tiny distilled",
    "base": "Identity control",
    "lora": "LoRA",
    "lora_dpo": "LoRA + DPO",
    "dpo": "LoRA + DPO",
    "none": "Target only",
}
COMMITS = {
    "alignment": ("7a84891", ["draftlab/eval/alpha.py", "draftlab/tests/test_alpha_known.py"]),
    "cache": ("ae371e3", ["draftlab/jaxlm/spec.py", "draftlab/tests/test_spec.py"]),
    "sampling": ("1e9f5d3", ["draftlab/jaxlm/spec.py", "draftlab/tests/test_spec.py"]),
    "bootstrap": ("0158d68", ["draftlab/eval/alpha.py", "draftlab/tests/test_alpha.py"]),
    "heldout": ("5f98414", ["draftlab/data/corpus.py", "tools/bench_speedup.py"]),
    "serving": ("48cf15b", ["draftlab/serve/app.py", "draftlab/tests/test_serve.py"]),
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def build() -> dict:
    sources = []

    def record(path: Path, purpose: str) -> bytes:
        raw = path.read_bytes()
        sources.append(
            {
                "path": str(path.relative_to(ROOT)),
                "purpose": purpose,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
        return raw

    alpha = {}
    for name in ("ngram", "lora", "dpo", "tiny"):
        d = json.loads(record(ROOT / f"results/alpha/{name}.json", "Held-out distribution overlap"))
        if d["split"] != "heldout" or d["target"] != "qwen2.5-1.5b-instruct":
            raise ValueError("Report expects held-out alpha against the 1.5B target")
        alpha[name] = {
            k: d[k]
            for k in (
                "target",
                "split",
                "n_blocks",
                "n_runs",
                "n_tokens",
                "max_tokens",
                "ctx_budget",
                "bootstrap",
                "overall",
                "by_kind",
            )
        }
    identities = {
        (d["n_blocks"], d["n_runs"], d["n_tokens"], tuple(d["by_kind"])) for d in alpha.values()
    }
    if len(identities) != 1:
        raise ValueError("Alpha comparisons must use the same evaluation support")

    corpus = json.loads(record(ROOT / "results/corpus_stats.json", "Original corpus composition"))
    heldout_path = ROOT / "data/corpus/heldout.parquet"
    # Only aggregate columns are read; text is never included in this report.
    record(heldout_path, "Temporal hold-out size; aggregate columns only")
    heldout = pq.read_table(heldout_path, columns=["run", "kind", "n_tokens"]).to_pylist()
    heldout_stats = {
        "blocks": len(heldout),
        "runs": len({r["run"] for r in heldout}),
        "tokens": sum(r["n_tokens"] for r in heldout),
        "kinds": dict(Counter(r["kind"] for r in heldout)),
    }

    p = ROOT / "results/sweep.parquet"
    record(p, "Corrected 0.5B decoding sweep, completed cells")
    rows = pq.read_table(p).to_pylist()
    for part in sorted(p.parent.glob(p.name + ".part-*.jsonl")):
        raw = record(part, "Recoverable partial sweep cell; read without modifying it")
        rows.extend(json.loads(line) for line in raw.splitlines() if line.strip())
    unique = {}
    for row in rows:
        if row.get("n_gen") == 32 and row.get("target") == "qwen2.5-0.5b-instruct":
            unique[(row["drafter"], row["k"], row["temp"], row["i"])] = row
    rows = list(unique.values())
    groups = defaultdict(list)
    for row in rows:
        groups[(row["drafter"], row["k"], row["temp"])].append(row)
    expected_prompts = max(len(g) for g in groups.values())
    full_prompt_sets = {
        tuple(sorted(r["i"] for r in g)) for g in groups.values() if len(g) == expected_prompts
    }
    if len(full_prompt_sets) != 1:
        raise ValueError("Completed sweep cells disagree on prompt support")
    support = set(next(iter(full_prompt_sets)))
    cells = []
    for name in ("ngram", "tiny", "base", "lora", "lora_dpo", "none"):
        for k in [0] if name == "none" else [1, 2, 4, 8]:
            for temp in (0.0, 0.7):
                g = groups.get((name, k, temp), [])
                if any(r["i"] not in support for r in g):
                    raise ValueError("Partial sweep contains an unexpected prompt")
                n_draft = sum(r["n_draft"] for r in g)
                n_steps = sum(r["n_steps"] for r in g)
                n_emit = sum(r["n_emit"] for r in g)
                cells.append(
                    {
                        "drafter": name,
                        "k": k,
                        "temp": temp,
                        "n_prompts": len(g),
                        "n_runs": len({r["run"] for r in g}),
                        "status": "complete"
                        if len(g) == expected_prompts
                        else "partial"
                        if g
                        else "pending",
                        "n_draft": n_draft,
                        "n_acc": sum(r["n_acc"] for r in g),
                        "n_steps": n_steps,
                        "n_emit": n_emit,
                        "accept": sum(r["n_acc"] for r in g) / n_draft if n_draft else None,
                        "tpc": n_emit / n_steps if n_steps else None,
                    }
                )
    sweep = {
        "target": "qwen2.5-0.5b-instruct",
        "n_gen": 32,
        "n_prompts_per_cell": expected_prompts,
        "rows": len(rows),
        "cells": cells,
        "statuses": dict(Counter(c["status"] for c in cells)),
    }

    judge = json.loads(record(ROOT / "results/judge.json", "Corrected held-out judge rerun"))
    judge_summary = {}
    for key, value in judge.items():
        item = {z: v for z, v in value.items() if z != "per_prompt"}
        if "per_prompt" in value:
            valid_pairs = [p for p in value["per_prompt"] if p.get("ab") and p.get("ba")]
            item["both_orders_valid"] = len(valid_pairs)
            item["both_orders_disagree"] = sum(p["ab"] != p["ba"] for p in valid_pairs)
        judge_summary[key] = item
    judge = judge_summary
    curves = {}
    for name in ("distill_curve", "distill_curve_tinit"):
        curves[name] = json.loads(
            record(ROOT / f"results/{name}.json", "Recorded distillation training curve")
        )
    for path in [
        "ckpt/lora/adapter_config.json",
        "ckpt/dpo/adapter_config.json",
        "speculative_agent_workloads/jaxlm/spec.py",
        "speculative_agent_workloads/jaxlm/model.py",
        "speculative_agent_workloads/eval/alpha.py",
        "speculative_agent_workloads/data/corpus.py",
        "speculative_agent_workloads/drafters/dpo.py",
        "speculative_agent_workloads/drafters/lora.py",
        "speculative_agent_workloads/drafters/distill.py",
        "speculative_agent_workloads/kernels/verify.py",
        "speculative_agent_workloads/eval/judge.py",
        "speculative_agent_workloads/serve/app.py",
        "speculative_agent_workloads/sweep.py",
        "gateway/metrics.go",
        "TODO.md",
    ]:
        record(ROOT / path, "Implementation or outstanding work used for this report")
    missing = [
        name
        for name in ("results/speedup_bench.json", "results/sweep_1.5b.parquet")
        if not (ROOT / name).exists()
    ]
    if len(missing) != 2:
        raise ValueError(
            "New benchmark results exist: update the report narrative before publishing"
        )

    commits = {}
    for key, (ref, paths) in COMMITS.items():
        commits[key] = {
            "sha": git("rev-parse", ref),
            "short": ref,
            "date": git("show", "-s", "--format=%cs", ref),
            "subject": git("show", "-s", "--format=%s", ref),
            "paths": paths,
            "diff": git("show", "--format=", "--unified=2", ref, "--", *paths),
        }
    now = datetime.now(UTC)
    return {
        "schema_version": 1,
        "generated_at": now.isoformat(timespec="seconds"),
        "revision": git("rev-parse", "HEAD"),
        "alpha": alpha,
        "corpus": {k: corpus[k] for k in ("by_split", "by_kind", "length_percentiles")},
        "heldout": heldout_stats,
        "sweep": sweep,
        "judge": judge,
        "curves": curves,
        "commits": commits,
        "sources": sources,
        "missing": missing,
        "notes": [
            "This is a frozen snapshot, not a live dashboard.",
            "No raw corpus text, prompts, token sequences or run identifiers are exported.",
            "Alpha entries are saved bootstrap means and run-level 95% intervals.",
            "Sweep metrics are pooled counts; partial cells use different prompt subsets.",
            "Sweep wall time is intentionally omitted; speedup needs the dedicated benchmark.",
        ],
    }


def csv_data(d: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "experiment",
            "target",
            "scope",
            "drafter",
            "k",
            "temperature",
            "metric",
            "value",
            "ci95_low",
            "ci95_high",
            "n_tokens",
            "n_runs",
            "n_prompts",
            "status",
        ]
    )
    for name, a in d["alpha"].items():
        for kind, val in {"overall": a["overall"], **a["by_kind"]}.items():
            for metric in ("alpha", "greedy", "coverage"):
                w.writerow(
                    [
                        "heldout_alpha",
                        a["target"],
                        kind,
                        name,
                        "",
                        "1 (unscaled softmax)",
                        metric,
                        val[metric],
                        *val.get(metric + "_ci95", ["", ""]),
                        val["n_tokens"],
                        val["n_runs"],
                        a["n_blocks"] if kind == "overall" else "",
                        "complete",
                    ]
                )
    for c in d["sweep"]["cells"]:
        for metric in ("accept", "tpc"):
            w.writerow(
                [
                    "corrected_sweep",
                    d["sweep"]["target"],
                    "generated continuations",
                    c["drafter"],
                    c["k"],
                    c["temp"],
                    metric,
                    c[metric] if c[metric] is not None else "",
                    "",
                    "",
                    c["n_emit"],
                    c["n_runs"],
                    c["n_prompts"],
                    c["status"],
                ]
            )
    for comparison in ("spec_vs_target", "spec_vs_drafter_alone"):
        j = d["judge"][comparison]
        for metric in ("win_rate", "loss_rate", "wins", "ties", "losses"):
            w.writerow(
                [
                    "judge",
                    d["judge"]["meta"]["target"],
                    comparison,
                    "lora_dpo",
                    d["judge"]["meta"]["k"],
                    d["judge"]["meta"]["temperature"],
                    metric,
                    j[metric],
                    *(
                        j[metric.replace("_rate", "_ci95")]
                        if metric.endswith("_rate")
                        else ["", ""]
                    ),
                    "",
                    "",
                    j["n_prompts"],
                    "complete",
                ]
            )
    return buf.getvalue()


def current_path(path: str) -> str:
    """Display the renamed package without altering the historical evidence."""
    return path.replace("draftlab/", "speculative_agent_workloads/", 1)


def render(d: dict) -> str:
    template = (ROOT / "tools/presentation.html").read_text()
    a = d["alpha"]["lora"]
    evidence = []
    for key, commit in d["commits"].items():
        evidence.append(
            f'<details id="commit-{key}"><summary><code>{commit["short"]}</code> '
            f'{html.escape(commit["subject"])}</summary><p class="caption">'
            f"{commit['date']} · {html.escape(', '.join(map(current_path, commit['paths'])))}</p>"
            f"<pre><code>{html.escape(commit['diff'])}</code></pre></details>"
        )
    sources = "".join(
        f"<tr><td><code>{html.escape(current_path(s['path']))}</code></td><td>{html.escape(s['purpose'])}</td>"
        f'<td><code title="{s["sha256"]}">{s["sha256"][:12]}</code></td></tr>'
        for s in d["sources"]
    )
    overall = "".join(
        f'<tr><th scope="row">{LABELS[n]}</th><td>{v["overall"]["alpha"] * 100:.2f}%</td>'
        f"<td>{v['overall']['alpha_ci95'][0] * 100:.2f}–"
        f"{v['overall']['alpha_ci95'][1] * 100:.2f}%</td>"
        f"<td>{v['overall']['coverage'] * 100:.1f}%</td></tr>"
        for n, v in d["alpha"].items()
    )
    replacements = {
        "DATA": json.dumps(d, separators=(",", ":")).replace("<", "\\u003c"),
        "CSV": json.dumps(csv_data(d)).replace("<", "\\u003c"),
        "COMMITS": "\n".join(evidence),
        "COMMIT_INDEX": "".join(
            f"<tr><td><code>{c['short']}</code></td><td>{c['date']}</td>"
            f"<td>{html.escape(c['subject'])}</td></tr>"
            for c in d["commits"].values()
        ),
        "SOURCES": sources,
        "OVERALL_TABLE": overall,
        "DATE": d["generated_at"][:10],
        "REV": d["revision"][:7],
        "LORA_ALPHA": f"{a['overall']['alpha'] * 100:.1f}",
        "LORA_CI": (
            f"{a['overall']['alpha_ci95'][0] * 100:.2f}–{a['overall']['alpha_ci95'][1] * 100:.2f}"
        ),
        "RUNS": str(a["n_runs"]),
        "BLOCKS": f"{a['n_blocks']:,}",
        "TOKENS": f"{a['n_tokens']:,}",
        "HELDOUT_BLOCKS": f"{d['heldout']['blocks']:,}",
        "HELDOUT_TOKENS": f"{d['heldout']['tokens'] / 1e6:.2f}",
        "COMPLETE": str(d["sweep"]["statuses"].get("complete", 0)),
        "PARTIAL": str(d["sweep"]["statuses"].get("partial", 0)),
        "PENDING": str(d["sweep"]["statuses"].get("pending", 0)),
        "PROMPTS": str(d["sweep"]["n_prompts_per_cell"]),
        "SIDECARS": str(sum(".part-" in s["path"] for s in d["sources"])),
        "DPO_DELTA": f"{(d['alpha']['dpo']['overall']['alpha'] - a['overall']['alpha']) * 100:.3f}",
        "TINY_ALPHA": f"{d['alpha']['tiny']['overall']['alpha'] * 100:.2f}",
    }
    for key, value in replacements.items():
        template = template.replace("@@" + key + "@@", value)
    if re.search(r"@@[A-Z_]+@@", template):
        raise ValueError("Unresolved template marker")
    return template


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "report/index.html")
    ap.add_argument(
        "--publish", type=Path, help="Also copy the self-contained HTML to a local report folder"
    )
    args = ap.parse_args()
    d = build()
    page = render(d)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    (args.out.parent / "data.json").write_text(json.dumps(d, indent=2) + "\n")
    (args.out.parent / "metrics.csv").write_text(csv_data(d))
    if args.publish:
        args.publish.parent.mkdir(parents=True, exist_ok=True)
        args.publish.write_text(page)
    print(
        json.dumps(
            {
                "html": str(args.out),
                "published_copy": str(args.publish) if args.publish else None,
                "alpha_runs": d["alpha"]["lora"]["n_runs"],
                "sweep": d["sweep"]["statuses"],
                "sources": len(d["sources"]),
                "revision": d["revision"][:7],
            }
        )
    )


if __name__ == "__main__":
    main()
