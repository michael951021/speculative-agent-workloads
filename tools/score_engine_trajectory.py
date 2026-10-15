"""Engine-vs-metric check on the SAME text (known answer (b), CLAUDE.md).

tools/compare_sweep_alpha.py compares the sweep against the alpha metric on
corpus blocks; but the engine scores post-prompt positions on the target's own
continuation, while the corpus metric scores within-block positions. At T=0
every emitted token is a target-greedy token, so the spec trajectory equals
the `none` baseline trajectory (deterministic, same seed). This tool:

1. regenerates the none trajectory for all sweep prompts (SpecDecoder,
   NullDrafter, k=0, T=0, seed 90000 + crc32(run:seq) as in speculative_agent_workloads.sweep);
2. for each drafter, runs the alpha-metric drafter (HF, f32) on
   block = prompt_ids + gen_ids and counts argmax-q == emitted-token at exactly
   the draft positions the engine used (row L+j-1 for draft j, 1..n_steps,
   since row r of the empty-ctx dists() output predicts ids[r]);
3. compares per-drafter accept/tpc against results/sweep.parquet, with a
   bootstrap CI over prompts. base == target, so base is 1.0 exactly.

Usage: python tools/score_engine_trajectory.py [--drafters lora lora_dpo]
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "ckpt"
MODELS = ROOT / "models"
N_GEN = 32  # == speculative_agent_workloads.sweep.N_GEN


def gen_trajectories(prompts: list[dict]) -> dict[int, list[int]]:
    """Regenerate the T=0 greedy trajectory (== none baseline) per prompt."""
    from speculative_agent_workloads.jaxlm import model as M
    from speculative_agent_workloads.jaxlm import spec as S
    from speculative_agent_workloads.sweep import NullDrafter

    target = MODELS / "qwen2.5-0.5b-instruct"
    max_len = max(len(p["ids"]) for p in prompts) + N_GEN + 8
    tcfg, tparams = M.load_safetensors(target)
    traj: dict[int, list[int]] = {}
    for n, p in enumerate(prompts):
        # T=0 k=0: pure argmax, seed irrelevant; fresh decoder per prompt
        # exactly as speculative_agent_workloads.sweep.run_cell does.
        seed = 90000 + zlib.crc32(f"{p['run']}:{p['seq']}".encode())
        dec = S.SpecDecoder(tcfg, tparams, NullDrafter(), max_len, k=0, temperature=0.0, seed=seed)
        out, _ = dec.generate(p["ids"], N_GEN)
        traj[p["i"]] = out
        if (n + 1) % 50 == 0:
            print(f"  generated {n + 1}/{len(prompts)}", flush=True)
    return traj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafters", nargs="+", default=["lora", "lora_dpo"])
    args = ap.parse_args()

    from speculative_agent_workloads.eval.alpha import HFDrafter
    from speculative_agent_workloads.sweep import load_prompts

    df = pd.read_parquet(ROOT / "results" / "sweep.parquet")
    sub = df[(df["temp"] == 0.0) & (df["k"] == 1)]
    prompts = load_prompts()
    assert [p["i"] for p in prompts] == sorted(sub["i"].unique()), "prompt sets differ"

    cache = ROOT / "results" / "engine_traj_none.json"
    if cache.exists():
        traj = {int(k): v for k, v in json.loads(cache.read_text()).items()}
        print(f"loaded trajectories from {cache.name}")
    else:
        print("regenerating T=0 trajectories (none baseline, deterministic)")
        traj = gen_trajectories(prompts)
        cache.write_text(json.dumps({str(k): v for k, v in traj.items()}))
        print(f"wrote {cache}")

    ok = True
    res: dict = {
        "metric": "greedy acceptance of HF drafter at the engine's k=1 T=0 draft positions "
        "(simulated from per-position hits: a hit skips the bonus position) on "
        "prompt+none-trajectory (== spec trajectory); step counts must equal the sweep's; "
        "bootstrap ci95 over prompts (mean of per-prompt accept)",
        "target": "qwen2.5-0.5b-instruct",
        "traj_cache": str(cache.relative_to(ROOT)),
    }
    print(
        f"{'drafter':10s} {'meas acc':>9s} {'traj acc':>9s} {'traj ci95':>22s} "
        f"{'meas tpc':>9s} {'verdict':>8s}"
    )
    for d in ["base"] + args.drafters:
        s = sub[sub["drafter"] == d].set_index("i")
        meas = s["n_acc"].sum() / s["n_draft"].sum()
        tpc = s["n_emit"].sum() / s["n_steps"].sum()
        if d == "base":
            # drafter == target: argmax q == argmax p by construction
            per = {i: (int(r.n_acc), int(r.n_draft)) for i, r in s.iterrows()}
            vals = np.array([a for a, _ in per.values()])
            lo = hi = float(vals.mean())
        else:
            # sweep drafter name -> ckpt dir (dpo adapter lives in ckpt/dpo)
            drafter = HFDrafter(CKPT / {"lora_dpo": "dpo"}.get(d, d))
            per: dict[int, tuple[int, int]] = {}
            for p in prompts:
                i = p["i"]
                r = s.loc[i]
                ids = p["ids"] + traj[i]
                q, valid = drafter.dists([], ids)
                L = len(p["ids"])
                gen = np.asarray(traj[i])
                # row L+j predicts gen[j] (row r predicts ids[r+1]; empty ctx: q[L+j] -> gen[j])
                hit = (q[L : L + len(gen)].argmax(axis=-1) == gen) & valid[L : L + len(gen)]
                # k=1 at T=0: each step drafts at pos; a hit emits draft + bonus (pos += 2), a miss
                # emits the target token (pos += 1). Draft positions are NOT the first n_steps ones:
                # the position after every accepted draft is never drafted.
                pos = m = steps = 0
                while pos < len(gen):
                    steps += 1
                    m += int(hit[pos])
                    pos += 2 if hit[pos] else 1
                assert steps == int(r["n_steps"]), (i, steps, int(r["n_steps"]))
                per[i] = (m, steps)
            vals = np.array([a / dr for a, dr in per.values()])
            rng = np.random.default_rng(0)
            boot = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(2000)]
            lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
            traj_acc = sum(a for a, _ in per.values()) / sum(dr for _, dr in per.values())
        in_ci = d == "base" or (lo <= meas <= hi)
        ok &= in_ci
        tstr = (
            f"{traj_acc:9.4f} [{lo:.4f}, {hi:.4f}]"
            if d != "base"
            else f"{'1.0000':>9s} [1.0000, 1.0000]"
        )
        print(f"{d:10s} {meas:9.4f} {tstr} {tpc:9.4f} {'PASS' if in_ci else 'FAIL':>8s}")
        res[d] = {
            "meas_acc": meas,
            "meas_tpc": tpc,
            "traj_acc": 1.0 if d == "base" else traj_acc,
            "traj_ci95": [1.0, 1.0] if d == "base" else [lo, hi],
            "in_ci": bool(in_ci),
        }
    print("KNOWN ANSWER (b) on engine trajectory:", "PASS" if ok else "FAIL")
    out = ROOT / "results" / "engine_traj_metric.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
