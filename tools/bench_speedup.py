"""Head-to-head CPU speedup: each config vs `none` on the same prompts, one process,
nothing else running. A warm-up pass compiles every shape; then `--reps` passes
run the configs interleaved (none, A, B, ..., none, A, B, ...), so drift hits all
configs alike. Speedup per rep = none wall / config wall over the same prompts.

    python tools/bench_speedup.py            # -> results/speedup_bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from draftlab.jaxlm import model as M  # noqa: E402
from draftlab.jaxlm import spec as S  # noqa: E402
from draftlab.sweep import (  # noqa: E402
    CKPT,
    LORA_BASE,
    MODELS,
    RESULTS,
    load_prompts,
    merge_adapter,
)

# (target, drafter, k, temperature)
CONFIGS = [
    ("qwen2.5-0.5b-instruct", "ngram", 8, 0.0),
    ("qwen2.5-1.5b-instruct", "ngram", 8, 0.0),
    ("qwen2.5-1.5b-instruct", "lora", 2, 0.0),
    ("qwen2.5-1.5b-instruct", "lora", 4, 0.0),
    ("qwen2.5-1.5b-instruct", "lora", 4, 0.7),
]


def _drafter(name, tcfg, max_len, cache):
    if name == "none":
        return S.NullDrafter()
    if name == "ngram":
        return S.NGramDrafter(tcfg.vocab_size)
    if name not in cache:
        dcfg, dp = M.load_safetensors(LORA_BASE)
        cache[name] = (dcfg, merge_adapter(dp, dcfg, CKPT / ("lora" if name == "lora" else "dpo")))
    dcfg, dp = cache[name]
    return S.ModelDrafter(dcfg, dp, max_len)


def _pass(tcfg, tp, prompts, drafter, k, temp, n_gen, max_len, cache):
    wall = drafted = acc = steps = toks = 0
    for p in prompts:
        dec = S.SpecDecoder(tcfg, tp, _drafter(drafter, tcfg, max_len, cache), max_len,
                            k=k if drafter != "none" else 0, temperature=temp, seed=p["i"])
        t0 = time.perf_counter()
        out, st = dec.generate(p["ids"], n_gen)
        wall += time.perf_counter() - t0
        toks += len(out)
        steps += len(st)
        drafted += sum(s.drafted for s in st)
        acc += sum(s.accepted for s in st)
    return {"wall_s": wall, "tok_s": toks / wall, "tokens_per_step": toks / steps,
            "accept": acc / drafted if drafted else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-kind", type=int, default=3)
    ap.add_argument("--n-gen", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=str(RESULTS / "speedup_bench.json"))
    a = ap.parse_args()
    out: dict = {"n_gen": a.n_gen, "reps": a.reps, "split": "heldout", "targets": {}}
    for target in dict.fromkeys(c[0] for c in CONFIGS):
        tdir = MODELS / target
        prompts = load_prompts("heldout", a.n_per_kind, 128, tdir)
        max_len = max(len(p["ids"]) for p in prompts) + a.n_gen + 16
        tcfg, tp = M.load_safetensors(tdir)
        cache: dict = {}
        cfgs = [c for c in CONFIGS if c[0] == target]
        temps = sorted({c[3] for c in cfgs})
        runs = [("none", 0, t) for t in temps] + [(d, k, t) for _, d, k, t in cfgs]
        print(f"{target}: {len(prompts)} prompts, {len(runs)} configs", flush=True)
        for d, k, t in runs:  # warm-up: compile every prompt/step shape
            _pass(tcfg, tp, prompts, d, k, t, a.n_gen, max_len, cache)
        reps = {f"{d}-k{k}-T{t}": [] for d, k, t in runs}
        for r in range(a.reps):
            for d, k, t in runs:
                res = _pass(tcfg, tp, prompts, d, k, t, a.n_gen, max_len, cache)
                reps[f"{d}-k{k}-T{t}"].append(res)
                print(f"  rep {r} {d} k={k} T={t}: {res['tok_s']:.2f} tok/s", flush=True)
        rows = {}
        for d, k, t in runs:
            key = f"{d}-k{k}-T{t}"
            base = reps[f"none-k0-T{t}"]
            sp = [b["wall_s"] / x["wall_s"] for b, x in zip(base, reps[key], strict=True)]
            rows[key] = {
                "drafter": d, "k": k, "temp": t,
                "tok_s": float(np.mean([x["tok_s"] for x in reps[key]])),
                "speedup": float(np.mean(sp)), "speedup_range": [float(min(sp)), float(max(sp))],
                "accept": reps[key][0]["accept"],
                "tokens_per_step": reps[key][0]["tokens_per_step"],
            }
        out["targets"][target] = {"n_prompts": len(prompts), "configs": rows}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
