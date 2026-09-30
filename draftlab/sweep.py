"""Ray sweep: measured acceptance and wall-clock speed of drafters, JAX engine.

Grid drafter x k x temperature over val prompts (per kind), target = 0.5B-instruct
JAX. Resumable at prompt granularity: workers append each finished prompt to a
sidecar JSONL as it completes; the driver folds sidecars into the output parquet
on every cell finish and on startup, so a kill loses at most one in-flight
prompt. Rows are per-prompt; aggregate (accept_rate, tpc, tok/s) on read.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import ray

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CKPT = ROOT / "ckpt"
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "corpus"

DRAFTERS = ["ngram", "tiny", "base", "lora", "lora_dpo"]
LORA_BASE = MODELS / "qwen2.5-0.5b"  # lora/dpo adapters were built on base, not instruct
ATTN_W = {"q_proj": "q_w", "k_proj": "k_w", "v_proj": "v_w", "o_proj": "o_w"}
MLP_W = {"gate_proj": "gate_w", "up_proj": "up_w", "down_proj": "down_w"}
RAW_COLS = ["drafter", "k", "temp", "kind", "run", "seq", "i", "prompt_len", "n_steps",
            "n_draft", "n_acc", "n_emit", "wall_s", "n_gen", "max_tokens", "target"]
# (run, seq) is NOT unique in the corpus (one run/seq can hold several blocks of
# one kind), so rows are keyed by the global prompt index i.
KEY_COLS = ["drafter", "k", "temp", "i"]


class NullDrafter:
    """No drafting: plain autoregressive target decoding (SpecDecoder k=0 path)."""

    def propose(self, ctx, k):
        return [], []

    def rollback(self, n_kept):
        pass


def load_prompts(split: str = "val", n_per_kind: int = 100, max_tokens: int = 128,
                 target_dir: str | Path = MODELS / "qwen2.5-0.5b-instruct",
                 include_text: bool = False) -> list[dict]:
    """First n_per_kind blocks (by run, seq) per kind, n_tokens <= max_tokens."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from draftlab.eval.alpha import kind_label

    t = pq.read_table(DATA / f"{split}.parquet").to_pydict()
    by_kind: dict[str, list[tuple[str, int, str]]] = {}
    for i in range(len(t["run"])):
        if t["n_tokens"][i] > max_tokens:
            continue
        by_kind.setdefault(kind_label(t["kind"][i], t["tool"][i]), []).append(
            (t["run"][i], t["seq"][i], t["text"][i]))
    tok = AutoTokenizer.from_pretrained(target_dir)
    prompts: list[dict] = []
    for kind in sorted(by_kind):
        for run, seq, text in sorted(by_kind[kind])[:n_per_kind]:
            d = {"kind": kind, "run": run, "seq": seq,
                 "ids": tok.encode(text, add_special_tokens=False)}
            if include_text:
                d["text"] = text
            prompts.append(d)
    for i, p in enumerate(prompts):
        p["i"] = i
    return prompts


def merge_adapter(params: dict, dcfg, adapter_dir: Path) -> dict:
    """Return params with W += (alpha/r) * B @ A added on the 7 target modules."""
    import jax.numpy as jnp
    from safetensors import safe_open

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    deltas: dict[tuple[int, str], np.ndarray] = {}
    with safe_open(adapter_dir / "adapter_model.safetensors", framework="pt") as f:
        keys = list(f.keys())
        for n in keys:
            if n.endswith("lora_B.weight"):
                parts = n.split(".")
                B = f.get_tensor(n).numpy()
                A = f.get_tensor(n.replace("lora_B", "lora_A")).numpy()
                deltas[(int(parts[4]), parts[6])] = scale * (B @ A)

    def put(params, i, group, key, arr):
        layers = list(params["layers"])
        layer = dict(layers[i])
        g = dict(layer[group])
        g[key] = g[key] + jnp.asarray(arr, g[key].dtype)
        layer[group] = g
        layers[i] = layer
        return {**params, "layers": layers}

    for (i, mod), d in deltas.items():
        if mod in ATTN_W:
            params = put(params, i, "attn", ATTN_W[mod], d)
        else:
            params = put(params, i, "mlp", MLP_W[mod], d)
    return params


def _sidecar_path(out: Path, dname: str, k: int, temp: float) -> Path:
    return out.with_name(f"{out.name}.part-{dname}-{k}-{temp:.1f}.jsonl")


@ray.remote
def run_cell(job: dict) -> list[dict]:
    """Run one (drafter, k, temp) cell: all prompts through SpecDecoder."""
    from draftlab.jaxlm import model as M
    from draftlab.jaxlm import spec as S

    tcfg, tparams = M.load_safetensors(job["target_dir"])
    dname, max_len = job["drafter"], job["max_len"]
    dcfg = dparams = None
    if dname in ("tiny", "base", "lora", "lora_dpo"):
        if dname == "tiny":
            dpath = CKPT / "tiny" / "export"
        elif dname == "base":
            dpath = job["target_dir"]
        else:
            dpath = LORA_BASE
        dcfg, dparams = M.load_safetensors(dpath)
        if dname == "lora":
            dparams = merge_adapter(dparams, dcfg, CKPT / "lora")
        elif dname == "lora_dpo":
            dparams = merge_adapter(dparams, dcfg, CKPT / "dpo")
    rows = []
    pf = _sidecar_path(Path(job["out"]), dname, job["k"], job["temp"]).open("a")
    try:
        for p in job["prompts"]:
            if dname == "none":
                drafter = NullDrafter()
            elif dname == "ngram":
                drafter = S.NGramDrafter(tcfg.vocab_size)
            else:
                drafter = S.ModelDrafter(dcfg, dparams, max_len)
            seed = job["seed"] + zlib.crc32(f"{p['run']}:{p['seq']}".encode())
            dec = S.SpecDecoder(tcfg, tparams, drafter, max_len, k=job["k"],
                                temperature=job["temp"], seed=seed)
            t0 = time.perf_counter()
            out, stats = dec.generate(p["ids"], job["n_gen"])
            wall_s = time.perf_counter() - t0
            rows.append({"drafter": dname, "k": job["k"], "temp": job["temp"], "kind": p["kind"],
                         "run": p["run"], "seq": p["seq"], "i": p["i"], "prompt_len": len(p["ids"]),
                         "n_steps": len(stats), "n_draft": sum(s.drafted for s in stats),
                         "n_acc": sum(s.accepted for s in stats), "n_emit": len(out),
                         "wall_s": wall_s, "n_gen": job["n_gen"], "max_tokens": job["max_tokens"],
                         "target": Path(job["target_dir"]).name})
            pf.write(json.dumps(rows[-1]) + "\n")
            pf.flush()
    finally:
        pf.close()
    return rows


def to_argv(a: dict) -> list[str]:
    out: list[str] = []
    for k, v in a.items():
        if isinstance(v, list):
            out += ["--" + k.replace("_", "-"), *[str(x) for x in v]]
        else:
            out += ["--" + k.replace("_", "-"), str(v)]
    return out


def _fold_sidecars(out: Path, old: pd.DataFrame | None) -> pd.DataFrame | None:
    """Fold leftover sidecar rows (a killed run) into the parquet, then remove them."""
    parts = sorted(out.parent.glob(out.name + ".part-*.jsonl"))
    if not parts:
        return old
    rows = [json.loads(line) for part in parts
            for line in part.read_text().splitlines() if line.strip()]
    if not rows:
        for part in parts:
            part.unlink()
        return old
    old = (pd.DataFrame(rows, columns=RAW_COLS) if old is None else
           pd.concat([old, pd.DataFrame(rows, columns=RAW_COLS)], ignore_index=True))
    old = old.drop_duplicates(subset=KEY_COLS, keep="last").reset_index(drop=True)
    old.to_parquet(out, index=False)
    for part in parts:
        part.unlink()
    print(f"sweep: recovered {len(rows)} sidecar rows into {out}")
    return old


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m draftlab.sweep")
    ap.add_argument("--drafters", nargs="+", default=DRAFTERS, choices=DRAFTERS)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--temps", nargs="+", type=float, default=[0.0, 0.7])
    ap.add_argument("--n-per-kind", type=int, default=100)
    ap.add_argument("--n-gen", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--num-cpus", type=int, default=16)
    ap.add_argument("--cpus-per-cell", type=int, default=4)
    ap.add_argument("--max-cells", type=int, default=0, help="cap cells this run (0 = all)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--split", default="val")
    ap.add_argument("--target", default=str(MODELS / "qwen2.5-0.5b-instruct"))
    ap.add_argument("--out", default=str(RESULTS / "sweep.parquet"))
    args = ap.parse_args(argv)

    prompts = load_prompts(args.split, args.n_per_kind, args.max_tokens, args.target)
    max_len = max(len(p["ids"]) for p in prompts) + args.n_gen + 8
    jobs = []
    for di, d in enumerate(args.drafters):
        for ki, k in enumerate(args.ks):
            for ti, temp in enumerate(args.temps):
                jobs.append({"drafter": d, "k": int(k), "temp": float(temp),
                             "seed": args.seed + 100 * di + 10 * ki + ti})
    for ti, temp in enumerate(args.temps):  # no-drafter baseline per temperature
        jobs.append({"drafter": "none", "k": 0, "temp": float(temp), "seed": 90000 + ti})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    old = None
    if out.exists():
        old = pd.read_parquet(out)
    old = _fold_sidecars(out, old)
    # prompts already measured, per cell (resume granularity)
    done: dict[tuple[str, int, float], set[int]] = {}
    if old is not None:
        for (d, k, t), g in old.groupby(["drafter", "k", "temp"], sort=False):
            done[(d, int(k), float(t))] = set(g["i"].astype(int))
    todo = []
    for j in jobs:
        have = done.get((j["drafter"], j["k"], j["temp"]), set())
        if not have:
            todo.append(j)
        else:
            pend = [p for p in prompts if p["i"] not in have]
            if pend:  # partially done cell: rerun only the missing prompts
                todo.append(dict(j, prompts=pend))
    if args.max_cells:
        todo = todo[: args.max_cells]
    if not todo:
        print(f"sweep: all {len(jobs)} cells already in {out}")
        return {"ran": []}
    n_prompts = sum(len(j.get("prompts", prompts)) for j in todo)
    print(f"sweep: {len(todo)}/{len(jobs)} cells, {n_prompts} prompts, "
          f"n_gen={args.n_gen}, target={Path(args.target).name}", flush=True)
    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True,
             runtime_env={"env_vars": {"OMP_NUM_THREADS": str(args.cpus_per_cell),
                                       "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=true"}})
    try:
        refs = []
        for j in todo:
            jj = dict(j, prompts=j.get("prompts", prompts), target_dir=str(args.target),
                      n_gen=args.n_gen, max_len=max_len, max_tokens=args.max_tokens,
                      out=str(out))
            refs.append((j, run_cell.options(num_cpus=args.cpus_per_cell).remote(jj)))
        new_rows: list[dict] = []
        for j, ref in refs:  # in order: print a line per finished cell
            new_rows.extend(ray.get(ref))
            # incremental write: a kill loses at most the running cell's
            # in-flight prompt (the rest is already in the sidecar), and
            # resume (skips by (drafter, k, temp, run, seq)) works across runs
            part = _sidecar_path(out, j["drafter"], j["k"], j["temp"])
            if part.exists():
                part.unlink()
            df = pd.DataFrame(new_rows, columns=RAW_COLS)
            if old is not None:
                df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(subset=KEY_COLS, keep="last").reset_index(drop=True)
            df.to_parquet(out, index=False)
            old = df
            print(f"  cell {j['drafter']:8s} k={j['k']} T={j['temp']}: done", flush=True)
    finally:
        ray.shutdown()
    cells = set(zip(df["drafter"], df["k"], df["temp"], strict=True))
    print(f"sweep: wrote {out} ({len(df)} rows, {len(cells)} cells)")
    return {"ran": [f"{j['drafter']}/{j['k']}/{j['temp']}" for j in todo]}


if __name__ == "__main__":
    main()
