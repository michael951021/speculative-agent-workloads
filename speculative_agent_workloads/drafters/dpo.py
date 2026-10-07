"""Acceptance-aware DPO: the verifier's own accept/reject signal as preference.

- pairs: batched T=1 spec run — the 1.5B target scores the LoRA drafter's greedy
  draft; accept iff u*q[d] < p[d] (same test as jaxlm/spec.py), else the drafter's
  token d is rejected and the committed token is resampled from max(0, p-q).
  Pair = (ctx, chosen=accepted/resampled token, rejected=d). No-op resamples
  (resample == d) carry no signal and are excluded.
- train: TRL DPOTrainer (beta 0.1) on the LoRA adapter; ref = adapter disabled.
  Sequences are pre-tokenized: prompt = last --prompt-len ctx tokens, completion 1.
- report: results/dpo.json — before/after per-kind alpha (from results/alpha/
  lora.json and dpo.json), pair stats, DPO curve, train-pair margin before/after.

CLI: python -m speculative_agent_workloads.drafters.dpo [--pairs 2000 ...]
     after-eval: python -m speculative_agent_workloads.eval.alpha --drafter hf
                  --draft-model ckpt/dpo
                 --out results/alpha/dpo.json   (then rerun with --report-only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from trl import DPOConfig, DPOTrainer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RESULTS = _ROOT / "results" / "alpha"


def accept_step(p: np.ndarray, q: np.ndarray, d: int, u: float, rng: np.random.Generator) -> int:
    """One T=1 spec step (Leviathan 2023): accept d iff u*q[d] < p[d], else
    resample from the normalized residual max(0, p-q). Returns committed token."""
    if u * q[d] < p[d]:
        return int(d)
    r = np.maximum(0.0, p - q)
    z = r.sum()
    if z <= 0:
        return int(np.argmax(p))
    return int(rng.choice(len(p), p=r / z))


def pairs_from_spec_log(log: list[dict], seed: int = 0) -> tuple[list[dict], dict]:
    """Replay a recorded T=1 spec log (steps with ctx, draft, u, p, q) and return
    (pairs, stats). Deterministic in `seed`. Pairs exclude no-op resamples."""
    rng = np.random.default_rng(seed)
    pairs: list[dict] = []
    n_acc = 0
    for st in log:
        d = st["draft"]
        b = accept_step(np.asarray(st["p"]), np.asarray(st["q"]), d, st["u"], rng)
        if b == d:
            n_acc += 1
        else:
            pairs.append({"ctx": list(st["ctx"]), "chosen": b, "rejected": d})
    stats = {"n_steps": len(log), "n_accepted": n_acc, "n_pairs": len(pairs)}
    return pairs, stats


class PreTokDPOTrainer(DPOTrainer):
    """DPOTrainer over pre-tokenized (ctx, chosen, rejected) rows: skips TRL's
    text/EOS/tokenize pipeline, feeds the collator its prompt_ids/chosen_ids/
    rejected_ids columns directly (single-token completions)."""

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        def tok(ex):
            return {
                "prompt_ids": ex["ctx"],
                "chosen_ids": [ex["chosen"]],
                "rejected_ids": [ex["rejected"]],
            }

        return dataset.map(tok)


def pair_margin(model, ctx: list[int], c: int, r: int) -> float:
    """log pi(c|ctx) - log pi(r|ctx) from one forward (row len(ctx)-1 predicts
    the next token)."""
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ctx + [c]])).logits[0, len(ctx) - 1]
        ls = F.log_softmax(logits.float(), dim=-1)
    return float(ls[int(c)] - ls[int(r)])


def _seeds(n: int, seed_len: int, seed: int) -> list[list[int]]:
    """First `seed_len` tokens of shuffled train blocks long enough to seed."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    t = pq.read_table(_ROOT / "data" / "corpus" / "train.parquet").to_pydict()
    perm = torch.randperm(len(t["run"]), generator=torch.Generator().manual_seed(seed)).tolist()
    tok = AutoTokenizer.from_pretrained(str(_ROOT / "models" / "qwen2.5-0.5b"))
    seeds: list[list[int]] = []
    for i in perm:
        ids = tok.encode(t["text"][i], add_special_tokens=False)
        if len(ids) >= seed_len + 8:
            seeds.append(ids[:seed_len])
        if len(seeds) >= n:
            break
    if len(seeds) < n:
        raise SystemExit(f"only {len(seeds)} train blocks >= {seed_len} tokens; need {n}")
    return seeds


def collect_pairs(args) -> tuple[list[dict], dict]:
    """Batched T=1 spec decode of `args.traj` trajectories per round until
    `args.pairs` pairs. Deterministic in args.seed."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    t_rng = torch.Generator().manual_seed(args.seed + 1)
    u_rng = np.random.default_rng(args.seed + 2)
    n_traj = int(np.ceil(args.pairs * 4 / args.decode_len)) + args.traj  # >=25% reject margin
    seeds = _seeds(n_traj, args.seed_len, args.seed)

    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16)
    drafter = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16), args.adapter
    )
    target.eval()
    drafter.eval()
    B = args.traj
    pairs: list[dict] = []
    n_acc = n_rej = 0
    t0 = time.perf_counter()
    r = 0
    while len(pairs) < args.pairs and r * B < len(seeds):
        B = min(B, len(seeds) - r * B)  # last round may be partial
        arange = torch.arange(B)
        ids = torch.tensor(seeds[r * B : r * B + B])
        hist = [row.tolist() for row in ids]
        with torch.no_grad():
            tc = target(input_ids=ids).past_key_values
            dc = drafter(input_ids=ids).past_key_values
            nxt = ids[:, -1:]
        for _ in range(args.decode_len):
            with torch.no_grad():
                po = target(input_ids=nxt, past_key_values=tc, use_cache=True)
                tc = po.past_key_values
                p = F.softmax(po.logits[:, -1].float(), dim=-1)
                do = drafter(input_ids=nxt, past_key_values=dc, use_cache=True)
                dc = do.past_key_values
                q = F.softmax(do.logits[:, -1].float(), dim=-1)
            d = q.argmax(dim=-1)
            u = torch.tensor(u_rng.uniform(size=B), dtype=torch.float32)
            acc = u * q[arange, d] < p[arange, d]
            res = torch.clamp(p - q, min=0.0)
            res = res / res.sum(dim=-1, keepdim=True)
            b = torch.multinomial(res, 1, generator=t_rng).squeeze(-1)
            committed = torch.where(acc, d, b)
            rej = ~acc
            for i in torch.nonzero(rej).flatten().tolist():
                if committed[i] != d[i]:  # exclude no-op resamples
                    pairs.append(
                        {
                            "ctx": hist[i][-args.prompt_len :],
                            "chosen": int(committed[i]),
                            "rejected": int(d[i]),
                        }
                    )
            n_acc += int(acc.sum())
            n_rej += int(rej.sum())
            for i in range(B):
                hist[i].append(int(committed[i]))
            nxt = committed.unsqueeze(1)
            if len(pairs) >= args.pairs:
                break
        dt = time.perf_counter() - t0
        print(
            f"pairs: round {r} done pairs={len(pairs)}/{args.pairs} "
            f"acc={n_acc} rej={n_rej} ({dt:.0f}s)",
            flush=True,
        )
        r += 1
    pairs = pairs[: args.pairs]
    stats = {
        "n_pairs": len(pairs),
        "n_steps": n_acc + n_rej,
        "n_accepted": n_acc,
        "n_rejected": n_rej,
        "accept_rate": round(n_acc / max(n_acc + n_rej, 1), 4),
        "traj": B,
        "seed_len": args.seed_len,
        "decode_len": args.decode_len,
        "target": Path(args.target).name,
        "drafter": Path(args.adapter).name,
        "dt_sec": round(dt, 1),
    }
    return pairs, stats


def _loss_of(h: dict) -> float:
    return float(h.get("loss", h.get("train_loss")))


def train_dpo(args, pairs: list[dict]) -> dict:
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    npairs = len(pairs)
    idx = np.linspace(0, npairs - 1, min(args.margin_sample, npairs), dtype=int)
    sub = [pairs[i] for i in idx]
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    pm = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    tok = AutoTokenizer.from_pretrained(args.base)

    margin_before = float(
        np.mean([pair_margin(pm, p["ctx"], p["chosen"], p["rejected"]) for p in sub])
    )
    ds = Dataset.from_list(pairs)
    cfg = DPOConfig(
        output_dir=args.out,
        beta=args.beta,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        max_length=args.prompt_len + 1,
        num_train_epochs=1,
        use_cpu=True,
        report_to=[],
        save_strategy="no",
        optim="adamw_torch",
        logging_steps=25,
        max_grad_norm=1.0,
        seed=args.seed,
        bf16=not args.f32,
        fp16=False,
        gradient_checkpointing=False,
    )
    tr = PreTokDPOTrainer(
        model=pm, ref_model=None, args=cfg, train_dataset=ds, processing_class=tok
    )
    t0 = time.perf_counter()
    tr.train()
    dt = time.perf_counter() - t0
    hist = [h for h in tr.state.log_history if "loss" in h or "train_loss" in h]
    margin_after = float(
        np.mean([pair_margin(pm, p["ctx"], p["chosen"], p["rejected"]) for p in sub])
    )

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pm.peft_config["default"].base_model_name_or_path = str(Path(args.base).resolve())
    pm.save_pretrained(out)
    info = {
        "beta": args.beta,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "prompt_len": args.prompt_len,
        "n_pairs": npairs,
        "loss_first": round(_loss_of(hist[0]), 5),
        "loss_last": round(_loss_of(hist[-1]), 5),
        "loss_curve": [[h.get("epoch"), round(_loss_of(h), 5)] for h in hist],
        "margin_subsample": len(sub),
        "margin_before": round(margin_before, 5),
        "margin_after": round(margin_after, 5),
        "dt_sec": round(dt, 1),
        "adapter_after": out.relative_to(_ROOT).as_posix(),
    }
    print(
        f"dpo: loss {info['loss_first']} -> {info['loss_last']} "
        f"margin {margin_before:.4f} -> {margin_after:.4f} ({dt:.0f}s)",
        flush=True,
    )
    return info


def _alpha_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())

    def keep(s):
        return {"n_tokens": s["n_tokens"], "alpha": s["alpha"], "alpha_ci95": s["alpha_ci95"]}

    return {
        "overall": keep(d["overall"]),
        "by_kind": {k: keep(v) for k, v in d["by_kind"].items()},
        "source": str(path.relative_to(_ROOT)),
    }


def write_report(args, pair_stats: dict | None, dpo_info: dict | None) -> None:
    report = {
        "metric": "acceptance-aware DPO: pairs (ctx, chosen=accepted/resampled, "
        "rejected=drafter draft) from T=1 spec runs of the target with the "
        "LoRA drafter; TRL DPO on the adapter. alpha = sum_x min(p(x),q(x)) "
        "per position vs the same target on val blocks (results/alpha).",
        "pairs": pair_stats,
        "dpo": dpo_info,
        "adapter_before": Path(args.adapter).as_posix(),
        "alpha": {
            "before": _alpha_summary(RESULTS / "lora.json"),
            "after": _alpha_summary(RESULTS / "dpo.json"),
        },
        "expected_alpha_range": "after overall alpha in [0.70, 0.78] (before 0.7306); "
        "|delta| > 0.1 is a bug until shown otherwise",
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m speculative_agent_workloads.drafters.dpo")
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--adapter", default="ckpt/lora")
    ap.add_argument("--target", default=str(_ROOT / "models" / "qwen2.5-1.5b-instruct"))
    ap.add_argument("--base", default=str(_ROOT / "models" / "qwen2.5-0.5b"))
    ap.add_argument("--traj", type=int, default=16)
    ap.add_argument("--seed-len", type=int, default=192)
    ap.add_argument("--decode-len", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16, help="DPO pairs per step")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument(
        "--f32", action="store_true", help="f32 DPO training (default bf16 on CPU is ~2x faster)"
    )
    ap.add_argument("--margin-sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ckpt/dpo")
    ap.add_argument("--report", default=str(_ROOT / "results" / "dpo.json"))
    ap.add_argument("--pairs-cache", default=str(_ROOT / "data" / "pairs" / "dpo.jsonl"))
    ap.add_argument("--skip-pairs", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)

    pairs: list[dict] | None = None
    pair_stats: dict | None = None
    dpo_info: dict | None = None
    if not args.report_only:
        cache = Path(args.pairs_cache)
        if cache.exists():
            pairs = [json.loads(line) for line in cache.read_text().splitlines() if line]
            pair_stats = (
                json.loads((cache.with_suffix(".stats.json")).read_text())
                if cache.with_suffix(".stats.json").exists()
                else {"n_pairs": len(pairs)}
            )
            print(f"pairs: loaded {len(pairs)} from cache", flush=True)
        elif not args.skip_pairs:
            pairs, pair_stats = collect_pairs(args)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
            cache.with_suffix(".stats.json").write_text(json.dumps(pair_stats, indent=1))
        if not args.skip_train and pairs is not None:
            dpo_info = train_dpo(args, pairs)
    write_report(args, pair_stats, dpo_info)


if __name__ == "__main__":
    main()
