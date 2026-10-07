"""Alpha metric: one cheap, exact drafter score, no full generations.

At every position of a block, with context = the run's recent preceding blocks
(within a token budget) + the block's own prefix (teacher-forced):
    alpha  = sum_x min(p(x), q(x)) = 1 - TV(p, q)   (p target, q drafter, full vocab)
    greedy = [argmax q == argmax p]          (real spec-accept for the target)
    greedy_corpus = [argmax q == actual token]  (what results/ngram.json measures)
"Batched" = one teacher-forced forward per model per block (all positions at once),
top-k-free. Alignment: logits row i predicts token i+1, so ids[j] is scored by row
len(ctx)+j-1; with empty ctx there is no row for ids[0] and position 0 is invalid
(valid[0] = False; no fabricated BOS). Blocks are selected round-robin across runs;
per-kind aggregation uses bootstrap CIs over runs (units = runs, so no run leaks).
Expected tokens per target call for k drafts: (1 - a^(k+1))/(1 - a); speedup divides
by 1 + k*c, c = measured draft/target forward cost ratio.

CLI: python -m speculative_agent_workloads.eval.alpha [--drafter ngram|jax|hf] [--draft-model PATH]
     [--blocks 400] [--max-tokens 256] [--ctx-tokens 1024] [--split val]
Writes results/alpha/<drafter>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "data" / "corpus"
TARGET_DIR = ROOT / "models" / "qwen2.5-1.5b-instruct"  # target T (TODO.md model roles)
RESULTS_DIR = ROOT / "results" / "alpha"

K_VALUES = tuple(range(1, 9))


def kind_label(kind: str, tool: str | None) -> str:
    """CLAUDE.md kind: tool_use blocks carry their tool name."""
    return f"tool_use:{tool}" if kind == "tool_use" and tool else kind


def alpha_and_greedy(
    p: np.ndarray, q: np.ndarray, valid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-position alpha and greedy match. p, q: (T, V) distributions; valid: (T,)
    bool marking positions where the drafter actually proposed (None = all)."""
    a = _alpha_sum(p, q)
    g = p.argmax(axis=-1) == q.argmax(axis=-1)
    if valid is not None:
        a = np.where(valid, a, 0.0)
        g = g & valid
    return a, g


def _alpha_sum(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row sum min(p, q). Fused Triton kernel on CUDA (log -> fused_alpha);
    numpy (float64) otherwise / on failure."""
    try:
        import torch

        if torch.cuda.is_available() and p.shape[0] > 0:
            from speculative_agent_workloads.kernels import verify as _vk

            lp = torch.log(torch.as_tensor(p, device="cuda", dtype=torch.float32)).contiguous()
            lq = torch.log(torch.as_tensor(q, device="cuda", dtype=torch.float32)).contiguous()
            return _vk.fused_alpha(lp, lq).to("cpu", torch.float64).numpy()
    except Exception:
        pass
    return np.minimum(p, q).sum(axis=-1)


def expected_tokens_per_call(alpha: float | np.ndarray, k: int) -> float | np.ndarray:
    """(1 - alpha^(k+1)) / (1 - alpha) (Leviathan et al. 2023); k+1 as alpha -> 1."""
    a = np.asarray(alpha, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(a >= 1.0 - 1e-12, k + 1.0, (1.0 - a ** (k + 1)) / (1.0 - a))
    return float(out) if out.ndim == 0 else out


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x, dtype=np.float32)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


class NGramDrafter:
    """One-hot drafts from NGramMatcher over ctx+block (no proposal -> zero row +
    valid=False)."""

    def __init__(self, vocab_size: int, max_n: int = 4) -> None:
        from speculative_agent_workloads.drafters.ngram import NGramMatcher

        self.NGramMatcher = NGramMatcher
        self.vocab_size = vocab_size
        self.max_n = max_n

    def dists(self, ctx: list[int], ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        # fresh matcher per call: the n-gram dict must be built only from this
        # call's own prefix (same semantics as the baseline's per-block matcher)
        props = self.NGramMatcher(self.max_n).proposals(ctx + ids)
        q = np.zeros((len(ids), self.vocab_size), dtype=np.float32)
        valid = np.zeros(len(ids), dtype=bool)
        nz = [(t, tok) for t, tok in enumerate(props[len(ctx) :]) if tok is not None]
        if nz:
            t = np.array([x[0] for x in nz])
            tok = np.array([x[1] for x in nz])
            q[t, tok] = 1.0
            valid[t] = True
        return q, valid


class JaxModel:
    """Full-vocab teacher-forced distributions from a JAX Qwen2 checkpoint
    (config.json + safetensors). One jitted prefill per (ctx, block)."""

    def __init__(self, cfg, params, max_len: int) -> None:
        from speculative_agent_workloads.jaxlm import model as M

        self.M = M
        self.cfg, self.params, self.max_len = cfg, params, max_len

    @classmethod
    def from_path(cls, path: str | Path, max_len: int) -> JaxModel:
        from speculative_agent_workloads.jaxlm import model as M

        cfg, params = M.load_safetensors(str(path))
        return cls(cfg, params, max_len)

    def dists(self, ctx: list[int], ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        # Pad to max_len: real-position logits are unchanged (causal mask), and one
        # fixed shape means one jit compile per run.
        full = ctx + ids
        if len(full) < self.max_len:
            full = full + [0] * (self.max_len - len(full))
        logits, _ = self.M.prefill(self.cfg, self.params, full, self.max_len)
        n = len(ids)
        # Row i predicts token i+1: ids[j] is scored by row len(ctx)+j-1 (slice
        # before softmax/transfer). Empty ctx: no row predicts ids[0] -> dummy
        # row 0, position 0 invalid (we do not fabricate a BOS).
        L = np.asarray(logits)
        valid = np.ones(n, dtype=bool)
        if ctx:
            rows = L[len(ctx) - 1 : len(ctx) + n - 1]
        else:
            valid[0] = False
            rows = L[: n - 1]
            rows = np.vstack([L[:1], rows]) if n > 1 else L[:1]
        return _softmax(rows), valid


class HFDrafter:
    """Full-vocab teacher-forced distributions from an HF causal-LM checkpoint or
    PEFT adapter dir (adapter_config.json -> base model + PeftModel)."""

    def __init__(self, path: str | Path) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        p = Path(path)
        if (p / "adapter_config.json").exists():
            from peft import PeftModel

            base = json.loads((p / "adapter_config.json").read_text())["base_model_name_or_path"]
            self.model = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32), p
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32)
        self.torch = torch
        self.model.eval()

    def dists(self, ctx: list[int], ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        n = len(ids)
        with self.torch.no_grad():
            logits = self.model(input_ids=self.torch.tensor([ctx + ids])).logits[0]
        # Same alignment as JaxModel: ids[j] is scored by row len(ctx)+j-1;
        # empty ctx -> dummy row 0, position 0 invalid.
        valid = np.ones(n, dtype=bool)
        if ctx:
            rows = logits[len(ctx) - 1 : len(ctx) + n - 1]
        else:
            valid[0] = False
            rows = logits[: n - 1]
            rows = self.torch.cat([logits[:1], rows]) if n > 1 else logits[:1]
        return _softmax(rows.float().numpy()), valid


def load_blocks(
    split: str = "val",
    max_tokens: int = 256,
    n_blocks: int = 400,
    out_dir: Path = CORPUS_DIR,
) -> list[dict]:
    """Eligible blocks (n_tokens <= max_tokens) selected round-robin across runs so
    the default budget spans several runs (bootstrap unit). Returns blocks in
    (run, seq) order, each with a `ctx_blocks` list of its own run's earlier
    eligible blocks (for the context window)."""
    import pyarrow.parquet as pq

    t = pq.read_table(out_dir / f"{split}.parquet").to_pydict()
    by_run: dict[str, list[dict]] = {}
    for i in range(len(t["run"])):
        if t["n_tokens"][i] > max_tokens:
            continue
        by_run.setdefault(t["run"][i], []).append(
            {
                "seq": t["seq"][i],
                "kind": kind_label(t["kind"][i], t["tool"][i]),
                "text": t["text"][i],
            }
        )
    runs = {r: sorted(bs, key=lambda b: b["seq"]) for r, bs in sorted(by_run.items())}
    if not any(runs.values()):
        raise RuntimeError(f"no blocks <= {max_tokens} tokens in {out_dir / (split + '.parquet')}")
    blocks: list[dict] = []
    pos = {r: 0 for r in runs}
    while len(blocks) < n_blocks and any(pos[r] < len(runs[r]) for r in runs):
        for r in runs:
            if pos[r] < len(runs[r]):
                blocks.append({"run": r, **runs[r][pos[r]]})
                pos[r] += 1
                if len(blocks) == n_blocks:
                    break
    # context candidates: earlier eligible blocks of the same run (by seq)
    sel = {(b["run"], b["seq"]) for b in blocks}
    for b in blocks:
        b["ctx_blocks"] = [
            {"kind": x["kind"], "text": x["text"]}
            for x in runs[b["run"]]
            if x["seq"] < b["seq"] and (b["run"], x["seq"]) in sel
        ]
    return blocks


def _context(ctx_blocks: list[dict], ids_of, budget: int) -> list[int]:
    """Token ids of the most recent preceding blocks that fit in `budget`."""
    if budget <= 0:
        return []
    ctx: list[int] = []
    for b in reversed(ctx_blocks):
        ids = ids_of(b["text"])
        if len(ctx) + len(ids) > budget and ctx:
            break
        ctx = ids + ctx
        if len(ctx) >= budget:
            break
    return ctx


def _bootstrap(
    records: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_boot: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]]:
    """Per kind (+ 'overall'): (resampled mean alpha (n_boot,), resampled greedy
    rate, resampled greedy_corpus rate, resampled coverage, n_tokens, n_runs),
    resampling runs with replacement."""
    rng = np.random.default_rng(seed)
    groups: dict[str, dict[str, list[tuple[np.ndarray, ...]]]] = {}
    for run, kind, a, g, gc, v in records:
        groups.setdefault(kind, {}).setdefault(run, []).append((a, g, gc, v))
        groups.setdefault("overall", {}).setdefault(run, []).append((a, g, gc, v))
    out = {}
    for scope, by_run in groups.items():
        runs = sorted(by_run)
        sums_a = np.array([np.concatenate([r[0] for r in by_run[rn]]).sum() for rn in runs])
        sums_g = np.array([np.concatenate([r[1] for r in by_run[rn]]).sum() for rn in runs])
        sums_gc = np.array([np.concatenate([r[2] for r in by_run[rn]]).sum() for rn in runs])
        sums_v = np.array([np.concatenate([r[3] for r in by_run[rn]]).sum() for rn in runs])
        cnts = np.array([sum(r[0].size for r in by_run[rn]) for rn in runs])
        # (n_boot, n_runs): each replicate draws len(runs) runs with replacement
        w = rng.multinomial(len(runs), np.full(len(runs), 1.0 / len(runs)), size=n_boot)
        # per-run rates over the drafter's valid (proposed) positions: this is
        # what the sweep's accept = n_acc/n_draft measures (spec.py only scores
        # positions the drafter proposed)
        gv = np.where(sums_v > 0, sums_g / np.maximum(sums_v, 1), 0.0)
        av = np.where(sums_v > 0, sums_a / np.maximum(sums_v, 1), 0.0)
        out[scope] = (
            w @ sums_a / (w @ cnts),
            w @ sums_g / (w @ cnts),
            w @ sums_gc / (w @ cnts),
            w @ sums_v / (w @ cnts),
            w @ gv / len(runs),
            w @ av / len(runs),
            int(cnts.sum()),
            len(runs),
        )
    return out


def _ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def _k_tables(alpha: np.ndarray, c: float) -> dict[str, dict[str, list[float] | float]]:
    """tokens_per_call / speedup per k: point estimates + CIs propagated from the
    bootstrap alpha distribution."""
    tpc, sp, tpc_ci, sp_ci = {}, {}, {}, {}
    for k in K_VALUES:
        e = expected_tokens_per_call(alpha, k)
        s = e / (1.0 + k * c)
        tpc[str(k)] = float(e.mean())
        sp[str(k)] = float(s.mean())
        tpc_ci[str(k)] = _ci(e)
        sp_ci[str(k)] = _ci(s)
    return {
        "tokens_per_call": tpc,
        "speedup": sp,
        "tokens_per_call_ci95": tpc_ci,
        "speedup_ci95": sp_ci,
    }


def evaluate(
    blocks: list[dict],
    target,
    drafter,
    ids_of,
    ctx_budget: int,
) -> tuple[list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], float]:
    """Score every block position. Returns (records, c) with records =
    (run, kind, alpha (T,), greedy (T,), greedy_corpus (T,), valid (T,)) and c
    the measured draft/target cost ratio. greedy = argmax q == argmax p (the
    real spec-accept); greedy_corpus = argmax q == actual token (comparable to
    results/ngram.json, which scores proposals against the corpus)."""
    records: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for i, b in enumerate(blocks):
        ctx = _context(b["ctx_blocks"], ids_of, ctx_budget)
        ids = ids_of(b["text"])
        p, _ = target.dists(ctx, ids)
        q, valid = drafter.dists(ctx, ids)
        a, g = alpha_and_greedy(p, q, valid)
        g_corpus = valid & (q.argmax(axis=-1) == np.asarray(ids))
        records.append((b["run"], b["kind"], a, g, g_corpus, valid))
        if (i + 1) % 50 == 0:
            print(f"  scored {i + 1}/{len(blocks)} blocks", flush=True)

    def _timed(f, reps: int = 3) -> float:
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t0)
        return median(ts)

    ctx, ids = _context(blocks[0]["ctx_blocks"], ids_of, ctx_budget), ids_of(blocks[0]["text"])
    t_target = _timed(lambda: target.dists(ctx, ids))
    t_draft = _timed(lambda: drafter.dists(ctx, ids))
    return records, max(t_draft / t_target, 0.0)


def aggregate(
    records: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    c: float,
    n_boot: int,
    seed: int,
) -> dict[str, dict]:
    """Per kind (+ overall): n_tokens, alpha, CIs, greedy (target agree),
    greedy_corpus (vs actual token), coverage, k tables."""
    out: dict[str, dict] = {}
    for scope, (b_alpha, b_greedy, b_gc, b_cov, b_gv, b_av, n_tok, n_runs) in sorted(
        _bootstrap(records, n_boot, seed).items()
    ):
        out[scope] = {
            "n_tokens": n_tok,
            "n_runs": n_runs,
            "alpha": float(b_alpha.mean()),
            "alpha_ci95": _ci(b_alpha),
            "greedy": float(b_greedy.mean()),
            "greedy_ci95": _ci(b_greedy),
            "greedy_corpus": float(b_gc.mean()),
            "greedy_corpus_ci95": _ci(b_gc),
            "coverage": float(b_cov.mean()),
            # over valid (proposed) positions only: comparable to the engine's
            # measured accept (n_acc/n_draft) and tpc (n_emit/n_steps)
            "greedy_valid": float(b_gv.mean()),
            "greedy_valid_ci95": _ci(b_gv),
            "alpha_valid": float(b_av.mean()),
            "alpha_valid_ci95": _ci(b_av),
            **_k_tables(b_alpha, c),
            "tpc_valid_ci95": {str(k): _ci(expected_tokens_per_call(b_av, k)) for k in K_VALUES},
        }
    return out


def run(
    drafter: str = "ngram",
    draft_model: str | None = None,
    target_dir: str | Path = TARGET_DIR,
    split: str = "val",
    n_blocks: int = 400,
    max_tokens: int = 256,
    ctx_budget: int = 1024,
    n_boot: int = 1000,
    seed: int = 0,
    out: str | Path | None = None,
) -> dict:
    """End-to-end: load blocks, score target vs drafter, aggregate, write JSON."""
    from transformers import AutoTokenizer

    blocks = load_blocks(split, max_tokens, n_blocks)
    tok = AutoTokenizer.from_pretrained(target_dir)  # shared vocab with the corpus tokenizer

    def ids_of(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    max_len = ctx_budget + max_tokens
    target = JaxModel.from_path(target_dir, max_len)
    vocab = target.cfg.vocab_size
    name = "ngram"
    if drafter == "ngram":
        drafter_obj = NGramDrafter(vocab)
    elif drafter == "jax":
        if not draft_model:
            raise SystemExit("--draft-model required for --drafter jax")
        name = Path(draft_model).name
        drafter_obj = JaxModel.from_path(draft_model, max_len)
    elif drafter == "hf":
        if not draft_model:
            raise SystemExit("--draft-model required for --drafter hf")
        name = Path(draft_model).name
        drafter_obj = HFDrafter(draft_model)
    else:
        raise SystemExit(f"unknown drafter {drafter!r}")

    print(
        f"alpha: target={Path(target_dir).name} drafter={name} split={split} "
        f"blocks={len(blocks)} (<= {max_tokens} tok, ctx <= {ctx_budget} tok)",
        flush=True,
    )
    records, c = evaluate(blocks, target, drafter_obj, ids_of, ctx_budget)
    per_scope = aggregate(records, c, n_boot, seed)

    result = {
        "metric": "alpha = sum_x min(p(x), q(x)) per position, teacher-forced (context = "
        "run's preceding selected blocks within ctx budget + block prefix), "
        "full vocab; greedy = argmax q == argmax p (target); greedy_corpus = "
        "argmax q == actual token (baseline ngram.json metric); coverage = "
        "fraction of positions the drafter proposes",
        "drafter": name,
        "drafter_type": drafter,
        "target": Path(target_dir).name,
        "split": split,
        "n_blocks": len(blocks),
        "n_runs": len({b["run"] for b in blocks}),
        "n_tokens": per_scope["overall"]["n_tokens"],
        "max_tokens": max_tokens,
        "ctx_budget": ctx_budget,
        "c": c,
        "k_values": list(K_VALUES),
        "bootstrap": {"n_resamples": n_boot, "seed": seed, "unit": "runs (with replacement)"},
        "by_kind": {k: v for k, v in per_scope.items() if k != "overall"},
        "overall": per_scope["overall"],
    }
    out = Path(out) if out else RESULTS_DIR / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {out}")
    for k in list(result["by_kind"]) + ["overall"]:
        s = result["overall"] if k == "overall" else result["by_kind"][k]
        print(
            f"  {k:18s} alpha={s['alpha']:.4f} greedy={s['greedy']:.4f} "
            f"greedy_corpus={s['greedy_corpus']:.4f} cov={s['coverage']:.3f} "
            f"tok/call@k8={s['tokens_per_call']['8']:.2f} speedup@k8={s['speedup']['8']:.2f}"
        )
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m speculative_agent_workloads.eval.alpha")
    ap.add_argument("--drafter", choices=["ngram", "jax", "hf"], default="ngram")
    ap.add_argument(
        "--draft-model",
        default=None,
        help="checkpoint dir (JAX: config+safetensors; HF: full or PEFT adapter)",
    )
    ap.add_argument("--target", default=str(TARGET_DIR))
    ap.add_argument("--split", default="val")
    ap.add_argument("--blocks", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--ctx-tokens",
        type=int,
        default=1024,
        help="token budget for the run's preceding blocks in the context",
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run(
        drafter=args.drafter,
        draft_model=args.draft_model,
        target_dir=args.target,
        split=args.split,
        n_blocks=args.blocks,
        max_tokens=args.max_tokens,
        ctx_budget=args.ctx_tokens,
        n_boot=args.n_boot,
        seed=args.seed,
        out=args.out,
    )


if __name__ == "__main__":
    main()
