"""N-gram prompt-lookup drafter (vLLM-style): the model-less floor.

At each position, match the last n tokens (n=max_n..1, longest first) against
the known context and propose the token that first followed that n-gram
earlier. No self-match: an n-gram only counts once its continuation token is
already in the context. Teacher-forced evaluation on a split: greedy acceptance
per position and, for k drafts, the mean accepted length (a run of accepted
proposals is exactly a run of matches of the precomputed per-position
proposals, since accepted tokens equal the real prefix).

Context for a block = the previous max_prev blocks of the same run (any kind,
seq order) + the block's own prefix. Splits are per-run, so no leakage.
"""

import json
from collections import defaultdict
from pathlib import Path

from draftlab.data.corpus import OUT_DIR, RESULTS_DIR

K_VALUES = (1, 2, 4, 8)
MAX_N = 4
MAX_PREV = 8
N_BOOTSTRAP = 200
SEED = 0


class NGramMatcher:
    """Prompt-lookup proposer over a token sequence (see module docstring)."""

    def __init__(self, max_n: int = MAX_N) -> None:
        self.max_n = max_n
        self._d = [dict() for _ in range(max_n + 1)]

    def proposals(self, ctx: list[int]) -> list[int | None]:
        """P[t] = proposed token for ctx[t] given ctx[:t], or None (no match)."""
        d = self._d
        max_n = self.max_n
        out: list[int | None] = [None] * len(ctx)
        for t in range(len(ctx)):
            for n in range(max_n, 0, -1):
                if t < n:
                    continue
                g = tuple(ctx[t - n : t])
                dn = d[n]
                if g in dn:
                    out[t] = dn[g]
                    break
            # ctx[t] is now known: register n-grams ending at t-1, continuation ctx[t]
            for n in range(1, max_n + 1):
                if t < n:
                    break
                g = tuple(ctx[t - n : t])
                dn = d[n]
                if g not in dn:
                    dn[g] = ctx[t]
        return out

    @staticmethod
    def accept_runs(accept: list[bool]) -> list[int]:
        """runs[t] = consecutive Trues of `accept` starting at t."""
        if not accept:
            return []
        runs = [0] * len(accept)
        if accept[-1]:
            runs[-1] = 1
        for t in range(len(accept) - 2, -1, -1):
            if accept[t]:
                runs[t] = runs[t + 1] + 1
        return runs


def _zero_agg() -> dict:
    return {
        "n_pos": 0,
        "n_prop": 0,
        "n_acc": 0,
        "sum_L": {str(k): 0 for k in K_VALUES},
    }


def _add_agg(dst: dict, src: dict) -> None:
    dst["n_pos"] += src["n_pos"]
    dst["n_prop"] += src["n_prop"]
    dst["n_acc"] += src["n_acc"]
    for k in dst["sum_L"]:
        dst["sum_L"][k] += src["sum_L"][k]


def _metrics(agg: dict) -> dict:
    n = agg["n_pos"]
    if n == 0:
        return {"n_positions": 0}
    m = {
        "n_positions": n,
        "coverage": agg["n_prop"] / n,
        "greedy_accept": agg["n_acc"] / n,
    }
    if agg["n_prop"]:
        m["accept_given_proposal"] = agg["n_acc"] / agg["n_prop"]
    by_k = {}
    for k in K_VALUES:
        mean_l = agg["sum_L"][str(k)] / n
        by_k[str(k)] = {"mean_accepted": mean_l, "exp_tokens_per_call": mean_l + 1.0}
    m["by_k"] = by_k
    return m


def evaluate(
    split: str = "val",
    out_dir: Path = OUT_DIR,
    tok=None,
    k_values: tuple[int, ...] = K_VALUES,
    max_prev: int = MAX_PREV,
    max_n: int = MAX_N,
) -> dict:
    """Teacher-forced ngram eval on one split. Returns the results dict (unwritten)."""
    import pyarrow.parquet as pq

    if tok is None:
        from transformers import AutoTokenizer

        from draftlab.data.corpus import TOKENIZER_DIR

        tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    table = pq.read_table(out_dir / f"{split}.parquet").to_pydict()
    runs: dict[str, list[dict]] = defaultdict(list)
    for i in range(len(table["run"])):
        runs[table["run"][i]].append(
            {
                "seq": table["seq"][i],
                "kind": table["kind"][i],
                "text": table["text"][i],
            }
        )

    tok_cache: dict[str, list[int]] = {}

    def tokens_of(text: str) -> list[int]:
        t = tok_cache.get(text)
        if t is None:
            t = tok.encode(text, add_special_tokens=False)
            tok_cache[text] = t
        return t

    # per-run, per-kind aggregates for bootstrap; per-run list for resampling
    per_run: dict[str, dict[str, dict]] = {}
    for run, blocks in runs.items():
        blocks.sort(key=lambda b: b["seq"])
        prev: list[list[int]] = []
        agg_by_kind: dict[str, dict] = defaultdict(_zero_agg)
        for b in blocks:
            own = tokens_of(b["text"])
            ctx = [t for lst in prev[-max_prev:] for t in lst] + own
            off = len(ctx) - len(own)
            P = NGramMatcher(max_n).proposals(ctx)
            acc = [p is not None and p == own[j] for j, p in enumerate(P[off:])]
            runs_arr = NGramMatcher.accept_runs(acc)
            a = agg_by_kind[b["kind"]]
            a["n_pos"] += len(own)
            a["n_prop"] += sum(p is not None for p in P[off:])
            a["n_acc"] += sum(acc)
            for k in k_values:
                a["sum_L"][str(k)] += sum(min(r, k) for r in runs_arr[: len(own)])
            prev.append(own)
            if len(prev) > max_prev:
                prev.pop(0)
        per_run[run] = {k: v for k, v in agg_by_kind.items()}

    kinds = sorted({k for r in per_run.values() for k in r})
    all_agg = _zero_agg()
    kind_agg = {k: _zero_agg() for k in kinds}
    for r in per_run.values():
        for k, a in r.items():
            _add_agg(kind_agg[k], a)
            _add_agg(all_agg, a)

    def bootstrap(ci_keys=("greedy_accept",), k_ci_keys=("exp_tokens_per_call",)) -> dict:
        import numpy as np

        rng = np.random.default_rng(SEED)
        run_ids = list(per_run)
        samples: dict[str, list[float]] = defaultdict(list)
        for _ in range(N_BOOTSTRAP):
            resampled = [run_ids[i] for i in rng.integers(0, len(run_ids), len(run_ids))]
            scope_agg = {k: _zero_agg() for k in kinds}
            tot = _zero_agg()
            for run in resampled:
                for k, a in per_run[run].items():
                    _add_agg(scope_agg[k], a)
                    _add_agg(tot, a)
            for k in kinds:
                m = _metrics(scope_agg[k])
                for key in ci_keys:
                    if key in m:
                        samples[f"{k}|{key}"].append(m[key])
            mo = _metrics(tot)
            for key in ci_keys:
                if key in mo:
                    samples[f"overall|{key}"].append(mo[key])
            for k in kinds:
                m = _metrics(scope_agg[k])
                for kk, m2 in m.get("by_k", {}).items():
                    for key in k_ci_keys:
                        if key in m2:
                            samples[f"{k}|k{kk}|{key}"].append(m2[key])
            mo = _metrics(tot)
            for kk, m2 in mo.get("by_k", {}).items():
                for key in k_ci_keys:
                    if key in m2:
                        samples[f"overall|k{kk}|{key}"].append(m2[key])
        ci = {}
        for key, vals in samples.items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            ci[key] = [float(lo), float(hi)]
        return ci

    return {
        "method": (
            "ngram prompt-lookup: match last n=1..4 (longest first), first-occurrence continuation"
        ),
        "split": split,
        "max_n": max_n,
        "max_prev_blocks": max_prev,
        "k_values": list(k_values),
        "n_runs": len(per_run),
        "n_blocks": sum(len(r) for r in runs.values()),
        "by_kind": {k: _metrics(kind_agg[k]) for k in kinds},
        "overall": _metrics(all_agg),
        "bootstrap": {
            "n_resamples": N_BOOTSTRAP,
            "seed": SEED,
            "unit": "runs (with replacement)",
            "ci95": bootstrap(),
        },
    }


def main(split: str = "val", out_path: Path | None = None) -> Path:
    res = evaluate(split)
    out_path = out_path or (RESULTS_DIR / "ngram.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1) + "\n")
    print(f"ngram eval ({split}) -> {out_path}")
    for k, m in res["by_kind"].items():
        acc = m.get("greedy_accept", float("nan"))
        tpc = {kk: v["exp_tokens_per_call"] for kk, v in m.get("by_k", {}).items()}
        print(f"  {k:12s} accept={acc:6.3f}  tokens/call {tpc}")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(args.split, Path(args.out) if args.out else None)
