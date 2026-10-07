"""Independent reference for the T=0.7 acceptance arithmetic (batch 3).

Cross-check the engine's T>0 accept path on the ngram k=1 T=0.7 cell: rerun the same prompts
through SpecDecoder (same seed -> same trajectory) and compute per-draft
expected acceptance BY HAND from raw target logits, p7 = softmax(lg/0.7),
q one-hot (ngram) => E[accept] = p7[d]. No engine acceptance code is reused.

PASS: pooled measured acc within the bootstrap 95% CI of the hand-computed
expected acc (per-prompt paired resampling), and consistent with the
parquet cell value.

Run AFTER the sweep (shares the CPU):
  OMP_NUM_THREADS=16 XLA_FLAGS=--xla_cpu_multi_thread_eigen=true \
  .venv/bin/python tools/check_ngram_t07_ref.py
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import pandas as pd  # noqa: E402

from speculative_agent_workloads.jaxlm import model as M  # noqa: E402
from speculative_agent_workloads.jaxlm import spec as S  # noqa: E402
from speculative_agent_workloads.sweep import (  # noqa: E402
    DRAFTERS,
    MODELS,
    N_GEN,
    SEED,
    load_prompts,
)

TARGET = str(MODELS / "qwen2.5-0.5b-instruct")
K, TEMP = 1, 0.7
# sweep cell seed = SEED + 100*di + 10*ki + ti with default --temps 0 0.7 (ti=1)
SEED_BASE = SEED + 100 * DRAFTERS.index("ngram") + 0 + 1


def main() -> int:
    tcfg, tparams = M.load_safetensors(TARGET)
    prompts = load_prompts(max_tokens=128, target_dir=TARGET)
    max_len = max(len(p["ids"]) for p in prompts) + N_GEN + 8

    sum_e = sum_m = 0.0
    per_prompt: list[tuple[float, float, int]] = []
    for p in prompts:
        seed = SEED_BASE + zlib.crc32(f"{p['run']}:{p['seq']}".encode())
        dec = S.SpecDecoder(
            tcfg,
            tparams,
            S.NGramDrafter(tcfg.vocab_size),
            max_len,
            k=K,
            temperature=TEMP,
            seed=seed,
        )
        out, stats = dec.generate(p["ids"], N_GEN)
        L = len(p["ids"])
        lg, _ = M.prefill(tcfg, tparams, p["ids"] + out, max_len)
        rows = np.asarray(lg)
        ng = S.NGramDrafter(tcfg.vocab_size)
        c = 0  # committed generated tokens
        e_tot = 0.0
        n_draft = 0
        for st in stats:
            rem = N_GEN - c
            ctx = p["ids"] + out[:c]
            toks, probs = ng.propose(ctx, min(K, rem))
            for j in range(len(toks)):
                p7 = np.asarray(jax.nn.softmax(rows[L + c + j - 1] / TEMP), np.float32)
                e_tot += float(np.minimum(1.0, p7[toks[j]] / probs[j][toks[j]]))
                n_draft += 1
            c += min(rem, st.accepted + 1)
            if c >= N_GEN:
                break
        sum_e += e_tot
        sum_m += st_accept(stats, c)
        per_prompt.append((e_tot, st_accept(stats, c), n_draft))

    def pooled(num, den):
        return num / den

    n_draft_tot = sum(n for _, _, n in per_prompt)
    a_m = sum_m / n_draft_tot
    a_e = sum_e / n_draft_tot
    # paired bootstrap CI of per-prompt (measured - expected) acceptance
    rng = np.random.default_rng(0)
    d = np.array([m / n - e / n for e, m, n in per_prompt if n])
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    lo, hi = (float(x) for x in np.percentile(b, [2.5, 97.5]))

    print(
        f"ngram k=1 T=0.7: measured {a_m:.4f} vs hand-computed expected "
        f"{a_e:.4f} over {n_draft_tot} draft positions "
        f"({len(per_prompt)} prompts); CI95(diff) [{lo:+.4f}, {hi:+.4f}]"
    )
    cell = pd.read_parquet(ROOT / "results" / "sweep.parquet")
    s = cell[(cell.drafter == "ngram") & (cell.k == 1) & (cell.temp == 0.7)]
    a_cell = float(s["n_acc"].sum() / s["n_draft"].sum())
    print(f"parquet cell acc {a_cell:.4f} (rerun measured {a_m:.4f})")
    ok = lo < 0 < hi
    ok &= abs(a_cell - a_m) < 0.02  # same cell, independent rerun
    print("NGRAM T=0.7 REFERENCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def st_accept(stats, c_end: int) -> int:
    """Accepted drafts, recomputed from StepStats (== sum of accepted)."""
    return int(sum(st.accepted for st in stats))


if __name__ == "__main__":
    sys.exit(main())
