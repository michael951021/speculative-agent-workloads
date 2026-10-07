"""Known-answer cross-check for the tiny.json re-run (id: alphafix).

Hand-computed alpha (pure transformers logits, no speculative-agent-workloads metric code) vs the
pipeline (JAX 1.5B target + HFDrafter tiny export + alpha_and_greedy) on 4 real
val blocks (2 empty-ctx, 2 non-empty-ctx). Target-row alignment is anchored by
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative_agent_workloads.eval.alpha import (
    TARGET_DIR,
    HFDrafter,
    JaxModel,
    _context,
    alpha_and_greedy,
    load_blocks,
)

TOL = 2e-3
DRAFT = "ckpt/tiny/export"
# blocks 0,1: empty ctx (run starts); 3,5: non-empty ctx (the off-by-one region)
BLOCK_IDX = (0, 1, 3, 5)


def main() -> int:
    blocks = [load_blocks("val", 256, 400)[i] for i in BLOCK_IDX]
    tok = AutoTokenizer.from_pretrained(TARGET_DIR)

    def ids_of(t: str) -> list[int]:
        return tok.encode(t, add_special_tokens=False)

    target = JaxModel.from_path(TARGET_DIR, 1024 + 256)
    hf = HFDrafter(DRAFT)
    ref_t = AutoModelForCausalLM.from_pretrained(TARGET_DIR, torch_dtype=torch.float32).eval()
    ref_d = AutoModelForCausalLM.from_pretrained(DRAFT, torch_dtype=torch.float32).eval()

    ok, worst = True, 0.0
    for i, b in enumerate(blocks):
        ctx = _context(b["ctx_blocks"], ids_of, 1024)
        ids = ids_of(b["text"])
        p, _ = target.dists(ctx, ids)
        q, valid = hf.dists(ctx, ids)
        a, _ = alpha_and_greedy(p, q, valid)

        with torch.no_grad():
            lp = ref_t(input_ids=torch.tensor([ctx + ids])).logits[0]
            lq = ref_d(input_ids=torch.tensor([ctx + ids])).logits[0]
        n = len(ids)
        ref_a, ref_valid = np.zeros(n), np.ones(n, dtype=bool)
        if not ctx:
            ref_valid[0] = False
        for j in range(n):
            if not ref_valid[j]:
                continue
            sp = torch.softmax(lp[len(ctx) + j - 1], -1).numpy()
            sq = torch.softmax(lq[len(ctx) + j - 1], -1).numpy()
            ref_a[j] = np.minimum(sp, sq).sum()

        d = np.abs(a[valid] - ref_a[ref_valid])
        worst = max(worst, float(d.mean()))
        print(
            f"block[{i}] kind={b['kind']} run={b['run'][:8]} n={n} ctx={len(ctx)} "
            f"pipeline={a[valid].mean():.6f} ref={ref_a[ref_valid].mean():.6f} "
            f"mean|d|={d.mean():.2e}",
            flush=True,
        )
        ok = ok and bool(d.mean() < TOL)

    print("PASS" if ok else "FAIL", f"(worst mean|d|={worst:.2e}, tol={TOL})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
