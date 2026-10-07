"""Bench fused verify/alpha kernels vs torch references. GPU only.

Usage: .venv/bin/python -m speculative_agent_workloads.kernels.bench
Prints a median-latency table at V=151936 (Qwen2 vocab), k=16, fp32; exits 0
with a skip message when CUDA is unavailable (CPU sandbox).
"""

import sys
import time

import torch


def _median_ms(fn, reps: int = 50) -> float:
    fn()  # warmup
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available - skipping kernel bench (CPU sandbox).")
        return 0
    from speculative_agent_workloads.kernels import verify

    torch.manual_seed(0)
    V, k = 151936, 16
    p = torch.randn(k + 1, V, device="cuda")
    q = torch.randn(k, V, device="cuda")
    t = torch.randint(0, V, (k,), device="cuda", dtype=torch.int32)
    u = torch.rand(k, device="cuda")
    rows = p[:k]
    row = (
        f"{'op':<10} {'kernel ms':>11} {'torch ms':>11} {'speedup':>9}\n"
        f"{'-' * 10} {'-' * 11} {'-' * 11} {'-' * 9}"
    )
    print(f"V={V}, k={k}, fp32, median of 50 runs")
    kv, tv = (
        _median_ms(lambda: verify.fused_verify(p, q, t, u)),
        _median_ms(lambda: verify.verify_torch(p, q, t, u)),
    )
    ka, ta = (
        _median_ms(lambda: verify.fused_alpha(rows, q)),
        _median_ms(lambda: verify.alpha_torch(rows, q)),
    )
    print(
        f"{row}\n"
        f"{'verify':<10} {kv:>11.4f} {tv:>11.4f} {tv / kv:>8.2f}x\n"
        f"{'alpha':<10} {ka:>11.4f} {ta:>11.4f} {ta / ka:>8.2f}x"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
