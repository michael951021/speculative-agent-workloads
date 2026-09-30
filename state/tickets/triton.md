# triton — Triton verify and alpha kernels

## Issue
The alpha eval and spec-decode verify loop are per-step overhead; the GPU run (scripts/gpu_run.sh)
needs one fused Triton kernel for α and acceptance at V=151936, and the project needs concrete
Triton work. Today alpha/verify would be chains of torch ops (multiple launches, temporaries).

## Question
Can a single Triton launch (a) verify k drafts — online-softmax target [k+1,V] and draft [k,V]
rows, accept test p > u·q per drafted token, first-reject index, normalized residual
max(0, p−q) at that row — and (b) compute per-row Σ min(softmax p, softmax q), each matching a
torch reference to fp32 1e-5? Refutation: any elementwise mismatch > 1e-5, wrong reject index,
or residual row not summing to 1.

## Evidence
- `draftlab/eval/alpha.py:44` alpha_and_greedy (numpy ref); `draftlab/jaxlm/spec.py` accept
  convention: `u·q[t] < p[t]`, residual `max(0, p−q)` (strict → 0/0 rejects; kernel uses
  `p > u·q`, same).
- Repro: `.venv/bin/python -m pytest draftlab/tests/test_verify_kernel.py -q` with
  TRITON_INTERPRET=1, V=1000 random logits vs torch reference (test file is the check; run
  before kernel exists → collection error = expected failure).

## Done when
- `verify.py`: `fused_verify(p_logits [k+1,V], q_logits [k,V], tokens [k], u [k]) ->
  (first_reject int, residual [V])` (residual row = target row fr if fr<k else row k, q=0 →
  bonus p); `fused_alpha(p [N,V], q [N,V]) -> [N]`; single launch each, V tiled, online softmax.
- Torch refs `verify_torch`/`alpha_torch` in same module.
- Tests (TRITON_INTERPRET=1, V=1000, fp32, atol 1e-5): random vs ref; identical rows → all
  accepted + residual = p_k; forced reject at 0; u=0 → accept iff p[t]>0; alpha: identical → 1,
  uniform q → Σ min(p, 1/V) exactly (known-answer/analytic), random vs ref.
- `bench.py`: prints kernel-vs-torch table at V=151936, k=16; exits 0 with skip message w/o CUDA.
- `alpha.py` uses fused_alpha when `torch.cuda.is_available()` (fallback numpy otherwise).
- `make check` subset (ruff + this test file) green.

## Stop when
Scope: no other kernel types (alpha kernels for ngram etc.), no serve/ integration, no GPU
timing claims in sandbox. If triton 3.8 interpret mode can't express the 2-pass online softmax
after 2 attempts with new evidence, stop and split (e.g. separate M/S + value kernels).

## Plan
1. This ticket. 2. `draftlab/kernels/{__init__.py,verify.py}` (2 jit kernels + torch refs +
   wrappers). 3. `draftlab/tests/test_verify_kernel.py`; run, debug. 4. `bench.py`.
   5. Hook `alpha_and_greedy` on CUDA. 6. ruff + pytest, NOTES.md, commit.

## Result (2026-09-30)
Tests 7/7 under TRITON_INTERPRET=1 (V=1000, atol 1e-5); bench skip path OK; alpha.py hooked
(CUDA-only kernel path, numpy fallback — unchanged CPU behavior, test_alpha 6/6). Suite log:
.agent-evidence/triton-pytest.log.
