# gpu — GPU run script

## Issue
Numbers at realistic target size (Qwen2.5-7B-Instruct) and served end-to-end (vLLM + gateway) can only
be measured on the host GPU; the repo has no single script to run that, so the GPU-scale section of the
report has no reproducible procedure.

## Question
Can one script (`scripts/gpu_run.sh`) run the full GPU procedure on the host — alpha eval at 7B with the
Triton kernel, vLLM with ngram and LoRA speculative configs, gateway load bench at concurrency 1/4/16 —
writing results/gpu/*.json? Refutation: `--dry-run` in the sandbox does GPU work, fails on host-only
files, or omits any of the three stages.

## Evidence
- Components exist: `speculative_agent_workloads.eval.alpha --drafter ngram|hf` (auto-uses `fused_alpha` on CUDA,
  `eval/alpha.py:63`), `deploy/docker/vllm-entrypoint.sh` spec-config pattern, gateway
  (`gateway/config.yaml` backends+keys, `/v1/chat/completions` SSE, `/metrics` with
  speculative_agent_workloads_{draft,accepted}_tokens_total, ttft/latency histograms), `ckpt/lora/adapter_config.json`.
- Check: `bash scripts/gpu_run.sh --dry-run` (run before the script exists → expected failure: missing file).
  Output kept in `.agent-evidence/`.

## Done when
- `bash scripts/gpu_run.sh --dry-run` exits 0 in the sandbox: prints the stage plan and per-file
  existence checks (sandbox files must exist; host-only files like the 7B model dir reported as
  host-only, not failures), performs no GPU/vLLM action.
- Real mode (host, after `./agent stop`): (1) alpha eval ngram and lora-drafter at target 7B-instruct →
  results/gpu/alpha_{ngram,lora}_7b.json; (2) vLLM serve baseline + `--speculative-config` ngram +
  LoRA-drafter arms, each benched through the gateway at concurrency 1/4/16 via
  `tools/bench_gateway.py` → results/gpu/bench_<arm>_c{1,4,16}.json; (3) summary.json.
- Known-answer checks: alpha anchors (Leviathan: small drafter α 0.5–0.9; bigram ~0.2; llama.cpp
  0.5B→7B k=4 ≈1.41–1.75×) written as expected ranges in the plan; bench sanity: baseline arm must have
  draft_tokens_total delta 0 (no spec config), spec arms must have accepted>0.

## Stop when
No GPU in sandbox: real mode is not executed here. Two attempts to make `--dry-run`/bench tool work
without new evidence → record blocker, stop. No changes to speculative_agent_workloads/ or gateway/ beyond a new file in
tools/; no deploy of anything.

## Plan
1. Write `tools/bench_gateway.py` (stdlib+requests SSE stream, TTFT/latency/throughput, /metrics delta).
2. Write `scripts/gpu_run.sh` (env-configurable, --dry-run, stages 1–3, port cleanup, logs to results/gpu/).
3. Accept: `bash scripts/gpu_run.sh --dry-run` exit 0; probe bench tool against a mock SSE server.
4. NOTES.md + commit.
