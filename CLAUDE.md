# draftlab — how predictable is a coding agent?

Speculative decoding measured on this machine's own agent traffic. `data/raw/agent_turns.jsonl` holds ~50k content
blocks (thinking, text, tool calls, tool results) that the local Qwen 27B agents produced or read. draftlab asks: how
many of those tokens could a small draft model (or plain n-gram lookup) have guessed, per kind of content, and what
speedup does that buy? It builds the whole stack to answer that: a JAX speculative-decoding engine, trained drafters
(JAX distillation, PyTorch LoRA/FSDP, DPO), a Triton verify kernel, a Go gateway with Prometheus metrics, Ray sweeps,
an LLM-as-judge check, deploy configs (Helm, Terraform, Grafana) and a React explorer of the results.

## HARD RULES
1. **No GPU in the sandbox.** Everything must run on CPU: JAX `jax[cpu]`, torch CPU wheels, Triton under
   `TRITON_INTERPRET=1`. GPU paths exist behind flags and in `scripts/gpu_run.sh`, which Kevin runs on the host.
   Cap threads: `OMP_NUM_THREADS=16 XLA_FLAGS=--xla_cpu_multi_thread_eigen=true` (other loops share the CPU).
2. **Never apply or deploy anything.** Terraform: `init -backend=false` + `validate` only, no credentials, no `plan`
   against a real project. Helm/k8s: lint + kubeconform only. Docker is not available in the sandbox; write
   Dockerfiles and lint them with hadolint, don't try to build.
3. LLM calls (judge only) go to `$OLLAMA_BASE_URL` (OpenAI-compatible /v1) through the `openai` SDK. Never call
   127.0.0.1:11434. That model serves the other agents too, so cap calls (see each task) and cache responses.
4. `data/`, `models/`, `ckpt/`, `.tools/`, `.venv/` are gitignored and shared between agents. Commit `results/`
   (small JSON/Parquet, < 5 MB each) and figures.
5. Keep compute budgets small and configurable (tokens, steps, blocks). A task's default run must finish in < 30 min
   on CPU; bigger numbers are for the GPU run.
6. BE SUCCINCT in comments, notes and docs.

## Layout (target)
- `draftlab/data/` corpus → Parquet, labels, splits · `draftlab/jaxlm/` Qwen2 in JAX, KV cache, spec decoding
- `draftlab/drafters/` ngram, distill (JAX/Optax), lora (PyTorch PEFT + FSDP), dpo (TRL)
- `draftlab/kernels/` Triton verify + alpha kernels · `draftlab/serve/` FastAPI engine server
- `draftlab/eval/` alpha metric, judge · `draftlab/sweep.py` Ray · `draftlab/export_ui.py`
- `gateway/` Go module (OpenAI-compatible proxy, auth, rate limit, breaker, /metrics, serves web/dist at /ui)
- `web/` Vite + React + TypeScript explorer · `deploy/{docker,helm,terraform,grafana,prometheus,kind}`
- `tools/` bootstrap + host-only export · `scripts/gpu_run.sh` host GPU run · `results/` · `REPORT.md`

## Commands
- `make bootstrap` (uv, Python 3.12 .venv, Go/helm/kubeconform/terraform/tflint/hadolint/promtool into .tools/)
- `make check` = ruff + pytest + go test + vitest + helm lint/kubeconform + terraform validate + hadolint + promtool
- Python: `.venv/bin/python -m draftlab ...`; Go: `PATH=$PWD/.tools/go/bin:$PATH go test ./...` in gateway/

## Key definitions
- **alpha** (expected acceptance) at a position = Σ_x min(p(x), q(x)) = 1 − TV(p, q), p target, q draft.
  Greedy acceptance = argmax q == argmax p. Expected tokens per target call for k drafts:
  (1 − α^(k+1)) / (1 − α) (Leviathan et al. 2023). Speedup also needs the draft/target cost ratio c.
- **kind**: thinking | text | code (fenced blocks inside text) | tool_use:<ToolName> | tool_result.
- Splits are by run (hash of run id), never by block, so no run leaks between train and test.
