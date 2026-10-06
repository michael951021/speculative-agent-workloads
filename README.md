# draftlab

Speculative decoding measured on this machine's own agent traffic. `data/raw/agent_turns.jsonl`
holds ~50k content blocks (thinking, text, tool calls, tool results) produced by local
Qwen 27B coding agents. draftlab builds the full stack — JAX target + spec-decoding engine,
trained drafters, Triton verify kernel, Go gateway with Prometheus metrics, Ray sweeps,
LLM-as-judge quality check, deploy configs, and a React explorer — to answer: **how many
agent tokens could a draft model have guessed, per kind of content, and what speedup does
that buy?** Results and figures: [`REPORT.md`](REPORT.md), `results/`.

## Engineering report

The [interactive report](report/index.html) and [printable PDF](report/Draftlab-report.pdf)
cover the system, held-out results and lessons from six fixes in the commit history.
This October 6 snapshot includes the evaluation corrections and marks incomplete
benchmarks as pending. Aggregate evidence and CSV downloads are included in the report.

## Architecture

```mermaid
flowchart LR
  RAW[data/raw/agent_turns.jsonl] --> DATA[draftlab.data<br/>corpus → Parquet, splits by run]
  DATA --> ALPHA[draftlab.eval.alpha<br/>α = 1 − TV(p, q) per position]
  DATA --> DRAFTERS[draftlab.drafters<br/>ngram · distill (JAX) · lora (PyTorch/FSDP) · dpo (TRL)]
  TARGET[draftlab.jaxlm<br/>Qwen2 in JAX + KV cache + spec.py engine] --> SWEEP
  DRAFTERS --> SWEEP[draftlab.sweep<br/>Ray grid: prompt × k × T × drafter]
  KERN[draftlab.kernels<br/>Triton fused verify/alpha] --> SWEEP
  SWEEP --> RES[results/*.parquet,json + figs]
  RES --> EXPORT[draftlab.export_ui]
  RES --> JUDGE[draftlab.eval.judge<br/>LLM-as-judge, OpenAI-compatible API]
  TARGET --> SERVE[draftlab.serve<br/>FastAPI engine :8100]
  KERN --> SERVE
  SERVE --> GW[gateway/ (Go)<br/>OpenAI-compatible proxy, auth, rate limit,<br/>breaker, /metrics, /ui]
  EXPORT --> WEB[web/ Vite+React+TS explorer]
  GW --> WEB
  GW --> PROM[Prometheus rules → Grafana panels]
  GW --> DEPLOY[deploy/: Docker, Helm + kind e2e,<br/>Terraform/GKE]
```

## Quickstart

```bash
make bootstrap                                  # uv .venv (py3.12), go, helm, terraform, ...
python -m draftlab.data build                   # corpus → data/corpus/*.parquet
python -m draftlab.eval.alpha --drafter ngram   # per-kind α on the 0.5B target (CPU, minutes)
python -m draftlab.sweep --temps 0 --ks 1 8      # small slice of the Ray grid
python -m draftlab.export_ui --no-replays       # web/public/data/*.json
python -m draftlab.serve                        # engine on :8100
cd gateway && go run .                          # gateway on :8200, serves /ui from web/dist
cd web && npm install && npm run build          # explorer → web/dist
make check                                      # ruff + pytest + go test + vitest + lint gates
python tools/check_report.py                    # REPORT.md numbers vs results/
```

CPU only in the sandbox: JAX `jax[cpu]`, torch CPU wheels, Triton under `TRITON_INTERPRET=1`;
cap threads with `OMP_NUM_THREADS=16 XLA_FLAGS=--xla_cpu_multi_thread_eigen=true`.
GPU-scale runs live in `scripts/gpu_run.sh` (host-only).

## Skills map

| skill | where it lives |
|---|---|
| JAX (XLA, optax, pytrees) | `draftlab/jaxlm/model.py`, `draftlab/jaxlm/spec.py`, `draftlab/drafters/distill.py`, `draftlab/eval/alpha.py` |
| PyTorch + FSDP | `draftlab/drafters/lora.py` (PEFT on CPU; FSDP wrapper for the GPU run) |
| LoRA / QLoRA | `draftlab/drafters/lora.py` (PEFT r=16/α32, rank-stable eval) |
| DPO | `draftlab/drafters/dpo.py` (TRL DPO on spec-decoding pairs) |
| Triton | `draftlab/kernels/verify.py` (fused verify + alpha), `draftlab/kernels/bench.py` |
| Ray | `draftlab/sweep.py` (local-mode grid, per-prompt resume, sidecar fold) |
| Go | `gateway/` (OpenAI-compatible proxy: auth, 429s, routing, breaker, SSE tee, /metrics, /healthz, /ui) |
| Prometheus / Grafana | `deploy/prometheus/` (rules + recordings), `deploy/grafana/` (panels) |
| Kubernetes / Helm | `deploy/helm/draftlab/` (gateway+engine charts, HPA, vLLM optional), `deploy/kind/e2e.sh` |
| Terraform / GKE | `deploy/terraform/` (root + GKE module: WI, private clusters, Binary Authorization) |
| React / TypeScript | `web/` (Vite + React 19 + Recharts explorer of results/) |
| LLM-as-judge | `draftlab/eval/judge.py` (blind A/B, both orders, cached, capped) |
| OpenAI API | `draftlab/eval/judge.py` (openai SDK → `$OLLAMA_BASE_URL`), `gateway/` (OpenAI-compatible endpoints) |
| Docker | `deploy/docker/` (gateway distroless, engine slim, vLLM spec-config) — linted, not built |

## Layout

```
draftlab/   data, jaxlm (target+engine), drafters, kernels, serve, eval, sweep.py, export_ui.py
gateway/    Go module (proxy + metrics + /ui)
web/        Vite + React + TS explorer
deploy/     docker, helm, terraform, grafana, prometheus, kind
tools/      bootstrap helpers, host-only export, checks (check_report.py, check_sweep_*.py)
results/    committed numbers (parquet/json) + figs/
scripts/    gpu_run.sh (host-only)
```
