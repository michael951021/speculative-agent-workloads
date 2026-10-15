# Speculative Decoding for Agent Workloads

Speculative decoding measured on this machine's own agent traffic. `data/raw/agent_turns.jsonl`
holds ~50k content blocks (thinking, text, tool calls, tool results) produced by local
Qwen 27B coding agents. The project builds the full stack — JAX target + spec-decoding engine,
trained drafters, Triton verify kernel, Go gateway with Prometheus metrics, Ray sweeps,
LLM-as-judge quality check, deploy configs, and a React explorer — to answer: **how many
agent tokens could a draft model have guessed, per kind of content, and what speedup does
that buy?** Results and figures: [`REPORT.md`](REPORT.md), `results/`.

## Results

On 14 held-out agent runs, against a Qwen2.5-1.5B-Instruct target:

| drafter | alpha (95% CI) | measured CPU speedup |
|---|---|---|
| n-gram lookup | 0.3228 [0.3074, 0.3389] | 1.50× at k=8 |
| Qwen2.5-0.5B + LoRA | 0.7120 [0.7061, 0.7187] | 1.58× at k=2, 1.25× at k=4 |
| LoRA + DPO | 0.7135 [0.7078, 0.7201] | no difference from LoRA |
| distilled tiny model | 0.0114 | — |

Tool calls are the most predictable content and thinking the least. Speedups are wall clock,
head to head with plain decoding in one process (`tools/bench_speedup.py`), and they are CPU
numbers. [`REPORT.md`](REPORT.md) has the per-kind tables, the full draft-length grid, a
quality judge, and a corrections section for five bugs found in an audit of the first
version. `tools/check_report.py` recomputes every number in it from `results/`.

The [interactive report](report/index.html) is the earlier engineering write-up, a snapshot
from before the speed benchmark finished.

## Architecture

```mermaid
flowchart LR
  RAW[data/raw/agent_turns.jsonl] --> DATA[speculative_agent_workloads.data<br/>corpus → Parquet, splits by run]
  DATA --> ALPHA[speculative_agent_workloads.eval.alpha<br/>α = 1 − TV(p, q) per position]
  DATA --> DRAFTERS[speculative_agent_workloads.drafters<br/>ngram · distill (JAX) · lora (PyTorch/FSDP) · dpo (TRL)]
  TARGET[speculative_agent_workloads.jaxlm<br/>Qwen2 in JAX + KV cache + spec.py engine] --> SWEEP
  DRAFTERS --> SWEEP[speculative_agent_workloads.sweep<br/>Ray grid: prompt × k × T × drafter]
  KERN[speculative_agent_workloads.kernels<br/>Triton fused verify/alpha] --> SWEEP
  SWEEP --> RES[results/*.parquet,json + figs]
  RES --> EXPORT[speculative_agent_workloads.export_ui]
  RES --> JUDGE[speculative_agent_workloads.eval.judge<br/>LLM-as-judge, OpenAI-compatible API]
  TARGET --> SERVE[speculative_agent_workloads.serve<br/>FastAPI engine :8100]
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
python -m speculative_agent_workloads.data build                   # corpus → data/corpus/*.parquet
python -m speculative_agent_workloads.eval.alpha --drafter ngram   # per-kind α on the 0.5B target (CPU, minutes)
python -m speculative_agent_workloads.sweep --temps 0 --ks 1 8      # small slice of the Ray grid
python -m speculative_agent_workloads.export_ui --no-replays       # web/public/data/*.json
python -m speculative_agent_workloads.serve                        # engine on :8100
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
| JAX (XLA, optax, pytrees) | `speculative_agent_workloads/jaxlm/model.py`, `speculative_agent_workloads/jaxlm/spec.py`, `speculative_agent_workloads/drafters/distill.py`, `speculative_agent_workloads/eval/alpha.py` |
| PyTorch + FSDP | `speculative_agent_workloads/drafters/lora.py` (PEFT on CPU; FSDP wrapper for the GPU run) |
| LoRA / QLoRA | `speculative_agent_workloads/drafters/lora.py` (PEFT r=16/α32, rank-stable eval) |
| DPO | `speculative_agent_workloads/drafters/dpo.py` (TRL DPO on spec-decoding pairs) |
| Triton | `speculative_agent_workloads/kernels/verify.py` (fused verify + alpha), `speculative_agent_workloads/kernels/bench.py` |
| Ray | `speculative_agent_workloads/sweep.py` (local-mode grid, per-prompt resume, sidecar fold) |
| Go | `gateway/` (OpenAI-compatible proxy: auth, 429s, routing, breaker, SSE tee, /metrics, /healthz, /ui) |
| Prometheus / Grafana | `deploy/prometheus/` (rules + recordings), `deploy/grafana/` (panels) |
| Kubernetes / Helm | `deploy/helm/speculative-agent-workloads/` (gateway+engine charts, HPA, vLLM optional), `deploy/kind/e2e.sh` |
| Terraform / GKE | `deploy/terraform/` (root + GKE module: WI, private clusters, Binary Authorization) |
| React / TypeScript | `web/` (Vite + React 19 + Recharts explorer of results/) |
| LLM-as-judge | `speculative_agent_workloads/eval/judge.py` (blind A/B, both orders, cached, capped) |
| OpenAI API | `speculative_agent_workloads/eval/judge.py` (openai SDK → `$OLLAMA_BASE_URL`), `gateway/` (OpenAI-compatible endpoints) |
| Docker | `deploy/docker/` (gateway distroless, engine slim, vLLM spec-config) — linted, not built |

## Layout

```
speculative_agent_workloads/   data, jaxlm (target+engine), drafters, kernels, serve, eval, sweep.py, export_ui.py
gateway/    Go module (proxy + metrics + /ui)
web/        Vite + React + TS explorer
deploy/     docker, helm, terraform, grafana, prometheus, kind
tools/      bootstrap helpers, host-only export, bench_speedup.py, check_report.py
results/    committed numbers (parquet/json) + figs/
scripts/    gpu_run.sh (host-only)
```

## Package and repository names

The repository and distribution are named speculative-agent-workloads.
Python imports and module commands use speculative_agent_workloads.
Saved report evidence preserves the source paths and hashes from its historical revision.

## Rename verification

The rename was checked in an isolated checkout, without changing the active experiment.
Python verification: 90 passing tests across the existing suite; 15 skips for unavailable
local transcripts, datasets, model weights or GPU support. Ruff passed.
The Go gateway tests and 22 dashboard tests passed; the dashboard production build passed.
Helm lint and CPU/GPU schema validation passed (12 and 15 resources), alongside
Prometheus configuration/rule tests, Dockerfile lint and shell lint.
The printable report was regenerated from the unchanged October 6 research snapshot.
