# speculative-agent-workloads — how predictable is a coding agent?

Speculative decoding measured on this machine's own agent traffic. `data/raw/agent_turns.jsonl` holds ~50k content
blocks (thinking, text, tool calls, tool results) that the local Qwen 27B agents produced or read. speculative-agent-workloads asks: how
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
- `speculative_agent_workloads/data/` corpus → Parquet, labels, splits · `speculative_agent_workloads/jaxlm/` Qwen2 in JAX, KV cache, spec decoding
- `speculative_agent_workloads/drafters/` ngram, distill (JAX/Optax), lora (PyTorch PEFT + FSDP), dpo (TRL)
- `speculative_agent_workloads/kernels/` Triton verify + alpha kernels · `speculative_agent_workloads/serve/` FastAPI engine server
- `speculative_agent_workloads/eval/` alpha metric, judge · `speculative_agent_workloads/sweep.py` Ray · `speculative_agent_workloads/export_ui.py`
- `gateway/` Go module (OpenAI-compatible proxy, auth, rate limit, breaker, /metrics, serves web/dist at /ui)
- `web/` Vite + React + TypeScript explorer · `deploy/{docker,helm,terraform,grafana,prometheus,kind}`
- `tools/` bootstrap + host-only export · `scripts/gpu_run.sh` host GPU run · `results/` · `REPORT.md`

## Commands
- `make bootstrap` (uv, Python 3.12 .venv, Go/helm/kubeconform/terraform/tflint/hadolint/promtool into .tools/)
- `make check` = ruff + pytest + go test + vitest + helm lint/kubeconform + terraform validate + hadolint + promtool
- Python: `.venv/bin/python -m speculative_agent_workloads.<module> ...`; Go: `PATH=$PWD/.tools/go/bin:$PATH go test ./...` in gateway/

## Key definitions
- **alpha** (expected acceptance) at a position = Σ_x min(p(x), q(x)) = 1 − TV(p, q), p target, q draft.
  Greedy acceptance = argmax q == argmax p. Expected tokens per target call for k drafts:
  (1 − α^(k+1)) / (1 − α) (Leviathan et al. 2023). Speedup also needs the draft/target cost ratio c.
- **kind**: thinking | text | code (fenced blocks inside text) | tool_use:<ToolName> | tool_result.
- Splits are by run (hash of run id), never by block, so no run leaks between train and test.

## Known answers (every metric ticket checks against these; see prompts: "Numbers are claims")
- **Alignment:** logits row i predicts token i+1. Any per-position metric must reproduce
  `transformers` `model(ids, labels=ids).loss` (HF shifts internally) within 1e-3 on the same block.
- **Target sanity:** Qwen2.5-0.5B top-1 next-token accuracy on corpus val blocks ≈ 40–70% (57.5% measured on
  372 tokens, 2026-09-30). Under 20% means an alignment bug, not a weak model.
- **Model size order:** Qwen2 wikitext fp16 PPL 0.5B 15.1 · 1.5B 10.4 · 7B 7.9 (Qwen2 GGUF card). A bigger
  target must show lower loss on the corpus.
- **Alpha anchors:** Leviathan 2023 Table 3: drafters ~100x smaller give α 0.5–0.9, even a bigram α ≈ 0.2.
  llama.cpp (discussion #10466) Qwen2.5-0.5B→7B, k=4: 1.41x general text, 1.75x code. Expect tool_use ≥ text.
- **Analytic cases:** q = p → α = 1; q uniform → α = Σ min(p, 1/V) exactly; q = λp + (1−λ)u → α ≥ λ.
- **Planted effect / null control:** a block repeated verbatim → ngram acceptance on the copy ≈ 1; tokens shuffled
  within a block → ngram acceptance collapses. A drafter trained on agent traffic must gain more on agent text
  than on wikitext (otherwise it learned nothing agent-specific).
- **Engine vs metric:** SpecDecoder's measured accepted/drafted on real prompts must fall inside the alpha
  metric's predicted tokens-per-call CI for the same pair and k.
