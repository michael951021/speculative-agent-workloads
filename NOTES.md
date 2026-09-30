# draftlab — current state

## Toolchain (id: boot)
- `make bootstrap`: uv, Py3.12 .venv, torch CPU, .tools/; `make check` = ruff + pytest (others SKIP if absent); capture.py: no `VAR=x cmd` prefixes.

## Models / Corpus (id: models, corpus)
- `models.py fetch` → models/{0.5b-instruct,1.5b-instruct,0.5b} (shared vocab 151936); `tiny_config()` = 2L d=64 vocab-512.
- `python -m draftlab.data build` → data/corpus/{train,val,test}.parquet (code split, dedupe, trunc 4096, runs by sha1); 45257/667/566 blocks.

## N-gram (id: ngram)
- NGramMatcher (n=1..4, first occurrence, no self-match) → results/ngram.json; greedy accept@k=8 0.329 (tool_use 0.548 best, text 0.199 worst).

## JAX model / spec / engine (id: jaxmodel, spec, engine)
- jaxlm/model.py: pure-pytree Qwen2, bf16→f32, padded KV, one jitted step; 0.5B parity prefill 1.3e-4, 32 greedy identical.
- jaxlm/spec.py: one target call scores m drafts + bonus; accept u·q[t]<p[t], residual max(0,p−q); T=0==greedy; full-accept rollback advances `_last` by m (regression-tested).
- serve/app.py FastAPI OpenAI-compatible (SSE, /v1/models); `python -m draftlab.serve` (0.5b-instruct, :8100); tiny → ToyTokenizer.

## Alpha metric (id: alpha)
- `python -m draftlab.eval.alpha` (ngram|jax|hf); row i predicts i+1. Known: CE==HF labels <1e-3; top-1 val 42.4%; uniform-q=1-TV; repeat 63/64, shuffle collapses.
- 400 val: ngram α 0.3341 (greedy 0.3681), tpc@8 1.50 (Edit 0.56 > tool_result 0.33 > code 0.26 > text 0.19); tiny α 0.0105; 0.5B control 0.7675 → weak drafter, not metric.

## Drafters (id: distill, lora, dpo)
- drafters/distill.py: teacher top-64+tail, 4L d=256 KL AdamW → ckpt/tiny/export; 1200 steps val KL 3.79→2.63 ✅.
- drafters/lora.py: PEFT r=16/α32, 7 targets, SFT on packed blocks; --fsdp ⇒ bit-identical; step_loss==HF <1e-5; 300 steps → ckpt/lora. α 0.7306 [0.7302,0.7308] ✅; Edit 0.82 > Bash 0.77 > tool_result 0.74 > code 0.72 > text 0.66; tpc@8 3.49.
- drafters/dpo.py: T=1 spec pairs (1.5b target) → data/pairs/dpo.jsonl; TRL DPOTrainer β=0.1; 2000 pairs (accept 0.786); margin −3.79→−3.53; α 0.7306→0.7319 (CIs disjoint ✅).

## Kernels (id: triton)
- kernels/verify.py: fused_verify (online softmax, accept p>u·q, residual max(0,p−q)@first-reject), fused_alpha; tests 7/7 (TRITON_INTERPRET=1).

## Gateway (id: gateway)
- gateway/ Go 1.26: Bearer auth, 429, routing, breaker, SSE tee, /metrics (ttft, tok/s, draft/accepted{model}), /healthz, /ui; `go test ./...` 8/8; live smoke ✅.

## Observability (id: obs)
- deploy/prometheus: prometheus.yml + rules.yml (4 recordings, LowDraftAcceptance, HighTTFTp95) + rules_test.yml; promtool OK. Gotcha: windows LEFT-CLOSED; recording tests need `__name__`.
- deploy/grafana/draftlab.json: 8 panels.

## Deploy docker (id: docker)
- deploy/docker/: gateway (distroless nonroot :8200), engine (slim `.[jax,serve]` :8100), vllm (vllm-openai:v0.11.2; entrypoint adds --speculative-config from $DRAFTER_PATH). hadolint 3/3. Gotcha: inline `# hadolint ignore=DLxxxx` must be only text on its line.

## Ray sweep (id: sweep)
- `python -m draftlab.sweep` (Ray local, resumable by (drafter,k,temp)) → results/sweep.parquet; 488 val prompts (100/kind ≤128 tok), n_gen 16.
- Known: base@T0 k=1 accept exactly 1.0; k≥2 expect ≥0.99. Cost: 6.4 tok/s @16 thr → 42 cells ≈ 20 CPU-h.

## Helm (id: helm)
- deploy/helm/draftlab: gateway (Deployment/Service/HPA external metric `draftlab_accepted_tokens_per_second` via prometheus-adapter), engine (CPU, models PVC/hostPath), optional vLLM (values-gpu.yaml: gpu:1, tolerations, drafter PVC → --speculative-config), ServiceMonitors, Grafana ConfigMap, NetworkPolicies, PDB.
- check.sh: helm lint + `helm template` cpu AND gpu | kubeconform -strict; ServiceMonitor schema vendored (prom-op v0.76.2).
- Gotcha: kubeconform -schema-location vars lowercase + KindSuffix `-monitoring-v1`; all-digit sha256 must be quoted in values.yaml.

## Terraform GKE (id: tf)
- deploy/terraform root (thin pass-through vars) → module ./gke: VPC + regional subnet (secondary_ip_range pods/services), GKE Standard (workload_pool WI, private nodes+global access, BA ENFORCE, REGULAR channel, initial_node_count 0), node pool g2-standard-12 + nvidia-l4×1 autoscale 0-2 + NO_SCHEDULE taint, Artifact Registry docker repo (most_recent_versions keep 10), WI GCP SA + artifactregistry.reader. Outputs incl. workload_identity_sa_email + push base.
- init -backend=false + validate + tflint exit 0 (google v5.45.2). Gotchas: v5 renames identity_namespace→workload_pool, secondary_range→secondary_ip_range, monitoring enabled_components→enable_components, cleanup keep_tag_revision_policy→most_recent_versions; binary_authorization.enabled deprecated → evaluation_mode PROJECT_SINGLETON_POLICY_ENFORCE; TF 1.x does NOT load subdir .tf into root (must be a module); real schema: `terraform providers schema -json`.

## Judge (id: judge)
- `python -m draftlab.eval.judge`: gen {target, spec(k=4, lora_dpo), drafter_alone} T=0.7 (49 prompts, n_gen 16) → blind pairwise vs qwen-agent 27B ($OLLAMA_BASE_URL), cache data/judge_cache/, cap 150 → results/judge.json. Tests 7/7.
- Result: spec_vs_target n=81: win 0.444/loss 0.543, CIs overlap (binom p=0.434) — no significant loss; T=0.7 exact matches 0/46 = lossless signature; spec_vs_drafter gap 25pp ✓, underpowered.
- Corpus gotcha: test.parquet block contrib-loop-20260926-231240 ×8 (code-split shares (run,seq)) → 4 prompts share one output key (9 silent skips).
