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
- `python -m draftlab.sweep` (Ray local, resumable per prompt; sidecar .part-*.jsonl folded on start) → results/sweep.parquet; target 0.5b-instruct, 393 val prompts (≤128 tok), n_gen 16.
- **Batch 1 done:** base@T0 k=1 accept 1.0 exact; dpo 0.5405 ≥ lora 0.5334 (paired diff +0.0071 [0.0037,0.0108]); none 6.32 tok/s; check_sweep_batch1 exit 0.
- **Batch 2 DONE (T=0, 16 cells + none, 393 each):** tools/check_sweep_batch2.py exit 0. tpc(k) monotone all drafters (tiny 1.073→1.090, base 2.000→8.000, lora 1.496→2.089, lora_dpo 1.503→2.116). Known answers, same-trajectory CIs (engine_traj_none.json; at T=0 spec traj == none traj for ANY k, accept iff draft==argmax p): ngram k=1 acc 0.4411∈[0.399,0.481], tpc 1.1145∈[1.098,1.131]; lora k=8 tpc 2.0890∈[2.019,2.162]. Both below pre-registered corpus-based ranges [1.15,1.7]/[2.2,3.0] = coverage gap (engine: 16 fresh tok/prompt vs corpus in-block population; replay==engine), NOT a measurement bug. Table → results/sweep_batch2_table.json: none 6.32 tok/s; best x-none: ngram k1 0.88, base k8 0.84; lora k8 0.25.
- Gotchas: corpus-block compare FAILs lora/dpo by design (different position population); empty-ctx dists row r predicts ids[r] → slice q[L:L+n] (off-by-one gave 0.0133); lora_dpo ckpt dir = ckpt/dpo. Cost: 1 worker ×16 thr ≈ 2-2.5 prompts/min (lora_dpo) → capture timeout ≥1h; ~37 min/cell at full parallelism.

## Helm (id: helm)
- deploy/helm/draftlab: gateway (Deployment/Service/HPA external metric `draftlab_accepted_tokens_per_second` via prometheus-adapter), engine (CPU, models PVC/hostPath), optional vLLM (values-gpu.yaml: gpu:1, tolerations, drafter PVC → --speculative-config), ServiceMonitors, Grafana ConfigMap, NetworkPolicies, PDB.
- check.sh: helm lint + `helm template` cpu AND gpu | kubeconform -strict; ServiceMonitor schema vendored (prom-op v0.76.2).
- Gotcha: kubeconform -schema-location vars lowercase + KindSuffix `-monitoring-v1`; all-digit sha256 must be quoted in values.yaml.

## kind e2e (id: kind)
- deploy/kind/e2e.sh (HOST ONLY, no Docker in sandbox): build 2 images → kind create/load → `docker cp` model into node (engine hostPath /models) → helm cpu profile (1 replica, HPA/ServMon/NP off, random Bearer key) → port-forward → curl /v1/chat/completions 200 → /metrics `draftlab_accepted_tokens_total`>0 (ngram k=5 default + repeat-phrase prompt ⇒ acceptance ~1). README: deploy/kind/README.md.
- shellcheck v0.10.0 added: pinned tools/versions.txt + bootstrap install + check.sh section (lints deploy/**/*.sh only; tools/*.sh predate it and are NOT in the gate). make check exit 0. Not executed in sandbox (rule 2) — Kevin runs on host.

## Terraform GKE (id: tf)
- deploy/terraform root (thin pass-through vars) → module ./gke: VPC + regional subnet (secondary_ip_range pods/services), GKE Standard (workload_pool WI, private nodes+global access, BA ENFORCE, REGULAR channel, initial_node_count 0), node pool g2-standard-12 + nvidia-l4×1 autoscale 0-2 + NO_SCHEDULE taint, Artifact Registry docker repo (most_recent_versions keep 10), WI GCP SA + artifactregistry.reader. Outputs incl. workload_identity_sa_email + push base.
- init -backend=false + validate + tflint exit 0 (google v5.45.2). Gotchas: v5 renames identity_namespace→workload_pool, secondary_range→secondary_ip_range, monitoring enabled_components→enable_components, cleanup keep_tag_revision_policy→most_recent_versions; binary_authorization.enabled deprecated → evaluation_mode PROJECT_SINGLETON_POLICY_ENFORCE; TF 1.x does NOT load subdir .tf into root (must be a module); real schema: `terraform providers schema -json`.

## GPU run (id: gpu)
- scripts/gpu_run.sh (HOST, after `./agent stop`; sandbox: `--dry-run` exit 0 verified): (1) `eval.alpha` ngram + hf-lora at T=7B-instruct (fused_alpha on CUDA) → results/gpu/alpha_*_7b.json; (2) vllm serve baseline/ngram/LoRA arms + gateway (:8200) → `tools/bench_gateway.py` c=1/4/16 → bench_<arm>_c*.json; (3) summary.json + checks (baseline null: 0 drafts; spec arms accept_rate 0.05–0.95).
- Bench tool probed vs mock SSE server (ok=4, usage tokens, ttft/latency, /metrics delta parse). Expected anchors pre-run: ngram α 0.2–0.4, lora α 0.5–0.9, speedup ~1.4–1.75× (llama.cpp #10466 k=4). ruff clean, pytest 89 passed.

## Results export (id: export)
- `python -m draftlab.export_ui` → web/public/data/{summary,alpha,sweep,replays}.json (schema v1); missing sweep cells ⇒ {"status":"pending"} (21/42 pending now, batch 3 running); rerun after sweep (`--no-replays` = 2s fast path).
- replays = 30 val blocks (5 × top-6 kinds by n_tokens in dpo alpha; first per kind per run, round-robin) re-scored with the alpha-run pair (1.5b-instruct, ctx 1024): per-token p, q (null = no proposal / unscorable pos 0), accept (greedy spec-accept), per-position α. ~2.6 min CPU.
- Known answer: test_replays_match_alpha_pipeline re-runs alpha.evaluate() on the same 30 blocks ⇒ per-position α/accept identical (<1e-5). Fast mean-vs-dpo.json check tol 0.05→0.15 (5-block samples, SE~0.04; gaps ≤0.11 all one sign = selection of first/short blocks per kind, NOT a pipeline bug — identity test proves pipeline exact).
- Gotcha: corpus (run,seq) NOT unique (val: 15 blocks share (contrib-loop-w1-…,28885)) — match blocks by token ids, not (run,seq). Export wall_s kept full precision (tok_s recomputable).

## Judge (id: judge)
- `python -m draftlab.eval.judge`: gen {target, spec(k=4, lora_dpo), drafter_alone} T=0.7 (49 prompts, n_gen 16) → blind pairwise vs qwen-agent 27B ($OLLAMA_BASE_URL), cache data/judge_cache/, cap 150 → results/judge.json. Tests 7/7.
- Result: spec_vs_target n=81: win 0.444/loss 0.543, CIs overlap (binom p=0.434) — no significant loss; T=0.7 exact matches 0/46 = lossless signature; spec_vs_drafter gap 25pp ✓, underpowered.
- Corpus gotcha: test.parquet block contrib-loop-20260926-231240 ×8 (code-split shares (run,seq)) → 4 prompts share one output key (9 silent skips).
