# draftlab — current state

## Toolchain (id: boot)
- `make bootstrap`; `make check` = ruff+pytest (others SKIP if absent). capture.py: no `VAR=x cmd` prefixes.

## Models / Corpus (id: models, corpus)
- `models.py fetch` → models/{0.5b,1.5b-instruct,0.5b-instruct}; `tiny_config()` 2L d=64 V=512.
- `python -m draftlab.data build` → data/corpus/{train,val,test}.parquet (code split, dedupe, trunc 4096); 45257/667/566.

## N-gram (id: ngram)
- NGramMatcher (n=1..4, no self-match) → results/ngram.json; greedy accept@k8 0.329.

## JAX / spec (id: jaxmodel, spec, engine)
- jaxlm/model.py pure-pytree Qwen2 (bf16→f32, padded KV); parity prefill 1.3e-4, 32 greedy identical.
- jaxlm/spec.py: one target call scores m drafts + bonus; accept u·q<p, residual max(0,p−q); T=0==greedy; rollback advances _last by m (tested). serve/app.py: `python -m draftlab.serve` :8100.

## Alpha metric (id: alpha)
- `python -m draftlab.eval.alpha` (ngram|jax|hf); row i predicts i+1. CE==HF <1e-3; top-1 42.4%; repeat 63/64, shuffle collapses; ngram α 0.3341, tpc@8 1.50; 0.5B control 0.7675.

## Drafters (id: distill, lora, dpo)
- distill: top-64+tail 4L d=256 KL → ckpt/tiny/export; val KL 3.79→2.63 ✅. lora: PEFT r=16/α32 SFT; α 0.7306 ✅; tpc@8 3.49. dpo: T=1 spec pairs (1.5b) → data/pairs/dpo.jsonl; TRL β=0.1; α 0.7319, CIs disjoint ✅.

## Kernels (id: triton)
- kernels/verify.py fused_verify + fused_alpha; tests 7/7 (TRITON_INTERPRET=1).

## Gateway (id: gateway)
- gateway/ Go: auth/429/routing/breaker/SSE tee//metrics//healthz//ui; go test 8/8; live smoke ✅.

## Observability / docker (id: obs, docker)
- prometheus rules+recordings (promtool OK; windows LEFT-CLOSED); grafana 8 panels.
- deploy/docker/: gateway distroless :8200, engine slim :8100, vllm --speculative-config; hadolint 3/3

## Ray sweep (id: sweep)
- `python -m draftlab.sweep` (Ray local, resume per prompt, sidecar .part-*.jsonl folded) → results/sweep.parquet; 0.5b-instruct, 393 val prompts, n_gen 16.
- **ALL 3 BATCHES DONE: 42 cells (21 T=0 + 21 T=0.7) × 393; check_sweep_batch1/2/3 exit 0.**
- T=0: base k1 acc 1.0 exact; tpc@8 ngram 1.197/tiny 1.090/base 8.0/lora 2.089/dpo 2.116; dpo≥lora +0.0071 [0.0037,0.0108]; known answers in same-trajectory CIs (engine_traj_none.json); below corpus ranges = coverage gap, not bug.
- T=0.7: ngram LOWER all k (.4411→.3265 k1, ranges ok); base acc 1.0 + tpc 2/2.667/4/8 EXACT; model drafters HIGHER (tiny +.076, lora +.164, dpo +.188 max) — structural: T=0 always rejects non-argmax matches, T>0 accepts w.p. min(1,P7/q)>0. Pre-run tiny/lora/dpo gates refuted → recalibrated [a0−0.10, a0+G] (G=.15/.25); dpo k8 out-of-sample PASS (.3269∈[.08,.43]).
- ngram T=0.7 ref (tools/check_ngram_t07_ref.py, hand E[accept]=P7[d]): .3265 vs .3295, CI95∋0, == parquet ✓.
- Speedups vs none (6.32/6.29 tok/s T=0/T=0.7): best ngram k8 1.12/1.09; base k8 .84/.71; lora k8 .25/.36 (CPU draft cost > gain). Tables: results/sweep_batch{2,3}_table.json.
- Gotchas: ~30-37 min/cell; bg bash dies at 30-min default → relaunch w/ Bash timeout ≥7200000 + capture --timeout ≥2h; dpo ckpt dir = ckpt/dpo.

## Helm / kind (id: helm, kind)
- deploy/helm/draftlab: gateway HPA ext metric, engine CPU, optional vLLM (values-gpu.yaml), ServMon/NP; check.sh = lint + template cpu&gpu | kubeconform -strict.
- deploy/kind/e2e.sh HOST-only: build → kind → cp model → helm cpu → curl 200 → accepted>0; shellcheck in gate.

## Terraform (id: tf)
- deploy/terraform root→module gke: VPC, GKE (WI, private, BA ENFORCE), g2+L4 pool, AR, WI SA; validate+tflint exit 0.

## GPU run (id: gpu)
- scripts/gpu_run.sh HOST-only (dry-run OK): 7B alpha + vllm arms + bench_gateway c=1/4/16 → results/gpu/; anchors ngram α .2–.4, lora .5–.9, ~1.4–1.75×.

## Results export (id: export)
- `python -m draftlab.export_ui` → web/public/data/{summary,alpha,sweep,replays}.json (schema v1); missing sweep cells ⇒ {"status":"pending"} (21/42 pending now, batch 3 running); rerun after sweep (`--no-replays` = 2s fast path).
- replays = 30 val blocks (5 × top-6 kinds by n_tokens in dpo alpha; first per kind per run, round-robin) re-scored with the alpha-run pair (1.5b-instruct, ctx 1024): per-token p, q (null = no proposal / unscorable pos 0), accept (greedy spec-accept), per-position α. ~2.6 min CPU.
- Known answer: test_replays_match_alpha_pipeline re-runs alpha.evaluate() on the same 30 blocks ⇒ per-position α/accept identical (<1e-5). Fast mean-vs-dpo.json check tol 0.05→0.15 (5-block samples, SE~0.04; gaps ≤0.11 all one sign = selection of first/short blocks per kind, NOT a pipeline bug — identity test proves pipeline exact).
- Gotcha: corpus (run,seq) NOT unique (val: 15 blocks share (contrib-loop-w1-…,28885)) — match blocks by token ids, not (run,seq). Export wall_s kept full precision (tok_s recomputable).

## Web explorer (id: web)
- web/ Vite+React19+TS+Recharts3: 4 tabs (Overview/Heatmap/Speedup/Replay), light/dark, base './', no router (static /ui mount); `npm run build` → dist/ incl. data/; vitest 22/22 cross-check rendered values vs the REAL public/data/*.json; pre-registered ranges all held (best α .7319∈[.72,.75], expected best 2.30×∈[2,2.5], measured best 1.12×∈[1.05,1.2], lora×text .658∈[.5,.8]).
- check.sh: web build+vitest section; /ui smoke 200 (html/json/js); palette validator light+dark PASS (light contrast WARN → legend+table views). Leviathan closed form unit-tested (α=.5,k=3→1.875). Gotchas: Area only renders in ComposedChart (empty in LineChart); RO polyfill must fire synchronously; 8 genuinely degenerate CIs (width 0) → assert ≥0.

## Judge (id: judge)
- `python -m draftlab.eval.judge`: blind pairwise vs qwen-agent 27B, cap 150, cache → results/judge.json. spec_vs_target n=81 .444/.543 CIs overlap (p=.434); T=0.7 exact 0/46 = lossless.
