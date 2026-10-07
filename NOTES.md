# speculative-agent-workloads — current state

## Toolchain (id: boot, dl-cifix)
- make bootstrap; make check = ruff+pytest (others SKIP if absent); capture.py: no VAR=x prefixes (use env), won't reuse an existing log path.
- CI clone (no data/models/ckpt): 4 data tests skipif-guarded (sweep: heldout+val+0.5b-instr; replay: val+1.5b-instr+ckpt/dpo); near_alpha_reference: only dpo.json kinds (no code). Clone 7p/3s; all 4 pass in /work.

## Models / Corpus (id: models, corpus)
- models.py fetch → models/{0.5b,1.5b-instruct,0.5b-instruct}; tiny 2L d=64 V=512; data build → corpus parquet (code split, dedupe, trunc 4096); 45257/667/566.

## N-gram (id: ngram)
- NGramMatcher (n=1..4, no self-match) → results/ngram.json; greedy accept@k8 0.329.

## JAX / spec (id: jaxmodel, spec, engine)
- jaxlm/model.py pure-pytree Qwen2 (bf16→f32, padded KV); parity prefill 1.3e-4, 32 greedy identical.
- jaxlm/spec.py: 1 target call scores m drafts+bonus; accept u·q<p, residual max(0,p−q); T=0==greedy; rollback advances _last by m. serve :8100.

## Alpha metric (id: alpha)
- eval.alpha (ngram|jax|hf); row i→i+1; CE==HF <1e-3; top-1 42.4%; repeat 63/64, shuffle collapses; ngram α .3341 tpc@8 1.50; 0.5B control .7675.

## Drafters (id: distill, lora, dpo)
- distill 4L d=256 (top-64+tail KL) → ckpt/tiny; val KL 3.79→2.63. lora PEFT r=16/α32 SFT → α .7306, tpc@8 3.49. dpo TRL β=0.1 (T=1 spec pairs, 1.5b) → α .7319, CIs disjoint.

## Kernels (id: triton)
- kernels/verify.py fused_verify + fused_alpha; tests 7/7 (TRITON_INTERPRET=1).

## Gateway (id: gateway)
- gateway/ Go: auth/429/routing/breaker/SSE tee//metrics//healthz//ui; go test 8/8; live smoke ✅.

## Observability / docker (id: obs, docker)
- prometheus rules+recordings (promtool OK; windows LEFT-CLOSED); grafana 8 panels; deploy/docker 3 images (hadolint OK).

## Ray sweep (id: sweep)
- sweep (Ray local, resume) → results/sweep.parquet; 0.5b-instruct, 393 val prompts, n_gen 16.
- **ALL 42 CELLS DONE (21 T=0 + 21 T=0.7); check_sweep_batch1/2/3 exit 0.**
- T=0: base acc 1.0, tpc 2/2.667/4/8 EXACT (Leviathan a=1); tpc@8 ngram 1.197/tiny 1.090/lora 2.089/dpo 2.116; sweep acc ≪ corpus α = coverage gap.
- T=0.7: ngram LOWER all k (.4411→.3265 k1); model drafters HIGHER (lora +.164, dpo +.188): T=0 rejects non-argmax, T>0 accepts w.p. min(1,P7/q)>0. dpo k8 out-of-sample PASS.
- ngram T=0.7 k1 acc .3265 vs hand E[accept]=P7[d] .3295 (tools/check_ngram_t07_ref.py), CI∋0.
- Speedups vs none (6.32/6.29 tok/s): ngram k8 1.1186/1.0875; base .84/.71; lora .25/.36 (CPU draft cost > gain); tables results/sweep_batch{2,3}_table.json.
- Gotchas: ~30-37 min/cell; bg bash dies at 30-min default → Bash timeout ≥7200000 + capture ≥2h; dpo ckpt = ckpt/dpo.

## Helm / kind (id: helm, kind)
- deploy/helm/speculative-agent-workloads: gateway HPA (ext metric), engine CPU, optional vLLM, ServMon/NP; check.sh = lint + template | kubeconform -strict. kind e2e.sh HOST-only: build→kind→helm→curl 200→accepted>0.

## Terraform (id: tf)
- deploy/terraform root→gke: VPC, GKE (WI, private, BA ENFORCE), g2+L4 pool; validate+tflint exit 0.

## GPU run (id: gpu)
- gpu_run.sh HOST-only (dry-run OK): 7B alpha + vllm arms + bench c=1/4/16 → results/gpu/; anchors ngram .2–.4, lora .5–.9, 1.4–1.75×.

## Results export (id: export)
- export_ui → web/public/data/*.json (schema v1); missing cells ⇒ pending (42/42 done, 0); --no-replays fast path.
- replays = 30 val blocks re-scored w/ alpha pair (1.5b, ctx 1024): per-token p,q, accept, α; test <1e-5.
- Gotcha: corpus (run,seq) NOT unique — match blocks by token ids. wall_s full precision (tok_s recomputable).

## Web explorer (id: web)
- web/ Vite+React19+TS+Recharts3: 4 tabs, base './', /ui mount; npm run build → dist/ incl. data/; vitest 22/22 vs public/data/*.json; ranges held.
- Gotchas: Area only in ComposedChart; RO polyfill synchronous; width-0 CIs → assert ≥0.

## Report / README (id: report)
- REPORT.md + README.md written (attempt 1 died on harness API-500).
- check_report.py exit 0 (recomputes headline numbers from results/, asserts in REPORT.md); fixed check temp-filter bug + ruff. README: mermaid, quickstart, skills map.

## Judge (id: judge)
- eval.judge (blind pairwise, qwen-agent, cap 150, cache) → results/judge.json; spec_vs_target n=81 .444, CIs∋0.5 (p=.434); T=0.7 exact 0/46 lossless.
