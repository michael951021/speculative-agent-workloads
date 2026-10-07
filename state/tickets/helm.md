# Ticket: Helm chart (id: helm)

## Issue
The stack (gateway, JAX engine, optional vLLM w/ drafter) has Dockerfiles but no
Kubernetes deployment. A user cannot run speculative-agent-workloads on a real cluster: no
Deployment/Service/HPA, no secret for API keys, no GPU scheduling, no
monitoring hooks. Needed to answer "how would this run in production".

## Question
Can `helm lint` pass and `helm template` (cpu AND gpu profiles) render manifests
that validate with `kubeconform -strict`, including the ServiceMonitor CRD
schema? Refutation: any non-zero exit from lint or kubeconform.

## Evidence
- Observed: `tools/check.sh:48-55` already runs `helm lint deploy/helm` +
  kubeconform, but kubeconform checks raw chart yaml (helm syntax, no
  templating) and has no ServiceMonitor schema.
- Observed: gateway config = single YAML file (gateway/config.go: LoadConfig),
  keys are sha256 hex; metrics `speculative_agent_workloads_accepted_tokens_total{model}`
  (gateway/metrics.go); vLLM entrypoint adds `--speculative-config` when
  `$DRAFTER_PATH/adapter_config.json` exists; engine serves :8100
  (only /v1/models + /v1/chat/completions).
- Check (run first, expect FAIL): `helm lint deploy/helm` → "chart not found". PASSED pre-run.
- Output saved to .agent-evidence/helm-*.log. Post-run: lint 0 fail; kubeconform 12/12 cpu, 15/15 gpu, exit 0.
- Systemic gotcha found: kubeconform v0.6.7 -schema-location template vars are
  lowercased ResourceKind + KindSuffix (servicemonitor-monitoring-v1.json);
  `include|toYaml` on a string corrupts it; all-digit sha256 needs quoting.

## Done when
1. `helm lint deploy/helm` exits 0.
2. `helm template speculative-agent-workloads deploy/helm/speculative-agent-workloads | kubeconform -strict -summary` exits 0
   (cpu profile) and same with `-f values-gpu.yaml` (gpu profile), with the
   ServiceMonitor CRD schema vendored and used (no network at check time).
3. Both templates rendered into the manifests required: gateway
   Deployment/Service/HPA (external metric via prometheus-adapter)/Secret,
   engine Deployment/Service, vLLM Deployment (gpu:1, tolerations,
   --speculative-config from PVC, readiness /health)/Service (gpu profile),
   ServiceMonitor, Grafana dashboard ConfigMap (in sync with
   deploy/grafana/speculative-agent-workloads.json), NetworkPolicy, PDB.
Known-answer check: rendered gateway Deployment must reference the engine
Service DNS name and the Secret-mounted config; rendered HPA metric name must
match a prometheus-adapter rule documented in README; grafana ConfigMap bytes
== deploy/grafana/speculative-agent-workloads.json (diff, empty).
Plausible range: not a numeric ticket; acceptance is binary pass/fail.

## Stop when
Scope: chart only — no Terraform (id: tf), no kind run (id: kind), no actual
helm install. Stop/split: two failed attempts to make kubeconform accept
ServiceMonitor offline → record blocker (schema sourcing) and stop.

## Plan
1. Run failing checks (helm lint, template|kubeconform) — capture logs.
2. Vendor ServiceMonitor openAPIV3Schema from prometheus-operator manifests.
3. Write chart (values.yaml cpu base, values-gpu.yaml, templates per list above).
4. Extend tools/check.sh helm section: template both profiles | kubeconform
   with vendored schema; add grafana-sync diff check.
5. Run make check; commit.
