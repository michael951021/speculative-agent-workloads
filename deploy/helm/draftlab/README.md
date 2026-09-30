# draftlab Helm chart

Deploys the draftlab stack into one namespace:

| Component | Kind | Notes |
|---|---|---|
| `*-gateway` | Deployment + Service + HPA | OpenAI-compatible proxy, :8200, serves /ui |
| `*-engine` | Deployment + Service | JAX speculative-decoding engine (CPU), :8100 |
| `*-vllm` | Deployment + Service | optional (gpu profile): vLLM target, :8000 |
| `*-gateway-config` | Secret | gateway config.yaml (API keys as sha256) + raw keys |
| ServiceMonitor(s) | monitoring.coreos.com/v1 | gateway (+ vllm) /metrics, needs prometheus-operator |
| `*-grafana-dashboard` | ConfigMap | `grafana_dashboard: "1"`, copy of deploy/grafana/draftlab.json |
| NetworkPolicies | default-deny + allow ingress/DNS/intra-ns | |
| PDB | gateway | minAvailable 1 |

## Profiles

- **cpu** (default `values.yaml`): gateway + engine only, vLLM disabled.
- **gpu** (`values-gpu.yaml`): adds vLLM with `nvidia.com/gpu` requests,
  toleration for the `nvidia.com/gpu` taint, and nodeSelector `nvidia.com/gpu=true`.
  The trained drafter must live on the PVC named in `vllm.drafterPvc`
  (mounted at `/drafter`); the image entrypoint then starts vLLM with
  `--speculative-config` (num_speculative_tokens = `vllm.drafterK`).
  Engine weights come from the PVC/hostPath in `engine.modelVolume`.

```sh
# cpu
helm install draftlab deploy/helm/draftlab \
  --namespace draftlab --create-namespace \
  --set gateway.apiKeys[0].key=... --set gateway.apiKeys[0].sha256=$(python3 -c \
    'import hashlib,sys;print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' '...')
# gpu
helm install draftlab deploy/helm/draftlab \
  -f deploy/helm/draftlab/values-gpu.yaml --namespace draftlab --create-namespace
```

## HPA external metric (prometheus-adapter)

The gateway HPA targets the external metric `draftlab_accepted_tokens_per_second`
(AverageValue). Add this rule to your prometheus-adapter ConfigMap:

```yaml
- seriesQuery: 'draftlab_accepted_tokens_total{model!=""}'
  resources:
    ownNamespace: true
  name:
    as: "draftlab_accepted_tokens_per_second"
  metricsQuery: 'sum(rate(draftlab_accepted_tokens_total[2m]))'
```

Without the adapter rule the HPA reports `ExternalMetric` unsatisfied and holds
minReplicas.

## Validating (sandbox, no cluster)

```sh
helm lint deploy/helm
helm template draftlab deploy/helm/draftlab | \
  kubeconform -strict -summary \
  -schema-location default \
  -schema-location deploy/helm/draftlab/crd-schemas/{{ .ResourceKind }}{{ .KindSuffix }}.json
```

The ServiceMonitor schema is vendored under `crd-schemas/` (from
prometheus-operator v0.76.2), so the check works offline.
