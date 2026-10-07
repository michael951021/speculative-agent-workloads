#!/usr/bin/env bash
# speculative-agent-workloads kind end-to-end demo — HOST ONLY (needs Docker; never run in the sandbox).
#
# One command: build images -> kind cluster -> load images -> helm install (cpu profile,
# tiny model) -> port-forward -> authed chat completion -> check /metrics for accepted
# tokens (spec decoding via the engine's default ngram drafter, k=5).
#
# Prereqs: docker, kind, helm, kubectl, python3 and a fetched model on disk:
#   python -m speculative_agent_workloads.models fetch --name qwen2.5-0.5b-instruct
#
# Knobs (env): SPECULATIVE_AGENT_WORKLOADS_E2E_CLUSTER (default speculative-agent-workloads-e2e), SPECULATIVE_AGENT_WORKLOADS_E2E_MODEL,
#   SPECULATIVE_AGENT_WORKLOADS_E2E_TAG (default e2e), SPECULATIVE_AGENT_WORKLOADS_E2E_PORT (default 18200),
#   SPECULATIVE_AGENT_WORKLOADS_E2E_TEARDOWN=1 (uninstall release + delete cluster at the end).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER="${SPECULATIVE_AGENT_WORKLOADS_E2E_CLUSTER:-speculative-agent-workloads-e2e}"
MODEL="${SPECULATIVE_AGENT_WORKLOADS_E2E_MODEL:-qwen2.5-0.5b-instruct}"
TAG="${SPECULATIVE_AGENT_WORKLOADS_E2E_TAG:-e2e}"
PORT="${SPECULATIVE_AGENT_WORKLOADS_E2E_PORT:-18200}"
NS=speculative-agent-workloads
RELEASE=speculative-agent-workloads
ALIAS=qwen0.5b
NODE_MODELS_DIR=/models

for tool in docker kind helm kubectl python3; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done

MODEL_DIR="$ROOT/models/$MODEL"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "missing $MODEL_DIR — run: python -m speculative_agent_workloads.models fetch --name $MODEL" >&2
  exit 1
fi

PF_PID=""
cleanup() {
  if [ -n "$PF_PID" ]; then
    kill "$PF_PID" 2>/dev/null || true
    wait "$PF_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "==> build images (gateway, engine)"
docker build -q -f "$ROOT/deploy/docker/Dockerfile.gateway" -t "speculative-agent-workloads/gateway:$TAG" "$ROOT"
docker build -q -f "$ROOT/deploy/docker/Dockerfile.engine" -t "speculative-agent-workloads/engine:$TAG" "$ROOT"

echo "==> kind cluster: $CLUSTER"
if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER" --wait 120s
fi

echo "==> load images into kind"
kind load docker-image "speculative-agent-workloads/gateway:$TAG" "speculative-agent-workloads/engine:$TAG" --name "$CLUSTER"

# Copy the model into the kind node; the engine mounts it via a hostPath volume.
NODE="$(kind get nodes --name "$CLUSTER" | head -n 1)"
docker exec "$NODE" mkdir -p "$NODE_MODELS_DIR"
docker cp "$MODEL_DIR" "$NODE:$NODE_MODELS_DIR/$MODEL"

echo "==> helm install (cpu profile: 1 replica, HPA/ServiceMonitor/NP off)"
KEY="sk-e2e-$(head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n')"
SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$KEY")"
helm upgrade --install "$RELEASE" "$ROOT/deploy/helm/speculative-agent-workloads" \
  --namespace "$NS" --create-namespace \
  --set "image.gateway.tag=$TAG" --set "image.engine.tag=$TAG" \
  --set gateway.replicas=1 \
  --set gateway.hpa.enabled=false \
  --set "gateway.apiKeys[0].name=e2e" \
  --set "gateway.apiKeys[0].key=$KEY" \
  --set "gateway.apiKeys[0].sha256=$SHA" \
  --set "gateway.apiKeys[0].rate=60" --set "gateway.apiKeys[0].burst=120" \
  --set "engine.model=$MODEL" --set "engine.alias=$ALIAS" \
  --set engine.modelVolume.type=hostPath \
  --set "engine.modelVolume.hostPath=$NODE_MODELS_DIR" \
  --set observability.serviceMonitor.enabled=false \
  --set observability.grafanaDashboard.enabled=false \
  --set networkPolicy.enabled=false
helm --namespace "$NS" wait "$RELEASE" --for=deployment --timeout 15m

echo "==> port-forward svc/$RELEASE-gateway to 127.0.0.1:$PORT"
kubectl --namespace "$NS" port-forward "svc/$RELEASE-gateway" "$PORT:8200" >/dev/null 2>&1 &
PF_PID=$!
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null; then
    break
  fi
  sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null; then
  echo "gateway never became healthy on 127.0.0.1:$PORT" >&2
  kubectl --namespace "$NS" logs -l app.kubernetes.io/component=gateway --tail=50 >&2 || true
  exit 1
fi

echo "==> chat completion (model $ALIAS, default drafter ngram k=5)"
RESP="$(curl -sS -w $'\n%{http_code}' \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"'"$ALIAS"'","max_tokens":128,"stream":false,"messages":[{"role":"user","content":"Repeat the exact phrase \"the quick brown fox\" ten times, one phrase per line."}]}' \
  "http://127.0.0.1:$PORT/v1/chat/completions")"
CODE="${RESP##*$'\n'}"
BODY="${RESP%$'\n'*}"
if [ "$CODE" != "200" ]; then
  echo "completion failed (HTTP $CODE): $BODY" >&2
  exit 1
fi
echo "$BODY" | python3 -m json.tool | head -n 30

echo "==> check /metrics for accepted tokens"
METRICS="$(curl -sS "http://127.0.0.1:$PORT/metrics")"
ACCEPTED="$(printf '%s\n' "$METRICS" | awk '$1 ~ /^speculative_agent_workloads_accepted_tokens_total\{/ {s += $2} END {print s + 0}')"
echo "speculative_agent_workloads_accepted_tokens_total = $ACCEPTED"
if awk -v a="$ACCEPTED" 'BEGIN { exit (a > 0) ? 0 : 1 }'; then
  echo "==> e2e OK: served completion + spec-decoding accepted tokens visible in /metrics"
else
  echo "no accepted tokens in /metrics — spec decoding not observable" >&2
  exit 1
fi

echo "API key: $KEY (release kept for reuse: kubectl -n $NS port-forward svc/$RELEASE-gateway $PORT:8200)"
if [ "${SPECULATIVE_AGENT_WORKLOADS_E2E_TEARDOWN:-0}" = "1" ]; then
  cleanup
  trap - EXIT
  helm uninstall "$RELEASE" --namespace "$NS" --wait --timeout 5m
  kind delete cluster --name "$CLUSTER"
  echo "==> teardown done"
fi
