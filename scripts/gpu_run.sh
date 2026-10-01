#!/usr/bin/env bash
# GPU run: alpha eval at T=Qwen2.5-7B-Instruct (Triton kernel) + vLLM serving with
# ngram / LoRA speculative decoding + gateway load bench at concurrency 1/4/16.
#
# Host-only (needs GPUs, after `./agent stop` frees them). In the sandbox:
#   bash scripts/gpu_run.sh --dry-run   # prints the plan, checks files, exits 0
#
# Outputs: results/gpu/alpha_{ngram,lora}_7b.json, bench_<arm>_c{1,4,16}.json, summary.json
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=${PY:-.venv/bin/python}

# ---- knobs (env-overridable) ------------------------------------------------
TARGET_DIR=${TARGET_DIR:-$ROOT/models/qwen2.5-7b-instruct}   # alpha target
BASE_DIR=${BASE_DIR:-$ROOT/models/qwen2.5-0.5b}              # LoRA adapter base
LORA_DIR=${LORA_DIR:-$ROOT/ckpt/lora}
MODEL=${MODEL:-}                                             # vllm model: local dir or HF id
VLLM_PORT=${VLLM_PORT:-8000}
GW_PORT=${GW_PORT:-8200}
K=${K:-4}                                                    # num_speculative_tokens
GPU=${GPU:-0}
REQUESTS=${REQUESTS:-24}
MAX_TOKENS=${MAX_TOKENS:-128}
CONCURRENCIES="1 4 16"
GW_KEY=${GW_KEY:-sk-draftlab-dev}
RES=${RES:-$ROOT/results/gpu}
VLLM_LOG=$RES/logs/vllm.log
GW_LOG=$RES/logs/gateway.log
VLLM_TIMEOUT=${VLLM_TIMEOUT:-900}

DRY=0
STOP_AGENTS=0
for a in "$@"; do
  case $a in
    --dry-run) DRY=1 ;;
    --stop-agents) STOP_AGENTS=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $a (try --help)" >&2; exit 2 ;;
  esac
done

[ -n "$MODEL" ] || MODEL="$TARGET_DIR"

# ---- file checks -------------------------------------------------------------
MISS=0
need() { # need <label> <path>  (must exist, even in sandbox)
  if [ -e "$2" ]; then echo "OK      $1: $2"
  else echo "MISSING $1: $2" >&2; MISS=1; fi
}
hostonly() { # hostonly <label> <path>  (required on host; missing in sandbox is expected)
  if [ -e "$2" ]; then echo "OK      $1: $2"
  else
    if [ $DRY -eq 1 ]; then echo "HOST-ONLY (absent, expected in sandbox) $1: $2"
    else echo "MISSING $1: $2" >&2; MISS=1; fi
  fi
}

plan() {
  cat <<EOF
Plan (host GPU run; sandbox: --dry-run only):
  0. preflight: $([ $STOP_AGENTS -eq 1 ] && echo './agent stop; ' || echo 'expect GPUs free (run ./agent stop first); ')nvidia-smi free, vllm+go importable
  1. alpha eval, target $TARGET_DIR (fused_alpha Triton kernel on CUDA):
       $PY -m draftlab.eval.alpha --drafter ngram  --target $TARGET_DIR --blocks 400 --max-tokens 256 --out $RES/alpha_ngram_7b.json
       $PY -m draftlab.eval.alpha --drafter hf     --draft-model $LORA_DIR  --target $TARGET_DIR --blocks 400 --max-tokens 256 --out $RES/alpha_lora_7b.json
     expected ranges (written before the run, Leviathan 2023 + llama.cpp #10466 0.5B->7B k=4):
       ngram alpha ~0.2-0.4 (bigram anchor ~0.2; corpus has agent repetition)
       lora  alpha ~0.5-0.9 (small-drafter anchor)
  2. serving + bench through the gateway (model: $MODEL, vllm :$VLLM_PORT, gateway :$GW_PORT, key $GW_KEY):
     arms: baseline (no spec), ngram ({"method":"ngram","num_speculative_tokens":$K,"prompt_lookup_max":$K}),
           lora ({"model":"$LORA_DIR","num_speculative_tokens":$K})
     per arm: vllm serve -> gateway -> for c in $CONCURRENCIES:
       $PY tools/bench_gateway.py --url http://127.0.0.1:$GW_PORT --model $MODEL --concurrency \$c \
         --requests $REQUESTS --max-tokens $MAX_TOKENS --out $RES/bench_<arm>_c<c>.json
     speedup anchor (llama.cpp, k=4): ~1.4x general text, ~1.75x code over baseline
  3. summary: $RES/summary.json + sanity checks
       baseline arm: draft_tokens_proposed delta == 0 (null control: no spec config -> no drafts)
       spec arms: draft_tokens_accepted > 0, accept_rate in 0.05-0.95
EOF
}

plan
echo "--- file checks ---"
need  python-venv       "$PY"
need  corpus-val        data/corpus/val.parquet
need  alpha-eval        draftlab/eval/alpha.py
need  bench-tool        tools/bench_gateway.py
need  lora-adapter      "$LORA_DIR/adapter_config.json"
need  lora-weights      "$LORA_DIR/adapter_model.safetensors"
need  gateway-src       gateway/main.go
hostonly target-7b      "$TARGET_DIR/config.json"
hostonly lora-base-05b  "$BASE_DIR/config.json"
hostonly vllm-bin       "$(command -v vllm 2>/dev/null || echo /usr/local/bin/vllm)"
hostonly go-bin         "$(command -v go 2>/dev/null || echo $ROOT/.tools/go/bin/go)"
hostonly nvidia-smi     "$(command -v nvidia-smi 2>/dev/null || echo /usr/bin/nvidia-smi)"

if [ $DRY -eq 1 ]; then
  [ $MISS -eq 1 ] && { echo "DRY-RUN FAIL: sandbox files missing"; exit 1; }
  echo "DRY-RUN OK: plan printed, sandbox files present, host-only files listed, no GPU work performed."
  exit 0
fi
[ $MISS -eq 1 ] && { echo "FATAL: required files missing (see above)"; exit 1; }
export PATH="$ROOT/.venv/bin:$PATH"

# ---- 0. preflight -------------------------------------------------------------
[ $STOP_AGENTS -eq 1 ] && ./agent stop
mkdir -p "$RES/logs"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee "$RES/logs/gpus.txt"
if ! "$PY" -c "import vllm" 2>/dev/null; then
  echo "preflight: vllm not importable in $PY — installing on host is out of scope; aborting." >&2
  exit 1
fi
GW_BIN=$ROOT/.tools/draftlab-gateway
command -v go >/dev/null 2>&1 || export PATH="$ROOT/.tools/go/bin:$PATH"
if [ ! -x "$GW_BIN" ] || [ gateway/main.go -nt "$GW_BIN" ]; then
  (cd gateway && go build -o "$GW_BIN" .) || { echo "FATAL: gateway build failed" >&2; exit 1; }
fi
cleanup() { kill ${GW_PID:-} ${VLLM_PID:-} 2>/dev/null; }
trap cleanup EXIT

wait_http() { # wait_http <url> <timeout_s>
  local i=0
  while [ $i -lt "$2" ]; do
    curl -sf "$1" >/dev/null 2>&1 && return 0
    sleep 5; i=$((i + 5))
  done
  return 1
}

# ---- 1. alpha eval (Triton fused_alpha on CUDA) --------------------------------
CUDA_VISIBLE_DEVICES=$GPU "$PY" -m draftlab.eval.alpha --drafter ngram \
  --target "$TARGET_DIR" --blocks 400 --max-tokens 256 \
  --out "$RES/alpha_ngram_7b.json" | tee "$RES/logs/alpha_ngram.log"
CUDA_VISIBLE_DEVICES=$GPU "$PY" -m draftlab.eval.alpha --drafter hf \
  --draft-model "$LORA_DIR" --target "$TARGET_DIR" --blocks 400 --max-tokens 256 \
  --out "$RES/alpha_lora_7b.json" | tee "$RES/logs/alpha_lora.log"

# ---- 2. serving + bench ---------------------------------------------------------
SPEC_BASELINE=""
SPEC_NGRAM="{\"method\": \"ngram\", \"num_speculative_tokens\": $K, \"prompt_lookup_max\": $K}"
SPEC_LORA="{\"model\": \"$LORA_DIR\", \"num_speculative_tokens\": $K}"

run_arm() { # run_arm <name> <spec-config>
  local name=$1 spec=$2
  echo "== arm: $name (spec=$spec) =="
  if [ -n "$spec" ]; then
    vllm serve "$MODEL" --port "$VLLM_PORT" --speculative-config "$spec" \
      >"$VLLM_LOG" 2>&1 &
  else
    vllm serve "$MODEL" --port "$VLLM_PORT" >"$VLLM_LOG" 2>&1 &
  fi
  VLLM_PID=$!
  wait_http "http://127.0.0.1:$VLLM_PORT/health" "$VLLM_TIMEOUT" \
    || { echo "FATAL: vllm ($name) not healthy after ${VLLM_TIMEOUT}s; tail:" >&2; tail -20 "$VLLM_LOG" >&2; return 1; }

  local keysha cfg
  keysha=$(printf '%s' "$GW_KEY" | sha256sum | cut -d' ' -f1)
  cfg=$RES/gateway-$name.yaml
  cat > "$cfg" <<YAML
listen: ":$GW_PORT"
retries: 1
health_check_interval: 5s
backends:
  - name: vllm
    url: "http://127.0.0.1:$VLLM_PORT"
    models:
      - "$MODEL"
keys:
  - name: bench
    sha256: "$keysha"
    rate: 1000
    burst: 1000
YAML
  "$GW_BIN" -config "$cfg" >"$GW_LOG" 2>&1 &
  GW_PID=$!
  wait_http "http://127.0.0.1:$GW_PORT/healthz" 60 \
    || { echo "FATAL: gateway ($name) not healthy; tail:" >&2; tail -20 "$GW_LOG" >&2; return 1; }

  local c
  for c in $CONCURRENCIES; do
    "$PY" tools/bench_gateway.py --url "http://127.0.0.1:$GW_PORT" --key "$GW_KEY" \
      --model "$MODEL" --concurrency "$c" --requests "$REQUESTS" --max-tokens "$MAX_TOKENS" \
      --out "$RES/bench_${name}_c${c}.json" || echo "WARN: bench $name c=$c failed" >&2
  done
  kill $GW_PID $VLLM_PID 2>/dev/null
  GW_PID=; VLLM_PID=
  wait 2>/dev/null
  sleep 5 # let ports free
}

run_arm baseline "$SPEC_BASELINE" || exit 1
run_arm ngram    "$SPEC_NGRAM"    || exit 1
run_arm lora     "$SPEC_LORA"     || exit 1

# ---- 3. summary + sanity checks --------------------------------------------------
"$PY" - "$RES" "$K" <<'EOF' | tee "$RES/logs/summary.log"
import json, sys
from pathlib import Path

res, k = Path(sys.argv[1]), int(sys.argv[2])
out = {"target": "Qwen2.5-7b-instruct (host)", "k": k, "alpha": {}, "bench": {}, "checks": []}

def load(p):
    return json.loads(Path(p).read_text()) if p.exists() else None

for name in ("ngram", "lora"):
    a = load(res / f"alpha_{name}_7b.json")
    if a and "overall" in a:
        o = a["overall"]
        out["alpha"][name] = {
            "alpha": o.get("alpha"),
            "alpha_ci95": o.get("alpha_ci95"),
            "greedy": o.get("greedy"),
            "n_blocks": a.get("n_blocks"),
        }

speedup = {}
for c in ("1", "4", "16"):
    base = load(res / f"bench_baseline_c{c}.json")
    if not base or not base.get("throughput_tok_s"):
        continue
    row = {"baseline_tok_s": base["throughput_tok_s"]}
    for arm in ("ngram", "lora"):
        b = load(res / f"bench_{arm}_c{c}.json")
        if b and b.get("throughput_tok_s"):
            row[f"{arm}_tok_s"] = b["throughput_tok_s"]
            row[f"{arm}_speedup_x"] = b["throughput_tok_s"] / base["throughput_tok_s"]
            d = b.get("gateway_metrics_delta", {})
            row[f"{arm}_accept_rate"] = d.get("accept_rate")
    speedup[f"c{c}"] = row
out["bench"] = speedup

# sanity checks (known answers / null controls)
b1 = load(res / "bench_baseline_c1.json")
if b1:
    prop = b1.get("gateway_metrics_delta", {}).get("draft_tokens_proposed", 0)
    out["checks"].append({
        "name": "baseline-null-control: no spec config -> 0 draft tokens",
        "pass": prop == 0, "value": prop})
for arm in ("ngram", "lora"):
    b = load(res / f"bench_{arm}_c1.json")
    if b:
        acc = b.get("gateway_metrics_delta", {}).get("draft_tokens_accepted", 0)
        rate = b.get("gateway_metrics_delta", {}).get("accept_rate")
        ok = acc > 0 and rate is not None and 0.05 <= rate <= 0.95
        out["checks"].append({"name": f"{arm}: accepted>0 and accept_rate in [0.05, 0.95]",
                              "pass": ok, "value": {"accepted": acc, "rate": rate}})

(res / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
sys.exit(0 if all(c["pass"] for c in out["checks"]) and out["checks"] else 3)
EOF
RC=$?
echo "GPU run finished; results in $RES"
exit $RC
