#!/bin/sh
# draftlab/vllm entrypoint: `vllm serve` with an optional speculative drafter.
# VLLM_MODEL is required. If a drafter adapter is mounted at $DRAFTER_PATH
# (PVC; must contain adapter_config.json), enable speculative decoding with it.
set -e

VLLM_MODEL=${VLLM_MODEL:?VLLM_MODEL must be set, e.g. Qwen/Qwen2.5-7B-Instruct}

if [ -n "${DRAFTER_PATH:-}" ] && [ -e "${DRAFTER_PATH}/adapter_config.json" ]; then
  echo "draftlab: speculative drafter from ${DRAFTER_PATH}"
  SPEC_CONFIG="{\"model\": \"${DRAFTER_PATH}\", \"num_speculative_tokens\": ${DRAFTER_K:-4}}"
  exec vllm serve "$VLLM_MODEL" --speculative-config "$SPEC_CONFIG" "$@"
fi
exec vllm serve "$VLLM_MODEL" "$@"
