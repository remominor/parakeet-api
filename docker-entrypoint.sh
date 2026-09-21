#!/usr/bin/env bash
set -Eeuo pipefail
model_file="${PARAKEET_ASR_MODEL_FILE:-${PARAKEET_MODEL_FILE:-parakeet-unified-en-0.6b-Q8_0.gguf}}"
if [[ "$model_file" = /* ]]; then
  model_path="$model_file"
elif [[ -r "/models/asr/$model_file" ]]; then
  model_path="/models/asr/$model_file"
else
  model_path="/models/$model_file"
fi
[[ -r "$model_path" ]] || { echo "ASR model is missing or unreadable: $model_path" >&2; exit 66; }
exec uvicorn gateway.app:app --host 0.0.0.0 --port 8080 --no-access-log --proxy-headers --timeout-keep-alive 75
