#!/usr/bin/env bash
set -Eeuo pipefail
model_file="${PARAKEET_MODEL_FILE:-tdt-0.6b-v2-f16.gguf}"
case "$model_file" in tdt-0.6b-v2-f16.gguf|tdt-0.6b-v2-q8_0.gguf) ;; *) echo "PARAKEET_MODEL_FILE must name the F16 or Q8 v2 GGUF" >&2; exit 64;; esac
model_path="/models/$model_file"
[[ -r "$model_path" ]] || { echo "Model is missing or unreadable: $model_path" >&2; exit 66; }
exec uvicorn gateway.app:app --host 0.0.0.0 --port 8080 --no-access-log --proxy-headers --timeout-keep-alive 75
