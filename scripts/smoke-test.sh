#!/usr/bin/env bash
set -euo pipefail
base_url="${1:-http://127.0.0.1:5092}"; sample="${2:-sample.wav}"
auth=(); [[ -n "${PARAKEET_STT_API_KEY:-}" ]] && auth=(-H "Authorization: Bearer $PARAKEET_STT_API_KEY")
curl -fsS "$base_url/readyz" | grep -q '"ready":true'
curl -fsS "${auth[@]}" "$base_url/v1/models" | grep -q 'parakeet-tdt-0.6b-v2'
curl -fsS "${auth[@]}" -F "file=@$sample" -F "model=parakeet" "$base_url/v1/audio/transcriptions" | grep -q '"text"'
echo "smoke test passed"
