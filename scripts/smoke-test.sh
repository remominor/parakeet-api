#!/usr/bin/env bash
set -euo pipefail
base_url="${1:-http://127.0.0.1:5092}"; sample="${2:-sample.wav}"
auth=(); [[ -n "${PARAKEET_STT_API_KEY:-}" ]] && auth=(-H "Authorization: Bearer $PARAKEET_STT_API_KEY")
curl -fsS "$base_url/readyz" | grep -q '"ready":true'
curl -fsS "${auth[@]}" "$base_url/v1/models" | grep -q 'parakeet-tdt-0.6b-v2'
# json is the API default; the remaining calls cover every explicit response format.
curl -fsS "${auth[@]}" -F "file=@$sample" -F "model=parakeet" "$base_url/v1/audio/transcriptions" | grep -q '"text"'
curl -fsS "${auth[@]}" -F "file=@$sample" -F 'response_format=text' "$base_url/v1/audio/transcriptions" | grep -q .
curl -fsS "${auth[@]}" -F "file=@$sample" -F 'response_format=verbose_json' "$base_url/v1/audio/transcriptions" | grep -q '"text"'
curl -fsS "${auth[@]}" -F "file=@$sample" -F 'response_format=srt' "$base_url/v1/audio/transcriptions" | grep -q -- '-->'
curl -fsS "${auth[@]}" -F "file=@$sample" -F 'response_format=vtt' "$base_url/v1/audio/transcriptions" | grep -q '^WEBVTT'
echo "smoke test passed"
