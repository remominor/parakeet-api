#!/usr/bin/env bash
set -euo pipefail
base_url="${1:-http://127.0.0.1:5092}"; sample="${2:-sample.wav}"
auth=(); [[ -n "${PARAKEET_STT_API_KEY:-}" ]] && auth=(-H "Authorization: Bearer $PARAKEET_STT_API_KEY")
curl -fsS "$base_url/readyz" | grep -q '"ready":true'
curl -fsS "${auth[@]}" "$base_url/v1/models" | grep -q 'parakeet-tdt-0.6b-v2'
curl -fsS "${auth[@]}" "$base_url/info" | grep -q '"word_confidence":true'
# json is the API default; the remaining calls cover every explicit response format.
curl -fsS "${auth[@]}" -F "file=@$sample" -F "model=parakeet" "$base_url/v1/audio/transcriptions" | grep -q '"text"'
curl -fsS "${auth[@]}" -F "file=@$sample" -F 'response_format=text' "$base_url/v1/audio/transcriptions" | grep -q .
curl -fsS "${auth[@]}" -F "file=@$sample" -F 'response_format=verbose_json' "$base_url/v1/audio/transcriptions" | grep -q '"words"'
curl -fsS "${auth[@]}" -F "file=@$sample" -F 'response_format=srt' "$base_url/v1/audio/transcriptions" | grep -q -- '-->'
curl -fsS "${auth[@]}" -F "file=@$sample" -F 'response_format=vtt' "$base_url/v1/audio/transcriptions" | grep -q '^WEBVTT'
curl -sS -o /dev/null -D - "${auth[@]}" -H 'X-Request-ID: smoke-request-1' -F "file=@$sample" -F 'language=en-US' "$base_url/v1/audio/transcriptions" | grep -qi '^x-request-id: smoke-request-1'
curl -sS -o /dev/null -w '%{http_code}' "${auth[@]}" -F "file=@$sample" -F 'language=fr' "$base_url/v1/audio/transcriptions" | grep -qx '422'
curl -sS -o /dev/null -w '%{http_code}' "${auth[@]}" -F "file=@$sample" -F 'prompt=ignored' "$base_url/v1/audio/transcriptions" | grep -qx '422'
curl -sS -o /dev/null -w '%{http_code}' "${auth[@]}" -F "file=@$sample" -F 'temperature=0.1' "$base_url/v1/audio/transcriptions" | grep -qx '422'
echo "smoke test passed"
