# Parakeet TDT OpenAI API

One CUDA Docker container serving NVIDIA Parakeet TDT 0.6B v2 as an
OpenAI-compatible local-network transcription API. The image runs the native
`parakeet.cpp` GGUF engine and a small API gateway; only port 8080 is exposed.

It is optimized for completed, short voice-agent turns. It does not produce
live partial transcripts while the user is speaking.

## Requirements

- Docker Compose v2
- NVIDIA GPU, driver, and NVIDIA Container Toolkit
- One GGUF under `./models/`

| Precision | File | Use |
|---|---|---|
| F16 | `tdt-0.6b-v2-f16.gguf` | Default, highest fidelity |
| Q8 | `tdt-0.6b-v2-q8_0.gguf` | Lower memory use |

The F16 model already present in this checkout is used by default. Download Q8
from the `mudler/parakeet-cpp-gguf` collection and validate it against
`models/SHA256SUMS.txt` before use.

## Local benchmark

Measured on this setup after model and CUDA-graph warm-up:

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM)
- Input: bundled 12-second 16 kHz WAV (`sample.wav`)
- API path: local HTTP, WAV passthrough, `verbose_json` with word timestamps

| Model | Warm engine / total time | Result |
|---|---:|---|
| F16 | 80.4 ms | Reference transcript |
| Q8 | 51.6 ms | Same transcript as F16 |

Q8 was about **1.56× faster** in this single-request check while also using
less VRAM, so it is the recommended default for this host. These timings are
not a general throughput claim: repeat them with representative utterance
lengths and concurrent voice sessions before using them as a capacity estimate.

## Run

```bash
cp .env.example .env
# Set PARAKEET_API_KEYS in .env
docker compose up --build
```

The service listens on `http://127.0.0.1:5092` by default. Choose Q8 by setting
`PARAKEET_MODEL_FILE=tdt-0.6b-v2-q8_0.gguf` in `.env`, then restart the service.

```bash
curl http://127.0.0.1:5092/v1/audio/transcriptions \
  -H 'Authorization: Bearer YOUR_KEY' \
  -F file=@sample.wav -F model=parakeet
```

## API

- `POST /v1/audio/transcriptions`: WAV, MP3, OGG, WebM, FLAC, and M4A uploads;
  `json`, `text`, `verbose_json`, `srt`, and `vtt` response formats.
- `GET /v1/models`: returns `parakeet-tdt-0.6b-v2`.
- `GET /health` and `GET /readyz` return model readiness plus engine-process GPU
  memory. They return `200` only when the model is loaded, otherwise `503` while
  the gateway stays available for lifecycle control. Authenticated `GET /stats`
  and `GET /info` support operation and capability discovery.
- Authenticated `POST /internal/model/load` and `/internal/model/unload` (also
  available as `/v1/model/load` and `/v1/model/unload`) asynchronously load or
  unload the CUDA engine. They return `202` while a transition is in progress;
  poll `/health` until its `model_state` is `loaded`.

When loaded, health includes the host GPU index and the native engine process's
reported VRAM use (the native backend has no separate allocator-reserved value):

```json
{"status":"ready","model":"parakeet-tdt-0.6b-v2","model_state":"loaded","device":"cuda:1","vram_allocated_mb":1842,"vram_reserved_mb":1842}
```
- Uploads are capped while streaming. An `audio_url` form field supports
  `http`/`https` inputs, with redirects and private/local addresses rejected
  by default; use `PARAKEET_URL_ALLOWED_HOSTS` for trusted internal hosts.
- `verbose_json`, SRT, and VTT use timestamp-aware segments derived from native
  word metadata. WebSocket final events include aggregate confidence; append
  `?verbose=true` to include the native word list.

### Voice-turn WebSocket

`ws://HOST/v1/audio/transcriptions/ws` accepts raw, little-endian PCM16 mono
audio at 16 kHz. Send binary frames only (up to 65,536 bytes each). The server
uses CPU Silero VAD to identify voice turns and sends JSON lifecycle events:

```json
{"type":"ready","sample_rate":16000,"model":"parakeet-tdt-0.6b-v2"}
{"type":"speech_started","turn_id":"1","audio_start_ms":120.0}
{"type":"speech_stopped","turn_id":"1","audio_end_ms":1234.0}
{"type":"final","turn_id":"1","text":"...","duration_ms":1234.0,"engine_ms":25.1,"total_ms":25.3,"confidence":{"mean":0.95,"min":0.82,"low_word_count":0}}
```

The endpoint emits final transcription only after 350 ms of detected silence
or an 8-second maximum turn. Continue sending silence frames after a caller
stops speaking so VAD can close the final turn. It is VAD-endpointed streaming,
not token-by-token partial ASR.

When API keys are enabled, authenticate with `Authorization: Bearer TOKEN` or
`?api_key=TOKEN` (use the latter only for browser clients that cannot attach
headers during the WebSocket handshake).

The native socket also accepts JSON control events without closing the socket:
`{"type":"commit"}` finalizes the active turn immediately,
`{"type":"clear"}` discards it, and `{"type":"config","vad":{...}}`
overrides VAD settings for that connection. Completed turns are queued (depth
2) while the single GPU inference worker remains serialized; excess completed
turns receive an explicit `turn_queue_full` error rather than growing memory.

### OpenAI-style realtime transcription

`ws://HOST/v1/realtime?model=parakeet&intent=transcription` is a thin
transcription-only protocol adapter. It accepts `session.update`,
`input_audio_buffer.append` (base64 PCM16 at 24 kHz), `commit`, and `clear`.
It emits VAD lifecycle, committed-item, and completed-transcription events.
Parakeet produces completed turns only: it does not emit synthetic partial
transcript deltas.

Tune endpointing through `PARAKEET_WS_VAD_THRESHOLD`,
`PARAKEET_WS_MIN_SILENCE_MS`, `PARAKEET_WS_SPEECH_PAD_MS`, and
`PARAKEET_WS_MAX_UTTERANCE_MS`. Silero VAD is bundled as a pinned ONNX asset in
the image; it runs on CPU, keeping GPU memory available for ASR.

`parakeet`, `parakeet-en`, and `whisper-1` are accepted aliases. API keys are
optional only when `PARAKEET_API_KEYS` is blank; keep authentication enabled for
any network reachable beyond a trusted host.

For the WAV voice-agent hot path, inspect `X-Parakeet-Engine-Ms` and
`X-Parakeet-Total-Ms` response headers. Non-WAV uploads are decoded before
inference and include `X-Parakeet-Transcoded: 1`.

Set `PARAKEET_METRICS_ENABLED=true` to expose a dependency-free Prometheus
text endpoint at `/metrics`. Set `PARAKEET_WEBUI_ENABLED=true` to expose the
small local drag-and-drop UI at `/`; it has no server-side state.

Run the basic check after readiness:

```bash
PARAKEET_STT_API_KEY=YOUR_KEY ./scripts/smoke-test.sh
```
