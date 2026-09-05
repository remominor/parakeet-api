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
- `GET /health`, `GET /readyz`, and authenticated `GET /stats` support operation
  and latency inspection.

### Voice-turn WebSocket

`ws://HOST/v1/audio/transcriptions/ws` accepts raw, little-endian PCM16 mono
audio at 16 kHz. Send binary frames only (up to 65,536 bytes each). The server
uses CPU Silero VAD to identify voice turns and sends JSON lifecycle events:

```json
{"type":"ready","sample_rate":16000,"model":"parakeet-tdt-0.6b-v2"}
{"type":"speech_started","turn_id":"1"}
{"type":"final","turn_id":"1","text":"...","duration_ms":1234.0,"engine_ms":25.1,"total_ms":25.3}
```

The endpoint emits final transcription only after 350 ms of detected silence
or an 8-second maximum turn. Continue sending silence frames after a caller
stops speaking so VAD can close the final turn. It is VAD-endpointed streaming,
not token-by-token partial ASR.

When API keys are enabled, authenticate with `Authorization: Bearer TOKEN` or
`?api_key=TOKEN` (use the latter only for browser clients that cannot attach
headers during the WebSocket handshake).

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

Run the basic check after readiness:

```bash
PARAKEET_STT_API_KEY=YOUR_KEY ./scripts/smoke-test.sh
```
