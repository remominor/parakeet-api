# Parakeet local speech-intelligence API

A single CUDA container exposes an OpenAI-compatible transcription gateway on
port `5092`. Inference is in-process through transcribe.cpp; there is no CLI
output parsing or speech-engine sidecar. Completed-turn WebSockets retain
Silero VAD and do not provide token-by-token partials.

## Pinned runtime and models

- transcribe.cpp `v0.2.3`, commit `63a44d9239d610b3908e8a66b384924cd4a77217`
- Unified Q8 (default), 731,357,568 bytes,
  `4b50b6dd862bf6e346929aaf4f5eaacec003bfa3f56462d6c874b41ef2f38795`
- transcribe.cpp TDT-v2 Q8 (rollback), 729,574,912 bytes,
  `f0d0e99cebb6d3b83f1f7069b82b5d3c2e39a54545b0da039cb4bafd9c4e5caa`
- Sortformer v2.1 Q8, 139,310,336 bytes,
  `a5dacdc650790266c7a362e54e6bf51952015487edaa606c4e11632bc32442a9`
- English VoxCeleb CAM++, 29,596,978 bytes,
  `357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b`

The older 903-MB `tdt-0.6b-v2-q8_0.gguf` is a parakeet.cpp artifact and is not
compatible with transcribe.cpp. It is intentionally rejected. Verify the
files with `(cd models && sha256sum -c SHA256SUMS.txt)` (missing historical
files may be reported separately).

## Run

```bash
mkdir -p data/speakers
docker compose up --build
```

Models may be mounted at `/models/{asr,diarization,campp}` or at `/models`.
`PARAKEET_ASR_MODEL_FILE` has precedence; `PARAKEET_MODEL_FILE` remains a
compatibility fallback. Unified is the default. Set
`PARAKEET_ASR_MODEL_FILE=parakeet-tdt-0.6b-v2-Q8_0.gguf` for rollback.

Sortformer and identity are opt-in:

```dotenv
PARAKEET_DIARIZATION_ENABLED=true
PARAKEET_IDENTITY_ENABLED=true
```

The service accepts `parakeet`, `parakeet-en`, and `whisper-1`. It advertises
the actual loaded model as `parakeet-unified-en-0.6b` or
`parakeet-tdt-0.6b-v2`; the exact TDT ID is accepted only while TDT is loaded.

## APIs

`POST /v1/audio/transcriptions` preserves `json`, `text`, `verbose_json`,
`srt`, and `vtt`, file/URL rules, authentication, request IDs, timing headers,
and WAV passthrough semantics. Add `speech_context=diarization` or `full` to a
JSON response. Enriched requests always include timed words and a
`speech_context` object with segments, speakers, attribution, overlap
statistics, component state, timing, and machine-readable degradation errors.
ASR failure fails the request; optional enrichment fails open.

```bash
curl http://127.0.0.1:5092/v1/audio/transcriptions \
  -H 'Authorization: Bearer YOUR_KEY' \
  -F file=@sample.wav -F model=parakeet \
  -F response_format=verbose_json -F speech_context=full
```

Additional authenticated endpoints:

- `POST /v1/audio/diarizations` (`file` or `audio_url`, `identify=false`)
- `POST /v1/speakers/enroll` (`speaker_id`, optional `display_name`, repeated
  `files`)
- `GET /v1/speakers` and `GET /v1/speakers/{speaker_id}`
- `DELETE /v1/speakers/{speaker_id}`
- `POST /v1/speakers/verify` (one `file`, optional `speaker_id`)
- `POST /v1/model/load` and `POST /v1/model/unload`
- `GET /health`, `/readyz`, `/info`, `/stats`, and opt-in `/metrics`

`/readyz` is ASR decisive: optional component degradation does not make the
service unready. GPU memory is process-level because native per-model
allocation is unavailable; component VRAM remains `null` rather than being
invented.

### Speaker identity and privacy

CAM++ uses 16-kHz audio, 80-bin Kaldi-compatible fbank, dither 0, per-utterance
feature mean subtraction, and L2-normalized 512-D embeddings. Identity uses
only non-overlapping diarized regions of at least 500 ms, requires 1,500 ms of
usable audio by default, rejects near-silence, and caps processing at 15 s per
speaker.

Templates are atomic mode-`0600` JSON records in `/data/speakers`; the
directory is mode `0700`. Public APIs never return an embedding. Corrupt or
model-incompatible records are listed as incompatible and excluded from
matching. Enrollment never overwrites an ID.

Threshold and ambiguity margin are unset by default. Enrollment and embedding
remain available, but matching returns `calibration_required`; full-context
transcription still returns diarization. Calibrate from a labeled JSONL file:

```json
{"speaker_id":"alice","audio":"fixtures/alice-1.wav"}
{"speaker_id":"alice","audio":"fixtures/alice-2.wav"}
{"speaker_id":"bob","audio":"fixtures/bob-1.wav"}
```

```bash
python scripts/calibrate-speakers.py manifest.jsonl \
  --model models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx
```

The command reports same/different distributions and candidate threshold and
margin ranges; it never changes configuration. Validate the prebuilt ONNX
against a fresh official export with `scripts/validate-campp-parity.py`; every
fixture must achieve cosine similarity `>= 0.99999`. Torch, Torchaudio, and
ModelScope exist only in `gateway/requirements-campp-parity.txt`, not the
production image.

## WebSockets

- `/v1/audio/transcriptions/ws`: raw PCM16 mono at 16 kHz, native lifecycle
  events, `commit`, `clear`, and per-connection VAD configuration.
- `/v1/realtime`: OpenAI-style transcription sessions, base64 PCM16 at 24 kHz,
  server VAD or manual commit.

Both use the same in-process ASR abstraction. Queue depth remains two,
oversized frames are rejected, and no WebSocket diarization is performed.

## Verification and benchmarking

Normal tests do not download models or require CUDA:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Set `PARAKEET_REAL_ASR_MODEL`, `PARAKEET_REAL_DIARIZATION_MODEL`, and/or
`PARAKEET_REAL_CAMPP_MODEL` for opt-in native tests. The benchmark harness
covers 2.5 s, 7.5 s, and 12 s clips and records transcript, words, timings,
model size, and process VRAM:

```bash
python scripts/benchmark.py --output report.json \
  --model-file models/parakeet-unified-en-0.6b-Q8_0.gguf
```

See [the measured RTX 4070 Ti SUPER development report](docs/benchmark-rtx4070ti-super.md)
and use [the report template](docs/benchmark-report-template.md) for old
parakeet.cpp TDT, transcribe.cpp TDT, Unified, sequential/concurrent
Sortformer, and full-stack comparisons. Development may use the RTX 4070 Ti
SUPER, but release latency, concurrency, VRAM, and parity gates must be run on
the RTX 3070. Sequential enrichment is the default; concurrency must not be
enabled without the documented 10% full-context p95 win and adequate headroom.

## Configuration

Important variables are `PARAKEET_API_KEYS`, `PARAKEET_ASR_MODEL_FILE`,
`PARAKEET_DIARIZATION_ENABLED`, `PARAKEET_DIARIZATION_MODEL_FILE`,
`PARAKEET_IDENTITY_ENABLED`, `PARAKEET_CAMPP_MODEL_FILE`,
`PARAKEET_ASR_DEVICE`, `PARAKEET_DIARIZATION_DEVICE`,
`PARAKEET_SPEAKER_STORE`, `PARAKEET_IDENTITY_THRESHOLD`,
`PARAKEET_IDENTITY_MARGIN`, `PARAKEET_IDENTITY_MINIMUM_AUDIO_MS`, and the
`PARAKEET_CAMPP_*THREADS`/`CONCURRENCY` controls. Docker/NVIDIA visibility
controls still determine which devices exist; selectors resolve exact devices
from `transcribe_cpp.backends()`.

Word confidence intentionally changed from parakeet.cpp max probability to the
minimum native entropy-based token confidence belonging to each word.

## Milestone 2

Stateful WebSocket diarization is explicitly deferred. Guest tracking, speech
separation, and other stateful multi-turn speaker behavior are also out of
scope for this milestone.
