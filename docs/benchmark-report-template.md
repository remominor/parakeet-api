# Speech stack benchmark report

- Date / commit:
- Host / OS / driver:
- GPU: RTX 4070 Ti SUPER / RTX 3070
- transcribe.cpp: v0.2.3 (`63a44d9239d610b3908e8a66b384924cd4a77217`)
- Execution: ASR only / sequential Sortformer / concurrent Sortformer / full stack
- Model and checksum:

| Input | Cold load | Warm p50 | Warm p95 | CPU | Process VRAM | Transcript delta |
|---|---:|---:|---:|---:|---:|---|
| 2.5 s `sample.wav` | | | | | | |
| 7.5 s `sample.wav` | | | | | | |
| 12 s `sample.wav` | | | | | | |

Record transcript, timed words, aggregate confidence, engine/component timings,
model size, URL input, and each response format. The RTX 3070 is the release
gate. Enable concurrent ASR/Sortformer only when full-context p95 improves at
least 10% without ASR regression, instability, or inadequate GPU headroom.

## Calibration

- Household matrix:
- Same-speaker distribution:
- Different-speaker distribution:
- Threshold:
- Ambiguity margin:
- False accepts / rejects:

Do not claim known-speaker acceptance while threshold or margin is unset.
