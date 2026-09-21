# Milestone 1 development benchmark

- Date: 2026-09-21
- Host: Linux 7.2.3, NVIDIA driver 610.57.04
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16,376 MiB
- Input: repository `sample.wav`, decoded once to mono 16-kHz PCM
- Runs: 10 warm local HTTP requests per duration; p95 is linearly interpolated
- Native runtime: transcribe.cpp 0.2.3, commit
  `63a44d9239d610b3908e8a66b384924cd4a77217`

The CPU column is benchmark-client CPU time across all ten requests, not
service CPU utilization. VRAM is exact process memory from `nvidia-smi`; it is
not an invented per-model allocation.

| Stack | Model bytes | Cold load | Input | Warm p50 | Warm p95 | Client CPU | Process VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| parakeet.cpp TDT Q8 (legacy) | 903,835,936 | 2,391.9 ms | 2.5 s | 23.1 ms | 25.8 ms | 0.010 s | 1,108 MiB |
|  |  |  | 7.5 s | 33.8 ms | 36.3 ms | 0.010 s | 1,108 MiB |
|  |  |  | 12.0 s | 45.7 ms | 47.7 ms | 0.010 s | 1,216 MiB |
| transcribe.cpp TDT Q8 | 729,574,912 | 631.7 ms | 2.5 s | 37.8 ms | 92.0 ms | 0.010 s | 942 MiB |
|  |  |  | 7.5 s | 105.4 ms | 124.8 ms | 0.010 s | 952 MiB |
|  |  |  | 12.0 s | 122.4 ms | 131.7 ms | 0.010 s | 960 MiB |
| Unified Q8, ASR only | 731,357,568 | 608.5 ms | 2.5 s | 34.8 ms | 46.4 ms | 0.013 s | 950 MiB |
|  |  |  | 7.5 s | 112.6 ms | 119.8 ms | 0.008 s | 954 MiB |
|  |  |  | 12.0 s | 128.3 ms | 140.0 ms | 0.009 s | 962 MiB |
| Unified + Sortformer + CAM++, sequential |  | 737.0 ms | 2.5 s | 88.1 ms | 116.3 ms | 0.011 s | 1,096 MiB |
|  |  |  | 7.5 s | 200.3 ms | 215.1 ms | 0.012 s | 1,108 MiB |
|  |  |  | 12.0 s | 240.9 ms | 253.0 ms | 0.011 s | 1,124 MiB |

All stacks produced the same complete 12-second transcript. The 7.5-second
TDT result ended with a period while Unified did not. JSON, text,
verbose-JSON, SRT, and VTT returned HTTP 200 for every native ASR variant.
Unified word confidence over the 12-second fixture was mean 0.9812 and minimum
0.9041 using the minimum native entropy-based token score per word. Legacy
scores are not directly comparable because parakeet.cpp used max-probability
confidence.

For the final full-stack request, component timings were 47.7 ms Sortformer,
61.8 ms CAM++, and 109.7 ms total enrichment. It returned one request-local
speaker, zero overlap, and `calibration_required`, as thresholds intentionally
remain unset. Four simultaneous full-stack requests completed successfully in
821.3 ms wall time; model-local locks serialized each native session.

## Open release gates

- No RTX 3070 is installed on this host. Its parity, p50/p95, concurrency, and
  VRAM gates remain mandatory before release.
- Concurrent same-request ASR/Sortformer execution was not enabled or claimed;
  the required RTX 3070 10% p95 win has not been demonstrated.
- A reproducible two-speaker overlap audio fixture and the household identity
  matrix were not available, so overlap acceptance and threshold/margin
  calibration remain open.
- The supplied CAM++ ONNX passed shape, checksum, feature, and inference checks
  on the three locally supplied official examples. A fresh official 3D-Speaker
  ONNX export was not available, so the `>= 0.99999` cross-export parity gate
  remains open. Use `scripts/validate-campp-parity.py` when it is available.
- The legacy parakeet.cpp image materially outperformed both transcribe.cpp
  models on this development GPU. It must remain available as a rollback until
  the RTX 3070 decision is complete; this report does not authorize deleting
  the historical image/model path.

Raw reports are generated with `scripts/benchmark.py`; they intentionally are
not treated as portable performance claims.
