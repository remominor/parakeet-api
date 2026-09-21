# Third-party notices

The native speech-recognition and diarization runtime is
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), pinned to
release 0.2.3 and commit `63a44d9239d610b3908e8a66b384924cd4a77217`.
See that project's license for its terms.

Speaker embeddings use the English VoxCeleb CAM++ checkpoint
`iic/speech_campplus_sv_en_voxceleb_16k`, revision `v1.0.2`, distributed under
the Apache License 2.0. Production feature extraction is provided by
`kaldi-native-fbank`, and inference by ONNX Runtime; see those projects'
licenses for their terms.

The WebSocket endpoint bundles `silero_vad.onnx` from
[snakers4/silero-vad](https://github.com/snakers4/silero-vad), version 6.2.1.
The model is used only for CPU voice activity detection; it does not perform
speech recognition. See that project's MIT license for its terms.
