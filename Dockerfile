ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04@sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04@sha256:92906d87596d638d35015c6353053121bd299d25943b875763321653884ba924

FROM ${CUDA_DEVEL_IMAGE} AS native-builder
ARG TRANSCRIBE_CPP_COMMIT=63a44d9239d610b3908e8a66b384924cd4a77217
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake git libopenblas-dev ninja-build python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
COPY patches/transcribe-sortformer-persistent-state.patch /tmp/transcribe-sortformer-persistent-state.patch
COPY patches/transcribe-sortformer-rust-bindings.patch /tmp/transcribe-sortformer-rust-bindings.patch
RUN git clone https://github.com/handy-computer/transcribe.cpp /src/transcribe.cpp \
    && git -C /src/transcribe.cpp checkout --detach "${TRANSCRIBE_CPP_COMMIT}" \
    && test "$(git -C /src/transcribe.cpp rev-parse HEAD)" = "${TRANSCRIBE_CPP_COMMIT}" \
    && git -C /src/transcribe.cpp apply --check /tmp/transcribe-sortformer-persistent-state.patch \
    && git -C /src/transcribe.cpp apply /tmp/transcribe-sortformer-persistent-state.patch \
    && git -C /src/transcribe.cpp apply --check /tmp/transcribe-sortformer-rust-bindings.patch \
    && git -C /src/transcribe.cpp apply /tmp/transcribe-sortformer-rust-bindings.patch
RUN cmake -S /src/transcribe.cpp -B /build/transcribe -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/transcribe \
      -DCMAKE_CUDA_ARCHITECTURES="86;89" \
      -DTRANSCRIBE_BUILD_SHARED=ON \
      -DTRANSCRIBE_CUDA=ON \
      -DTRANSCRIBE_BUILD_TESTS=OFF \
      -DTRANSCRIBE_BUILD_EXAMPLES=OFF \
    && cmake --build /build/transcribe --parallel \
    && cmake --install /build/transcribe
RUN python3 -m venv /wheel-venv \
    && /wheel-venv/bin/pip install --no-cache-dir build \
    && /wheel-venv/bin/python -m build --wheel --outdir /wheels /src/transcribe.cpp/bindings/python

FROM ${CUDA_RUNTIME_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TRANSCRIBE_LIBRARY=/opt/transcribe/lib/libtranscribe.so
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgomp1 libopenblas0-pthread python3 python3-venv tini \
    && python3 -m venv "$VIRTUAL_ENV" \
    && rm -rf /var/lib/apt/lists/*
COPY --from=native-builder /opt/transcribe /opt/transcribe
COPY --from=native-builder /wheels /wheels
WORKDIR /app
COPY gateway/requirements.txt /app/gateway/requirements.txt
RUN pip install --no-cache-dir --no-deps /wheels/transcribe_cpp-0.2.3-py3-none-any.whl \
    && pip install --no-cache-dir -r /app/gateway/requirements.txt \
    && rm -rf /wheels
COPY gateway /app/gateway
ARG SILERO_VAD_VERSION=6.2.1
ARG SILERO_VAD_SHA256=1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
RUN python3 - <<'PY'
import hashlib, pathlib, urllib.request
url = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/src/silero_vad/data/silero_vad.onnx"
path = pathlib.Path("/app/gateway/silero_vad.onnx")
path.write_bytes(urllib.request.urlopen(url).read())
assert hashlib.sha256(path.read_bytes()).hexdigest() == "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
PY
COPY docker-entrypoint.sh /usr/local/bin/parakeet-api
RUN chmod 0555 /usr/local/bin/parakeet-api \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin parakeet \
    && mkdir -p /models/asr /models/diarization /models/campp /data/speakers \
    && chown -R parakeet:parakeet /data/speakers
USER parakeet
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/parakeet-api"]
