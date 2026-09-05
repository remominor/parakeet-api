FROM ghcr.io/mudler/parakeet.cpp-server:latest-cuda@sha256:d11dd699119797efd82d757ecf0820a4adac1bfa436cffe942ece58f1b4a806a

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg python3 python3-venv tini \
    && python3 -m venv "$VIRTUAL_ENV" && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY gateway/requirements.txt /app/gateway/requirements.txt
RUN pip install --no-cache-dir -r /app/gateway/requirements.txt
COPY gateway /app/gateway
ARG SILERO_VAD_VERSION=6.2.1
ARG SILERO_VAD_SHA256=1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
RUN curl -fsSL --retry 3 \
      -o /app/gateway/silero_vad.onnx \
      "https://raw.githubusercontent.com/snakers4/silero-vad/v${SILERO_VAD_VERSION}/src/silero_vad/data/silero_vad.onnx" \
    && echo "${SILERO_VAD_SHA256}  /app/gateway/silero_vad.onnx" | sha256sum -c -
COPY docker-entrypoint.sh /usr/local/bin/parakeet-api
RUN chmod 0555 /usr/local/bin/parakeet-api && useradd --create-home --uid 10001 --shell /usr/sbin/nologin parakeet && mkdir -p /models && chown parakeet:parakeet /models
USER parakeet
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/parakeet-api"]
