#!/usr/bin/env python3
"""Reproducible API benchmark harness; writes machine-readable JSON."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
import sys
import resource

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate_confidence(body: dict) -> dict | None:
    if isinstance(body.get("confidence"), dict):
        return body["confidence"]
    values = [float(word.get("confidence", word.get("conf"))) for word in body.get("words", []) if word.get("confidence", word.get("conf")) is not None]
    if not values:
        return None
    return {"mean": sum(values) / len(values), "min": min(values), "low_word_count": sum(value < .7 for value in values)}


def clip(source: Path, seconds: float) -> bytes:
    from gateway.audio import decode_audio, wav_bytes
    import numpy as np
    decoded = decode_audio(source.read_bytes())
    pcm = decoded.pcm[:int(seconds * 16000)]
    pcm16 = (np.clip(pcm, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
    return wav_bytes(pcm16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5092")
    parser.add_argument("--sample", type=Path, default=Path("sample.wav"))
    parser.add_argument("--api-key")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--speech-context", choices=("none", "diarization", "full"), default="none")
    parser.add_argument("--measure-cold-load", action="store_true")
    parser.add_argument("--audio-url")
    args = parser.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    report = {"base_url": args.base_url, "sample": str(args.sample), "runs": args.runs, "model_size_bytes": args.model_file.stat().st_size if args.model_file else None, "fixtures": []}
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=300) as client:
        if args.measure_cold_load:
            client.post("/v1/model/unload").raise_for_status()
            for _ in range(300):
                if client.get("/health").status_code == 503: break
                time.sleep(.1)
            started = time.perf_counter(); client.post("/v1/model/load").raise_for_status()
            for _ in range(3000):
                if client.get("/readyz").status_code == 200: break
                time.sleep(.1)
            report["cold_load_ms"] = (time.perf_counter() - started) * 1000
        report["info"] = client.get("/info").json()
        for seconds in (2.5, 7.5, 12.0):
            audio = clip(args.sample, seconds); measurements = []; final = None
            cpu_started = resource.getrusage(resource.RUSAGE_SELF)
            for _ in range(args.runs):
                data = {"response_format": "verbose_json"}
                if args.speech_context != "none": data["speech_context"] = args.speech_context
                started = time.perf_counter(); response = client.post("/v1/audio/transcriptions", files={"file": ("sample.wav", audio, "audio/wav")}, data=data); elapsed = (time.perf_counter() - started) * 1000; response.raise_for_status(); measurements.append(elapsed); final = response
            cpu_finished = resource.getrusage(resource.RUSAGE_SELF)
            health = client.get("/health").json()
            body = final.json()
            report["fixtures"].append({"seconds": seconds, "warm_p50_ms": statistics.median(measurements), "warm_p95_ms": percentile(measurements, .95), "client_cpu_seconds": (cpu_finished.ru_utime + cpu_finished.ru_stime) - (cpu_started.ru_utime + cpu_started.ru_stime), "transcript": body.get("text"), "words": body.get("words"), "confidence": aggregate_confidence(body), "engine_ms": float(final.headers["x-parakeet-engine-ms"]), "speech_context": body.get("speech_context"), "process_gpu_memory_mb": health.get("vram_allocated_mb")})
        report["format_checks"] = {fmt: client.post("/v1/audio/transcriptions", files={"file": ("sample.wav", clip(args.sample, 2.5), "audio/wav")}, data={"response_format": fmt}).status_code for fmt in ("json", "text", "verbose_json", "srt", "vtt")}
        if args.audio_url:
            response = client.post("/v1/audio/transcriptions", data={"audio_url": args.audio_url}); report["audio_url_check"] = {"status_code": response.status_code, "body": response.text[:500]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
