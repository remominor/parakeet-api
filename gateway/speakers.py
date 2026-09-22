"""CAM++ speaker embeddings, private template storage, and identity matching."""
from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .audio import SAMPLE_RATE
from .artifacts import sha256_file
from .backends import SpeakerSegment
from .fusion import clean_regions

CAMPP_MODEL = "iic/speech_campplus_sv_en_voxceleb_16k"
CAMPP_REVISION = "v1.0.2"
CAMPP_SHA256 = "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b"
PREPROCESSING = "kaldi-fbank-16k-80bin-dither0-utterance-mean-v1"
SPEAKER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def l2_normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("speaker embedding has zero or non-finite norm")
    return array / norm


class CampPlusONNX:
    dimension = 512

    def __init__(self, path: str | Path, *, intra_threads: int = 0, inter_threads: int = 1, concurrency: int = 1):
        import onnxruntime as ort

        self.path = Path(path)
        actual = sha256_file(self.path)
        if actual != CAMPP_SHA256:
            raise ValueError(f"CAM++ checksum mismatch: expected {CAMPP_SHA256}, got {actual}")
        options = ort.SessionOptions()
        options.inter_op_num_threads = inter_threads
        if intra_threads:
            options.intra_op_num_threads = intra_threads
        self._session = ort.InferenceSession(str(self.path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].name not in {"x", "feature"}:
            raise ValueError("CAM++ ONNX must have one input named x or feature and one output")
        input_shape, output_shape = inputs[0].shape, outputs[0].shape
        if len(input_shape) != 3 or input_shape[-1] not in {80, "80", None}:
            raise ValueError(f"CAM++ input must be [batch, frames, 80], got {input_shape}")
        if len(output_shape) != 2 or output_shape[-1] not in {512, "512", None}:
            raise ValueError(f"CAM++ output must be [batch, 512], got {output_shape}")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._semaphore = asyncio.Semaphore(max(1, concurrency))

    @staticmethod
    def features(pcm: np.ndarray) -> np.ndarray:
        import kaldi_native_fbank as knf

        options = knf.FbankOptions()
        options.frame_opts.dither = 0.0
        options.frame_opts.samp_freq = float(SAMPLE_RATE)
        options.mel_opts.num_bins = 80
        fbank = knf.OnlineFbank(options)
        fbank.accept_waveform(SAMPLE_RATE, np.asarray(pcm, dtype=np.float32).tolist())
        fbank.input_finished()
        if fbank.num_frames_ready == 0:
            raise ValueError("audio is too short for CAM++ fbank extraction")
        features = np.stack([fbank.get_frame(index) for index in range(fbank.num_frames_ready)]).astype(np.float32)
        features -= features.mean(axis=0, keepdims=True)
        return features[None, :, :]

    async def embed(self, pcm: np.ndarray) -> np.ndarray:
        features = await asyncio.to_thread(self.features, pcm)
        async with self._semaphore:
            output = await asyncio.to_thread(
                self._session.run, [self._output_name], {self._input_name: features}
            )
        embedding = np.asarray(output[0], dtype=np.float32)
        if embedding.shape != (1, self.dimension):
            raise ValueError(f"CAM++ returned {embedding.shape}, expected [1, 512]")
        return l2_normalize(embedding[0])

    async def close(self) -> None:
        self._session = None


@dataclass
class SpeakerRecord:
    speaker_id: str
    display_name: str | None
    embedding: np.ndarray | None
    sample_count: int
    created_at: str
    updated_at: str
    compatibility: str = "compatible"
    error: str | None = None

    def public(self) -> dict:
        out = {
            "speaker_id": self.speaker_id,
            "display_name": self.display_name,
            "sample_count": self.sample_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "compatibility": self.compatibility,
        }
        if self.error:
            out["error"] = self.error
        return out


class SpeakerStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        uid, gid = os.geteuid(), os.getegid()
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as exc:
            raise RuntimeError(
                f"speaker store {self.directory} is not writable by uid {uid}; "
                f"fix volume ownership to {uid}:{gid}"
            ) from exc

        # A Docker/Unraid mount may replace the image's pre-owned directory.
        # chmod can then fail even when an ACL already grants this process
        # private, writable access, so validate the actual mount before failing.
        try:
            os.chmod(self.directory, 0o700)
        except OSError:
            pass
        try:
            mode = stat.S_IMODE(self.directory.stat().st_mode)
            writable = os.access(self.directory, os.R_OK | os.W_OK | os.X_OK)
        except OSError as exc:
            raise RuntimeError(
                f"speaker store {self.directory} is not writable by uid {uid}; "
                f"fix volume ownership to {uid}:{gid}"
            ) from exc
        if not writable or mode & 0o077:
            raise RuntimeError(
                f"speaker store {self.directory} is not writable by uid {uid} or private enough "
                f"(requires mode 0700); fix volume ownership to {uid}:{gid}"
            )

    def _path(self, speaker_id: str) -> Path:
        if not SPEAKER_ID_RE.fullmatch(speaker_id):
            raise ValueError("speaker_id must match [a-z0-9][a-z0-9_-]{0,63}")
        return self.directory / f"{speaker_id}.json"

    def exists(self, speaker_id: str) -> bool:
        return self._path(speaker_id).exists()

    def create(self, speaker_id: str, display_name: str | None, embeddings: Iterable[np.ndarray]) -> SpeakerRecord:
        path = self._path(speaker_id)
        vectors = [l2_normalize(item) for item in embeddings]
        if not vectors or any(item.shape != (512,) for item in vectors):
            raise ValueError("enrollment requires one or more 512-dimensional embeddings")
        template = l2_normalize(np.mean(np.stack(vectors), axis=0))
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema_version": 1,
            "speaker_id": speaker_id,
            "display_name": display_name,
            "model": CAMPP_MODEL,
            "revision": CAMPP_REVISION,
            "onnx_sha256": CAMPP_SHA256,
            "preprocessing": PREPROCESSING,
            "embedding_dimension": 512,
            "sample_count": len(vectors),
            "created_at": now,
            "updated_at": now,
            "embedding": template.tolist(),
        }
        fd, temporary = tempfile.mkstemp(prefix=f".{speaker_id}.", suffix=".tmp", dir=self.directory)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            # A same-filesystem hard link is an atomic no-replace publish:
            # exactly one concurrent creator can claim the final name.
            os.link(temporary, path)
            directory_fd = os.open(self.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return self.get(speaker_id)

    def _read(self, path: Path) -> SpeakerRecord:
        speaker_id = path.stem
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            required = {
                "schema_version": 1,
                "model": CAMPP_MODEL,
                "revision": CAMPP_REVISION,
                "onnx_sha256": CAMPP_SHA256,
                "preprocessing": PREPROCESSING,
                "embedding_dimension": 512,
            }
            if any(data.get(key) != value for key, value in required.items()):
                raise ValueError("record metadata is incompatible with the loaded embedding model")
            if data.get("speaker_id") != speaker_id or not SPEAKER_ID_RE.fullmatch(speaker_id):
                raise ValueError("record speaker_id is invalid")
            embedding = l2_normalize(np.asarray(data["embedding"], dtype=np.float32))
            if embedding.shape != (512,):
                raise ValueError("record embedding is not 512-dimensional")
            return SpeakerRecord(
                speaker_id, data.get("display_name"), embedding, int(data["sample_count"]),
                str(data["created_at"]), str(data["updated_at"]),
            )
        except Exception as exc:
            return SpeakerRecord(speaker_id, None, None, 0, "", "", "incompatible", str(exc))

    def list(self) -> list[SpeakerRecord]:
        return [self._read(path) for path in sorted(self.directory.glob("*.json"))]

    def get(self, speaker_id: str) -> SpeakerRecord:
        path = self._path(speaker_id)
        if not path.exists():
            raise KeyError(speaker_id)
        return self._read(path)

    def delete(self, speaker_id: str) -> bool:
        path = self._path(speaker_id)
        if not path.exists():
            return False
        path.unlink()
        return True


def match_embedding(
    embedding: np.ndarray,
    records: list[SpeakerRecord],
    *,
    threshold: float | None,
    margin: float | None,
    target: str | None = None,
) -> dict:
    compatible = [item for item in records if item.compatibility == "compatible" and item.embedding is not None]
    if target is not None:
        compatible = [item for item in compatible if item.speaker_id == target]
    if not compatible:
        return {"status": "calibration_required"} if threshold is None else {"status": "unknown", "score": None}
    query = l2_normalize(embedding)
    ranked = sorted(((float(np.dot(query, item.embedding)), item) for item in compatible), reverse=True, key=lambda value: value[0])
    top_score, top = ranked[0]
    if threshold is None:
        if target is not None:
            return {"status": "calibration_required", "speaker_id": top.speaker_id, "display_name": top.display_name, "score": round(top_score, 6)}
        return {"status": "calibration_required", "candidate_speaker_id": top.speaker_id, "candidate_display_name": top.display_name, "candidate_score": round(top_score, 6)}
    if target is None and len(compatible) > 1 and margin is None:
        return {"status": "calibration_required", "candidate_speaker_id": top.speaker_id, "candidate_display_name": top.display_name, "candidate_score": round(top_score, 6)}
    if top_score < threshold:
        return {"status": "unknown", "score": round(top_score, 6)}
    if target is None and len(ranked) > 1 and top_score - ranked[1][0] < float(margin):
        return {"status": "ambiguous", "score": round(top_score, 6)}
    return {"status": "known", "speaker_id": top.speaker_id, "display_name": top.display_name, "score": round(top_score, 6)}


class IdentityService:
    def __init__(self, backend: CampPlusONNX, store: SpeakerStore, *, threshold: float | None, margin: float | None, minimum_audio: float = 1.5):
        self.backend = backend
        self.store = store
        self.threshold = threshold
        self.margin = margin
        self.minimum_audio = minimum_audio

    async def embedding_for_regions(self, pcm: np.ndarray, regions: list[tuple[float, float]]) -> np.ndarray | None:
        chosen: list[tuple[np.ndarray, float]] = []
        remaining = 15.0
        for start, end in regions:
            duration = min(end - start, remaining)
            if duration <= 0:
                continue
            chunk = pcm[int(start * SAMPLE_RATE):int((start + duration) * SAMPLE_RATE)]
            if len(chunk) < SAMPLE_RATE // 2 or float(np.sqrt(np.mean(np.square(chunk)))) < 1e-4:
                continue
            chosen.append((await self.backend.embed(chunk), duration))
            remaining -= duration
            if remaining <= 0:
                break
        total = sum(duration for _, duration in chosen)
        if total < self.minimum_audio:
            return None
        return l2_normalize(sum(vector * duration for vector, duration in chosen) / total)

    async def identify(self, pcm: np.ndarray, segments: list[SpeakerSegment]) -> dict[str, dict]:
        output: dict[str, dict] = {}
        records = self.store.list()
        for speaker in dict.fromkeys(item.speaker for item in segments):
            embedding = await self.embedding_for_regions(pcm, clean_regions(segments, speaker))
            output[speaker] = (
                {"status": "insufficient_audio"}
                if embedding is None
                else match_embedding(embedding, records, threshold=self.threshold, margin=self.margin)
            )
        return output
