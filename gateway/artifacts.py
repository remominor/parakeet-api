"""Pinned model artifact identities for Milestone 1."""
from __future__ import annotations

import hashlib
from pathlib import Path

ARTIFACTS = {
    "parakeet-unified-en-0.6b-Q8_0.gguf": (731_357_568, "4b50b6dd862bf6e346929aaf4f5eaacec003bfa3f56462d6c874b41ef2f38795"),
    "diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf": (139_310_336, "a5dacdc650790266c7a362e54e6bf51952015487edaa606c4e11632bc32442a9"),
    "parakeet-tdt-0.6b-v2-Q8_0.gguf": (729_574_912, "f0d0e99cebb6d3b83f1f7069b82b5d3c2e39a54545b0da039cb4bafd9c4e5caa"),
    "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx": (29_596_978, "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: str | Path, *, allow_unpinned: bool = False) -> str:
    path = Path(path)
    expected = ARTIFACTS.get(path.name)
    if expected is None:
        if allow_unpinned:
            return sha256_file(path)
        raise ValueError(f"unrecognized model artifact {path.name!r}; set PARAKEET_ALLOW_UNPINNED_MODELS=true only for controlled testing")
    size, digest = expected
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ValueError(f"model size mismatch for {path.name}: expected {size}, got {actual_size}")
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError(f"model checksum mismatch for {path.name}: expected {digest}, got {actual}")
    return actual
