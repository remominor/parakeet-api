#!/usr/bin/env python3
"""Report CAM++ score distributions and calibration candidates without writing config."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gateway.audio import decode_audio
from gateway.speakers import CampPlusONNX


async def run(args) -> dict:
    backend = CampPlusONNX(args.model, intra_threads=args.threads)
    entries = []
    for line_number, line in enumerate(Path(args.manifest).read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item.get("speaker_id"), str) or not isinstance(item.get("audio"), str):
            raise ValueError(f"manifest line {line_number} requires string speaker_id and audio")
        path = Path(item["audio"])
        decoded = decode_audio(path.read_bytes())
        embedding = await backend.embed(decoded.pcm[:15 * 16000])
        entries.append((item["speaker_id"], str(path), embedding))
    same, different = [], []
    for left, right in combinations(entries, 2):
        score = float(np.dot(left[2], right[2]))
        (same if left[0] == right[0] else different).append(score)
    def summary(values):
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None, "p05": None, "p95": None}
        array = np.asarray(values)
        return {"count": len(values), "min": float(array.min()), "max": float(array.max()), "mean": float(array.mean()), "p05": float(np.quantile(array, .05)), "p95": float(np.quantile(array, .95))}
    same_summary, different_summary = summary(same), summary(different)
    lower = different_summary["p95"]
    upper = same_summary["p05"]
    thresholds = None if lower is None or upper is None else {"low": lower, "high": upper, "midpoint": (lower + upper) / 2, "separable": lower < upper}
    margins = []
    for speaker_id, _, query in entries:
        own = [float(np.dot(query, other)) for owner, _, other in entries if owner == speaker_id and not np.shares_memory(query, other)]
        impostors = [float(np.dot(query, other)) for owner, _, other in entries if owner != speaker_id]
        if own and impostors:
            margins.append(max(own) - max(impostors))
    output = {"model": str(args.model), "manifest": str(args.manifest), "samples": len(entries), "same_speaker": same_summary, "different_speaker": different_summary, "recommended_threshold_range": thresholds, "top1_margin": summary(margins), "recommended_margin_range": None if not margins else {"low": float(np.quantile(margins, .05)), "high": float(np.quantile(margins, .25))}, "note": "Review false accepts/rejects and household validation before setting production thresholds."}
    await backend.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="JSONL with speaker_id and audio fields")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
