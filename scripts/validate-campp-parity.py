#!/usr/bin/env python3
"""Compare a candidate CAM++ ONNX against a freshly exported official ONNX."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gateway.audio import decode_audio
from gateway.speakers import CampPlusONNX


async def main_async(args) -> None:
    candidate = CampPlusONNX(args.candidate)
    reference = CampPlusONNX.__new__(CampPlusONNX)
    # The official export may have a different checksum; retain all shape/name checks.
    import onnxruntime as ort
    reference.path = args.reference
    reference._session = ort.InferenceSession(str(args.reference), providers=["CPUExecutionProvider"])
    inputs, outputs = reference._session.get_inputs(), reference._session.get_outputs()
    if len(inputs) != 1 or inputs[0].name not in {"x", "feature"} or len(inputs[0].shape) != 3 or inputs[0].shape[-1] != 80:
        raise RuntimeError(f"invalid reference input: {[(item.name, item.shape) for item in inputs]}")
    if len(outputs) != 1 or len(outputs[0].shape) != 2 or outputs[0].shape[-1] != 512:
        raise RuntimeError(f"invalid reference output: {[(item.name, item.shape) for item in outputs]}")
    reference._input_name, reference._output_name = inputs[0].name, outputs[0].name
    reference._semaphore = asyncio.Semaphore(1)
    similarities = []
    for path in args.audio:
        pcm = decode_audio(path.read_bytes()).pcm
        left, right = await candidate.embed(pcm), await reference.embed(pcm)
        similarities.append(float(np.dot(left, right)))
    for path, score in zip(args.audio, similarities):
        print(f"{score:.9f}\t{path}")
    if not similarities or min(similarities) < args.minimum:
        raise SystemExit(f"CAM++ parity failed: minimum={min(similarities, default=float('nan')):.9f}, required={args.minimum}")
    print(f"parity passed: minimum cosine={min(similarities):.9f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path, help="fresh official 3D-Speaker export")
    parser.add_argument("--audio", required=True, type=Path, nargs="+")
    parser.add_argument("--minimum", type=float, default=.99999)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
