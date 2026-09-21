"""Deterministic interval analysis and word/speaker fusion."""
from __future__ import annotations

from collections import defaultdict

from .backends import SpeakerSegment, Word

EPSILON = 1e-9


def _events(segments: list[SpeakerSegment]) -> list[float]:
    return sorted({max(0.0, item.start) for item in segments} | {max(0.0, item.end) for item in segments})


def _active(segments: list[SpeakerSegment], start: float, end: float) -> set[str]:
    midpoint = (start + end) / 2
    return {item.speaker for item in segments if item.start <= midpoint < item.end}


def interval_statistics(segments: list[SpeakerSegment]) -> dict:
    points = _events(segments)
    durations: dict[str, float] = defaultdict(float)
    speech = overlap = 0.0
    single_regions: list[str] = []
    for start, end in zip(points, points[1:]):
        if end <= start:
            continue
        active = _active(segments, start, end)
        length = end - start
        if active:
            speech += length
        if len(active) >= 2:
            overlap += length
        for speaker in active:
            durations[speaker] += length
        if len(active) == 1:
            label = next(iter(active))
            if not single_regions or single_regions[-1] != label:
                single_regions.append(label)
    arrival = {item.speaker: index for index, item in enumerate(segments) if item.speaker not in {x.speaker for x in segments[:index]}}
    dominant = min(durations, key=lambda name: (-durations[name], arrival.get(name, 10**9))) if durations else None
    return {
        "speech_duration": round(speech, 6),
        "overlap_duration": round(overlap, 6),
        "overlap_ratio": round(overlap / speech, 6) if speech else 0.0,
        "dominant_speaker": dominant,
        "speaker_changes": sum(a != b for a, b in zip(single_regions, single_regions[1:])),
        "speaker_durations": {key: round(value, 6) for key, value in durations.items()},
    }


def _speaker_coverage(segments: list[SpeakerSegment], speaker: str, start: float, end: float) -> float:
    """Measure the union of one speaker's intervals inside a word."""
    clipped = sorted(
        (max(start, item.start), min(end, item.end))
        for item in segments
        if item.speaker == speaker and min(end, item.end) > max(start, item.start)
    )
    total = 0.0
    merged_start = merged_end = None
    for region_start, region_end in clipped:
        if merged_start is None:
            merged_start, merged_end = region_start, region_end
        elif region_start <= merged_end + EPSILON:
            merged_end = max(merged_end, region_end)
        else:
            total += merged_end - merged_start
            merged_start, merged_end = region_start, region_end
    if merged_start is not None:
        total += merged_end - merged_start
    return total


def attribute_words(words: list[Word], segments: list[SpeakerSegment]) -> list[Word]:
    arrival: list[str] = []
    for segment in segments:
        if segment.speaker not in arrival:
            arrival.append(segment.speaker)
    for word in words:
        if word.end <= word.start + EPSILON:
            contained = {item.speaker for item in segments if item.start <= word.start < item.end}
            word.speaker = next(iter(contained)) if len(contained) == 1 else ("overlap" if len(contained) > 1 else "unattributed")
            continue
        duration = word.end - word.start
        coverage = {
            speaker: _speaker_coverage(segments, speaker, word.start, word.end)
            for speaker in arrival
        }
        ordered = sorted(coverage, key=lambda name: (-coverage[name], arrival.index(name)))
        if ordered and coverage[ordered[0]] / duration >= 0.5 and all(coverage[name] / duration < 0.2 for name in ordered[1:]):
            word.speaker = ordered[0]
        elif sum(value > EPSILON for value in coverage.values()) >= 2:
            word.speaker = "overlap"
        else:
            word.speaker = "unattributed"
    return words


def clean_regions(segments: list[SpeakerSegment], speaker: str, minimum_seconds: float = 0.5) -> list[tuple[float, float]]:
    points = _events(segments)
    raw: list[tuple[float, float]] = []
    for start, end in zip(points, points[1:]):
        if end > start and _active(segments, start, end) == {speaker}:
            if raw and abs(raw[-1][1] - start) < EPSILON:
                raw[-1] = (raw[-1][0], end)
            else:
                raw.append((start, end))
    return [(start, end) for start, end in raw if end - start >= minimum_seconds]
