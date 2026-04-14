"""Phrase-scale structure and progression features for hierarchical control.

This module intentionally reuses compact helpers from ``phrase_generator`` so
the new hierarchy stays aligned with the existing segment/beat/section logic
instead of reimplementing timing utilities from scratch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - import style depends on entry point
    from .phrase_generator import (
        SectionSpan,
        count_phrase_beats,
        load_json,
        load_sections,
        resolve_section_index,
        top_labels,
    )
except ImportError:  # pragma: no cover
    from phrase_generator import (  # type: ignore
        SectionSpan,
        count_phrase_beats,
        load_json,
        load_sections,
        resolve_section_index,
        top_labels,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALENCE_MAP = {
    "low valence": -1.0,
    "moderate valence": 0.0,
    "high valence": 1.0,
}
AROUSAL_MAP = {
    "low arousal": -1.0,
    "moderate arousal": 0.0,
    "high arousal": 1.0,
}
TENSION_MAP = {
    "low tension": -1.0,
    "moderate tension": 0.0,
    "high tension": 1.0,
}


@dataclass(frozen=True)
class StructureFeature:
    phrase_index: int
    start: float
    end: float
    duration: float
    beat_count: int
    section_index: Optional[int]
    section_role: str
    section_progress: float
    relative_position: float
    current_emotion_vector: tuple[float, ...]
    decayed_history_vector: tuple[float, ...]
    difference_vector: tuple[float, ...]
    normalized_difference_vector: tuple[float, ...]
    emotion_change_magnitude: float
    normalized_change: float
    intensity: float
    strength: float
    novelty: float
    valence: float
    arousal: float
    tension: float
    dominant_moods: tuple[str, ...]
    dominant_characteristics: tuple[str, ...]
    anchor_weights: tuple[float, ...]
    motif_signature: str
    motif_family: int
    motif_repeat_index: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def clamp_signed(value: float, limit: float = 1.0) -> float:
    return float(np.clip(value, -limit, limit))


def weighted_label_score(
    feature: Dict[str, Any],
    category: str,
    mapping: Dict[str, float],
    default: float = 0.0,
) -> float:
    items = feature.get(category, []) or []
    if not items:
        return default

    total = 0.0
    weight_sum = 0.0
    for item in items:
        label = item.get("label")
        score = float(item.get("score", 0.0))
        total += mapping.get(str(label), default) * score
        weight_sum += score

    if weight_sum <= 1e-6:
        return default
    return total / weight_sum


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-6:
        return 0.0
    similarity = float(np.dot(a, b) / denom)
    return 1.0 - float(np.clip(similarity, -1.0, 1.0))


def quantize_signed(value: float) -> str:
    if value >= 0.35:
        return "high"
    if value <= -0.35:
        return "low"
    return "mid"


def beat_bucket(beat_count: int) -> str:
    if beat_count <= 4:
        return "short"
    if beat_count <= 8:
        return "mid"
    return "long"


def overlap_seconds(start: float, end: float, other: Dict[str, Any]) -> float:
    other_start = float(other.get("start", 0.0))
    other_end = float(other.get("end", other_start))
    return max(0.0, min(end, other_end) - max(start, other_start))


def find_matching_anchor_weights(
    start: float,
    end: float,
    anchor_weight_segments: Sequence[Dict[str, Any]],
    cursor: int = 0,
) -> tuple[list[float], int]:
    if not anchor_weight_segments:
        return [], cursor

    while (
        cursor + 1 < len(anchor_weight_segments)
        and float(anchor_weight_segments[cursor].get("end", 0.0)) <= start
    ):
        cursor += 1

    best_index = cursor
    best_overlap = -1.0
    upper = min(len(anchor_weight_segments), cursor + 4)
    for index in range(max(0, cursor - 1), upper):
        overlap = overlap_seconds(start, end, anchor_weight_segments[index])
        if overlap > best_overlap:
            best_index = index
            best_overlap = overlap

    weights = anchor_weight_segments[best_index].get("weights", []) or []
    return [float(value) for value in weights], best_index


def normalize_nonnegative(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    total = float(vector.sum())
    if total <= 1e-6:
        return np.zeros_like(vector)
    return vector / total


def extract_current_emotion_vector(
    phrase: Dict[str, Any],
    anchor_weights: Sequence[float],
) -> np.ndarray:
    feature = phrase.get("feature", {}) or {}
    summary = np.array(
        [
            weighted_label_score(feature, "valence", VALENCE_MAP),
            weighted_label_score(feature, "arousal", AROUSAL_MAP),
            weighted_label_score(feature, "tension", TENSION_MAP),
        ],
        dtype=float,
    )
    if not anchor_weights:
        return summary

    normalized_anchors = normalize_nonnegative(anchor_weights)
    return np.concatenate([summary, normalized_anchors])


def build_motif_signature(
    phrase: Dict[str, Any],
    beat_count: int,
    valence: float,
    arousal: float,
    tension: float,
) -> str:
    feature = phrase.get("feature", {}) or {}
    moods = top_labels(feature, "moods", 2) or ["unknown"]
    characteristics = top_labels(feature, "characteristics", 2) or ["plain"]
    parts = [
        moods[0],
        moods[1] if len(moods) > 1 else "none",
        characteristics[0],
        quantize_signed(valence),
        quantize_signed(arousal),
        quantize_signed(tension),
        beat_bucket(beat_count),
    ]
    return "|".join(parts)


def infer_section_role(
    current: Dict[str, Any],
    previous: Optional[Dict[str, Any]],
) -> str:
    if previous is None:
        return "stable"

    section_changed = current["section_index"] != previous["section_index"]
    progress = float(current["section_progress"])
    normalized_change = float(current["normalized_change"])
    intensity = float(current["intensity"])
    diff_arousal = 0.0
    if len(current["difference_vector"]) > 1:
        diff_arousal = float(current["difference_vector"][1])

    if section_changed and normalized_change >= 0.25:
        return "transition"
    if normalized_change >= 0.75:
        return "transition"
    if progress <= 0.35 and normalized_change >= 0.25:
        return "buildup"
    if intensity >= 0.60 and progress >= 0.45:
        return "peak"
    if progress >= 0.75 or diff_arousal <= -0.18:
        return "release"
    return "stable"


def build_structure_features(
    phrases: Sequence[Dict[str, Any]],
    beat_times: Optional[Sequence[float]] = None,
    sections: Optional[Sequence[SectionSpan]] = None,
    anchor_weight_segments: Optional[Sequence[Dict[str, Any]]] = None,
    history_decay: float = 0.82,
) -> List[StructureFeature]:
    if not phrases:
        return []

    beat_times = list(beat_times or [])
    sections = list(sections or [])
    anchor_weight_segments = list(anchor_weight_segments or [])

    total_start = float(phrases[0].get("start", 0.0))
    total_end = float(phrases[-1].get("end", total_start + 1.0))
    total_duration = max(1e-6, total_end - total_start)

    probe_anchor_weights, _ = find_matching_anchor_weights(
        float(phrases[0].get("start", 0.0)),
        float(phrases[0].get("end", 0.0)),
        anchor_weight_segments,
    )
    vector_len = 3 + len(probe_anchor_weights)

    history = np.zeros(vector_len, dtype=float)
    last_current = np.zeros(vector_len, dtype=float)
    anchor_cursor = 0
    raw_records: List[Dict[str, Any]] = []

    for phrase_index, phrase in enumerate(phrases):
        start = float(phrase.get("start", 0.0))
        end = float(phrase.get("end", start))
        duration = max(1e-6, end - start)
        midpoint = 0.5 * (start + end)
        feature = phrase.get("feature", {}) or {}

        beat_count = count_phrase_beats(beat_times, start, end)
        section_index = resolve_section_index(start, end, sections)

        if section_index is not None and 0 <= section_index < len(sections):
            section = sections[section_index]
            section_span = max(1e-6, section.end - section.start)
            section_progress = clamp01((midpoint - section.start) / section_span)
        else:
            section_progress = clamp01((midpoint - total_start) / total_duration)

        anchor_weights, anchor_cursor = find_matching_anchor_weights(
            start,
            end,
            anchor_weight_segments,
            cursor=anchor_cursor,
        )
        current = extract_current_emotion_vector(phrase, anchor_weights)
        if len(current) < vector_len:
            current = np.pad(current, (0, vector_len - len(current)))

        if phrase_index == 0:
            difference = np.zeros_like(current)
            change_magnitude = 0.0
        else:
            difference = current - history
            change_magnitude = float(
                np.linalg.norm(difference[:3]) / np.sqrt(3.0)
            )

        valence = weighted_label_score(feature, "valence", VALENCE_MAP)
        arousal = weighted_label_score(feature, "arousal", AROUSAL_MAP)
        tension = weighted_label_score(feature, "tension", TENSION_MAP)

        intensity = clamp01(
            0.5 * max(arousal, 0.0)
            + 0.5 * ((tension + 1.0) * 0.5)
        )
        strength = clamp01(float(np.linalg.norm(current[:3]) / np.sqrt(3.0)))
        relative_position = clamp01((midpoint - total_start) / total_duration)

        motif_signature = build_motif_signature(
            phrase,
            beat_count,
            valence,
            arousal,
            tension,
        )
        dominant_moods = tuple(top_labels(feature, "moods", 2))
        dominant_characteristics = tuple(top_labels(feature, "characteristics", 2))

        raw_records.append(
            {
                "phrase_index": phrase_index,
                "start": start,
                "end": end,
                "duration": duration,
                "beat_count": beat_count,
                "section_index": section_index,
                "section_progress": section_progress,
                "relative_position": relative_position,
                "current_emotion_vector": tuple(float(value) for value in current),
                "decayed_history_vector": tuple(float(value) for value in history),
                "difference_vector": tuple(float(value) for value in difference),
                "normalized_difference_vector": tuple(float(value) for value in difference),
                "emotion_change_magnitude": change_magnitude,
                "normalized_change": change_magnitude,
                "intensity": intensity,
                "strength": strength,
                "novelty": change_magnitude,
                "valence": valence,
                "arousal": arousal,
                "tension": tension,
                "dominant_moods": dominant_moods,
                "dominant_characteristics": dominant_characteristics,
                "anchor_weights": tuple(float(value) for value in anchor_weights),
                "motif_signature": motif_signature,
            }
        )

        history = history_decay * history + (1.0 - history_decay) * current
        last_current = current

    if raw_records:
        change_scale = max(
            max(float(record["emotion_change_magnitude"]) for record in raw_records),
            1e-6,
        )
        diff_matrix = np.asarray(
            [record["difference_vector"] for record in raw_records],
            dtype=float,
        )
        diff_scale = np.max(np.abs(diff_matrix), axis=0)
        diff_scale[diff_scale < 1e-6] = 1.0

        for record in raw_records:
            diff_vector = np.asarray(record["difference_vector"], dtype=float)
            normalized_change = clamp01(
                float(record["emotion_change_magnitude"]) / change_scale
            )
            record["normalized_change"] = normalized_change
            record["novelty"] = normalized_change
            record["normalized_difference_vector"] = tuple(
                float(value) for value in (diff_vector / diff_scale)
            )

    motif_families: Dict[str, int] = {}
    motif_counts: Dict[str, int] = {}
    next_family = 0
    previous_record: Optional[Dict[str, Any]] = None
    structured: List[StructureFeature] = []

    for record in raw_records:
        signature = record["motif_signature"]
        if signature not in motif_families:
            motif_families[signature] = next_family
            next_family += 1

        repeat_index = motif_counts.get(signature, 0)
        motif_counts[signature] = repeat_index + 1
        record["motif_family"] = motif_families[signature]
        record["motif_repeat_index"] = repeat_index
        record["section_role"] = infer_section_role(record, previous_record)

        structured.append(StructureFeature(**record))
        previous_record = record

    return structured


def build_structure_features_from_paths(
    clap_results_path: str = "json/clap_results.json",
    beat_times_path: str = "json/beat_times.json",
    sections_path: str = "json/sections.json",
    anchor_weights_path: str = "json/clap_weights.json",
    history_decay: float = 0.82,
) -> List[StructureFeature]:
    phrases = load_json(clap_results_path, default=[]) or []
    beat_times = load_json(beat_times_path, default=[]) or []
    sections = load_sections(sections_path)
    anchor_weight_segments = load_json(anchor_weights_path, default=[]) or []

    return build_structure_features(
        phrases,
        beat_times=beat_times,
        sections=sections,
        anchor_weight_segments=anchor_weight_segments,
        history_decay=history_decay,
    )
