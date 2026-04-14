"""Low-level embodied modulation for the hierarchical choreography pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - import style depends on entry point
    from .mind_planner import MacroState
    from .structure_features import StructureFeature, clamp01
except ImportError:  # pragma: no cover
    from mind_planner import MacroState  # type: ignore
    from structure_features import StructureFeature, clamp01  # type: ignore


BODY_PATTERNS = ("breath", "heartbeat", "stagger", "glide")


@dataclass(frozen=True)
class BodyState:
    phrase_index: int
    motif_seed: int
    beat_pattern: str
    body_amplitude: float
    pulse_gain: float
    breath_gain: float
    sway_gain: float
    swirl_gain: float
    tangential_gain: float
    tightness: float
    accent_gain: float
    dz_pulse: float
    dr_pulse: float
    dtheta_pulse: float
    separation_delta: float
    alignment_delta: float
    cohesion_delta: float
    goal_weight_delta: float
    jitter_delta: float
    accent_events: tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phrase_index": self.phrase_index,
            "motif_seed": self.motif_seed,
            "beat_pattern": self.beat_pattern,
            "body_amplitude": self.body_amplitude,
            "pulse_gain": self.pulse_gain,
            "breath_gain": self.breath_gain,
            "sway_gain": self.sway_gain,
            "swirl_gain": self.swirl_gain,
            "tangential_gain": self.tangential_gain,
            "tightness": self.tightness,
            "accent_gain": self.accent_gain,
            "dz_pulse": self.dz_pulse,
            "dr_pulse": self.dr_pulse,
            "dtheta_pulse": self.dtheta_pulse,
            "boid_deltas": {
                "separation": self.separation_delta,
                "alignment": self.alignment_delta,
                "cohesion": self.cohesion_delta,
                "goal_weight": self.goal_weight_delta,
                "jitter": self.jitter_delta,
            },
            "accent_events": list(self.accent_events),
        }


def stable_unit_values(signature: str, count: int = 4) -> List[float]:
    digest = hashlib.sha1(signature.encode("utf-8")).digest()
    values: List[float] = []
    for index in range(count):
        chunk = digest[index * 4 : index * 4 + 4]
        integer = int.from_bytes(chunk, "big", signed=False)
        values.append((integer % 10000) / 9999.0)
    return values


def bounded_delta(value: float, bound: float) -> float:
    return float(np.clip(value, -bound, bound))


def choose_beat_pattern(
    feature: StructureFeature,
    macro: MacroState,
    motif_template: Sequence[float],
) -> str:
    base_pattern = BODY_PATTERNS[int(motif_template[0] * len(BODY_PATTERNS)) % len(BODY_PATTERNS)]
    if feature.motif_repeat_index > 0:
        return base_pattern
    if feature.intensity <= 0.35 and macro.persistence >= 0.65:
        return "breath"
    if feature.intensity >= 0.65 and feature.beat_count >= 6:
        return "heartbeat"
    if feature.novelty >= 0.5:
        return "stagger"
    return base_pattern


def build_accent_events(feature: StructureFeature, beat_pattern: str) -> tuple[Dict[str, Any], ...]:
    beat_count = max(1, feature.beat_count)
    if beat_count == 1:
        return ({"beat": 1, "action": "accent"},)

    events: List[Dict[str, Any]] = [{"beat": 1, "action": "accent"}]
    midpoint = max(2, min(beat_count - 1, round(beat_count * 0.5)))
    late = max(2, min(beat_count, round(beat_count * 0.75)))

    if beat_pattern == "heartbeat":
        events.append({"beat": midpoint, "action": "accent"})
        events.append({"beat": late, "action": "accent"})
    elif beat_pattern == "stagger":
        events.append({"beat": midpoint, "action": "accent"})
    elif beat_pattern == "glide":
        events.append({"beat": beat_count, "action": "settle"})
    else:
        events.append({"beat": beat_count, "action": "settle"})

    deduped: List[Dict[str, Any]] = []
    seen_beats = set()
    for event in events:
        beat = int(event["beat"])
        if beat in seen_beats:
            deduped[-1] = {"beat": beat, "action": event["action"]}
            continue
        seen_beats.add(beat)
        deduped.append({"beat": beat, "action": event["action"]})
    return tuple(deduped)


def plan_body_states(
    features: Sequence[StructureFeature],
    macro_states: Sequence[MacroState],
    config: Optional[Dict[str, Any]] = None,
) -> List[BodyState]:
    config = config or {}
    body_cfg = config.get("body", {})
    motif_cfg = config.get("motif", {})
    delta_bounds = body_cfg.get("boid_delta_bounds", {})
    preservation_strength = float(motif_cfg.get("preservation_strength", 0.72))

    states: List[BodyState] = []
    motif_patterns: Dict[int, str] = {}

    for feature, macro in zip(features, macro_states):
        template = stable_unit_values(feature.motif_signature, count=4)
        motif_seed = int(hashlib.sha1(feature.motif_signature.encode("utf-8")).hexdigest()[:8], 16)

        amplitude = clamp01(
            0.18
            + 0.55 * feature.intensity
            + 0.20 * (1.0 - macro.persistence)
        )
        if feature.motif_repeat_index > 0:
            amplitude *= 0.85 + 0.15 * preservation_strength

        beat_pattern = motif_patterns.get(feature.motif_family)
        if beat_pattern is None:
            beat_pattern = choose_beat_pattern(feature, macro, template)
            motif_patterns[feature.motif_family] = beat_pattern
        pulse_gain = clamp01(float(body_cfg.get("pulse_gain", 0.68)) * amplitude)
        breath_gain = clamp01(
            float(body_cfg.get("breath_gain", 0.48))
            * (0.35 + 0.65 * amplitude)
        )
        sway_gain = clamp01(
            float(body_cfg.get("sway_gain", 0.42))
            * (0.30 + 0.70 * amplitude)
        )
        swirl_gain = clamp01(
            float(body_cfg.get("swirl_gain", 0.46))
            * (0.25 + 0.75 * amplitude)
            * (1.0 if macro.rotation_mode != "none" else 0.55)
        )
        tangential_gain = clamp01(
            float(body_cfg.get("tangential_gain", 0.44))
            * (0.25 + 0.75 * amplitude)
        )
        tightness = clamp01(
            0.65
            - 0.25 * macro.radius_base
            - 0.10 * amplitude
        )
        accent_gain = clamp01(
            float(body_cfg.get("accent_gain", 0.62))
            * (0.35 + 0.65 * amplitude)
        )

        dz_pulse = float(body_cfg.get("dz_pulse_max", 0.10)) * amplitude * (0.55 + 0.45 * accent_gain)
        dr_pulse = float(body_cfg.get("dr_pulse_max", 0.18)) * amplitude * (0.45 + 0.55 * (1.0 - tightness))
        dtheta_pulse = float(body_cfg.get("dtheta_pulse_max", 0.65)) * amplitude * (0.35 + 0.65 * swirl_gain)

        variation_scale = 0.75 if feature.motif_repeat_index > 0 else 1.0

        separation_delta = bounded_delta(
            variation_scale * amplitude * (template[2] - 0.5) * 0.4,
            float(delta_bounds.get("separation", 0.35)),
        )
        alignment_delta = bounded_delta(
            variation_scale * (0.25 * (0.5 - tightness) + 0.10 * (template[0] - 0.5)),
            float(delta_bounds.get("alignment", 0.35)),
        )
        cohesion_delta = bounded_delta(
            variation_scale * (0.25 * (0.5 - macro.radius_base) + 0.10 * (template[3] - 0.5)),
            float(delta_bounds.get("cohesion", 0.35)),
        )
        goal_weight_delta = bounded_delta(
            variation_scale * (0.20 * macro.persistence - 0.10 * amplitude),
            float(delta_bounds.get("goal_weight", 0.25)),
        )
        jitter_delta = bounded_delta(
            variation_scale * amplitude * (0.35 - 0.25 * macro.persistence),
            float(delta_bounds.get("jitter", 0.35)),
        )

        accent_events = build_accent_events(feature, beat_pattern)

        states.append(
            BodyState(
                phrase_index=feature.phrase_index,
                motif_seed=motif_seed,
                beat_pattern=beat_pattern,
                body_amplitude=amplitude,
                pulse_gain=pulse_gain,
                breath_gain=breath_gain,
                sway_gain=sway_gain,
                swirl_gain=swirl_gain,
                tangential_gain=tangential_gain,
                tightness=tightness,
                accent_gain=accent_gain,
                dz_pulse=dz_pulse,
                dr_pulse=dr_pulse,
                dtheta_pulse=dtheta_pulse,
                separation_delta=separation_delta,
                alignment_delta=alignment_delta,
                cohesion_delta=cohesion_delta,
                goal_weight_delta=goal_weight_delta,
                jitter_delta=jitter_delta,
                accent_events=accent_events,
            )
        )

    return states
