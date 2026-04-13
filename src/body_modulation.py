"""Low-level embodied modulation for the hierarchical choreography pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - import style depends on entry point
    from .mind_planner import MacroState
    from .phrase_generator import PhraseBlock, build_beat_plan
    from .structure_features import StructureFeature, clamp01
except ImportError:  # pragma: no cover
    from mind_planner import MacroState  # type: ignore
    from phrase_generator import PhraseBlock, build_beat_plan  # type: ignore
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
    if macro.stability >= 0.74 and feature.intensity <= 0.46:
        return "breath"
    if feature.intensity >= 0.62 and feature.beat_count >= 6:
        return "heartbeat"
    if feature.novelty >= 0.48:
        return "stagger"
    return BODY_PATTERNS[int(motif_template[0] * len(BODY_PATTERNS)) % len(BODY_PATTERNS)]


def compat_plan_for_body(feature: StructureFeature, macro: MacroState) -> Dict[str, Any]:
    return {
        "section_role": macro.section_role,
        "motion_mode": macro.motion_mode,
        "height_level": macro.z_base,
        "depth_level": macro.audience_bias,
        "speed_level": macro.energy,
        "vertical_trend": macro.vertical_trend,
        "transition_style": macro.transition_style,
    }


def make_phrase_block(feature: StructureFeature) -> PhraseBlock:
    return PhraseBlock(
        phrase_index=feature.phrase_index,
        start=feature.start,
        end=feature.end,
        beat_count=feature.beat_count,
        prompt_text="hierarchical body modulation",
        section_index=feature.section_index,
    )


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

    for feature, macro in zip(features, macro_states):
        template = stable_unit_values(feature.motif_signature, count=4)
        motif_seed = int(hashlib.sha1(feature.motif_signature.encode("utf-8")).hexdigest()[:8], 16)

        amplitude = clamp01(
            0.14
            + 0.52 * feature.intensity
            + 0.22 * feature.novelty
            - 0.42 * macro.stability
        )
        if macro.motif_mode in {"preserve", "return"}:
            amplitude *= 0.85 + 0.15 * preservation_strength
        elif macro.motif_mode == "rupture":
            amplitude = clamp01(amplitude + 0.12)

        beat_pattern = choose_beat_pattern(feature, macro, template)
        pulse_gain = clamp01(float(body_cfg.get("pulse_gain", 0.68)) * amplitude)
        breath_gain = clamp01(
            float(body_cfg.get("breath_gain", 0.48))
            * (0.35 + 0.65 * (macro.stability if beat_pattern == "breath" else amplitude))
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
            0.48
            + 0.28 * macro.stability
            - 0.18 * macro.radius_base
            + 0.10 * template[1]
            - 0.12 * feature.intensity
        )
        accent_gain = clamp01(
            float(body_cfg.get("accent_gain", 0.62))
            * (0.35 + 0.65 * feature.intensity)
        )

        dz_pulse = float(body_cfg.get("dz_pulse_max", 0.10)) * amplitude * (0.55 + 0.45 * accent_gain)
        dr_pulse = float(body_cfg.get("dr_pulse_max", 0.18)) * amplitude * (0.45 + 0.55 * (1.0 - tightness))
        dtheta_pulse = float(body_cfg.get("dtheta_pulse_max", 0.65)) * amplitude * (0.35 + 0.65 * swirl_gain)

        variation_scale = {
            "preserve": 0.55,
            "return": 0.70,
            "vary": 0.95,
            "rupture": 1.15,
        }.get(macro.motif_mode, 0.85)

        separation_delta = bounded_delta(
            variation_scale * (0.14 * feature.tension + 0.10 * template[2] - 0.08 * macro.radius_base),
            float(delta_bounds.get("separation", 0.35)),
        )
        alignment_delta = bounded_delta(
            variation_scale * (0.12 * (1.0 - tightness) + 0.06 * template[0] - 0.06 * feature.novelty),
            float(delta_bounds.get("alignment", 0.35)),
        )
        cohesion_delta = bounded_delta(
            variation_scale * (0.10 * macro.radius_base - 0.08 * feature.tension + 0.05 * template[3]),
            float(delta_bounds.get("cohesion", 0.35)),
        )
        goal_weight_delta = bounded_delta(
            variation_scale * (0.10 * macro.stability - 0.08 * feature.novelty),
            float(delta_bounds.get("goal_weight", 0.25)),
        )
        jitter_delta = bounded_delta(
            variation_scale * (0.16 * feature.novelty + 0.10 * feature.arousal - 0.14 * macro.stability),
            float(delta_bounds.get("jitter", 0.35)),
        )

        accent_events = tuple(
            build_beat_plan(
                make_phrase_block(feature),
                compat_plan_for_body(feature, macro),
            )
        )

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
