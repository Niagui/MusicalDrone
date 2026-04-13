"""Deterministic macro planning for the hierarchical choreography pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - import style depends on entry point
    from .structure_features import StructureFeature, clamp01, clamp_signed
except ImportError:  # pragma: no cover
    from structure_features import StructureFeature, clamp01, clamp_signed  # type: ignore


ROTATION_MODES = {"none", "orbit", "swirl"}
MOTIF_MODES = {"preserve", "vary", "rupture", "return"}
TRANSITION_STYLES = {"smooth", "drift", "surge", "snap"}
MOTION_MODES = {"hold", "advance", "retreat", "sweep_left", "sweep_right"}


@dataclass(frozen=True)
class MacroState:
    phrase_index: int
    z_base: float
    audience_bias: float
    radius_base: float
    rotation_mode: str
    rotation_bias: float
    translation_bias: tuple[float, float]
    stability: float
    persistence: float
    motif_mode: str
    energy: float
    openness: float
    section_role: str
    motion_mode: str
    vertical_trend: str
    transition_style: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["translation_bias"] = {
            "x": self.translation_bias[0],
            "y": self.translation_bias[1],
        }
        return payload


def stable_sign(seed_text: str) -> float:
    digest = hashlib.sha1(seed_text.encode("utf-8")).digest()
    return -1.0 if digest[0] % 2 else 1.0


def history_strength(feature: StructureFeature) -> float:
    history = np.asarray(feature.decayed_history_vector[:3], dtype=float)
    if history.size == 0:
        return 0.0
    return clamp01(float(np.linalg.norm(history) / np.sqrt(max(1, history.size))))


def pick_motif_mode(
    feature: StructureFeature,
    previous_feature: Optional[StructureFeature],
    config: Dict[str, Any],
) -> str:
    motif_cfg = config.get("motif", {})
    return_threshold = float(motif_cfg.get("return_threshold", 0.18))
    variation_threshold = float(motif_cfg.get("variation_threshold", 0.34))
    rupture_threshold = float(motif_cfg.get("rupture_threshold", 0.58))

    if feature.motif_repeat_index > 0:
        if feature.novelty <= return_threshold:
            return "return"
        return "vary"
    if feature.novelty >= rupture_threshold and feature.emotion_change_magnitude >= 0.32:
        return "rupture"
    if feature.novelty >= variation_threshold:
        return "vary"
    if previous_feature and previous_feature.section_role == "transition" and feature.novelty <= return_threshold:
        return "return"
    return "preserve"


def pick_rotation_mode(
    energy: float,
    stability: float,
    novelty: float,
    motif_mode: str,
) -> str:
    if stability >= 0.72:
        return "none"
    if energy >= 0.72 and novelty >= 0.20:
        return "orbit"
    if motif_mode in {"vary", "rupture"} and novelty >= 0.28:
        return "swirl"
    if energy >= 0.60:
        return "orbit"
    return "none"


def motion_mode_from_bias(translation_bias: tuple[float, float]) -> str:
    x_bias, y_bias = translation_bias
    if abs(y_bias) >= abs(x_bias) and abs(y_bias) >= 0.18:
        return "advance" if y_bias >= 0.0 else "retreat"
    if abs(x_bias) >= 0.18:
        return "sweep_right" if x_bias >= 0.0 else "sweep_left"
    return "hold"


def vertical_trend_from_levels(
    current_z: float,
    previous_z: Optional[float],
) -> str:
    if previous_z is None:
        return "hold"
    delta = current_z - previous_z
    if delta >= 0.06:
        return "rise"
    if delta <= -0.06:
        return "fall"
    return "hold"


def transition_style_from_macro(
    feature: StructureFeature,
    motif_mode: str,
    stability: float,
) -> str:
    if motif_mode == "rupture":
        return "snap" if feature.novelty >= 0.65 else "surge"
    if feature.section_role in {"buildup", "peak"}:
        return "surge"
    if feature.section_role == "transition":
        return "snap"
    if stability >= 0.68:
        return "drift"
    return "smooth"


def plan_macro_state(
    feature: StructureFeature,
    previous_feature: Optional[StructureFeature],
    previous_macro: Optional[MacroState],
    config: Dict[str, Any],
) -> MacroState:
    diff = np.asarray(feature.difference_vector[:3], dtype=float)
    diff_valence = float(diff[0]) if diff.size > 0 else 0.0
    diff_arousal = float(diff[1]) if diff.size > 1 else 0.0
    diff_tension = float(diff[2]) if diff.size > 2 else 0.0
    delta_strength = feature.strength - history_strength(feature)

    openness = clamp01(
        0.50
        + 0.28 * feature.valence
        - 0.16 * feature.tension
        + 0.18 * diff_valence
    )
    energy = clamp01(
        0.20
        + 0.42 * feature.intensity
        + 0.24 * max(feature.arousal, 0.0)
        + 0.14 * feature.novelty
    )
    motif_mode = pick_motif_mode(feature, previous_feature, config)

    role_z_nudge = {
        "peak": 0.10,
        "buildup": 0.05,
        "release": -0.03,
        "transition": 0.02,
        "stable": 0.0,
    }
    role_radius_nudge = {
        "peak": 0.08,
        "buildup": 0.05,
        "release": -0.05,
        "transition": 0.00,
        "stable": 0.0,
    }

    z_base = clamp01(
        0.45
        + 0.18 * feature.arousal
        + 0.14 * energy
        + 0.12 * max(diff_arousal, 0.0)
        + role_z_nudge.get(feature.section_role, 0.0)
    )
    audience_bias = clamp01(
        0.48
        + 0.18 * feature.valence
        + 0.12 * openness
        + 0.10 * max(delta_strength, 0.0)
        - 0.08 * max(diff_tension, 0.0)
    )
    radius_base = clamp01(
        0.38
        + 0.24 * openness
        + 0.14 * feature.novelty
        - 0.10 * feature.tension
        + role_radius_nudge.get(feature.section_role, 0.0)
    )

    stability = clamp01(
        0.74
        - 0.28 * energy
        - 0.24 * feature.novelty
        + 0.10 * (motif_mode == "preserve")
        + 0.08 * (feature.section_role == "release")
    )
    persistence = clamp01(
        0.50
        + 0.28 * (1.0 - feature.novelty)
        + 0.14 * (motif_mode in {"preserve", "return"})
        - 0.14 * (motif_mode == "rupture")
    )

    lateral_sign = stable_sign(feature.motif_signature)
    translation_x = clamp_signed(
        lateral_sign * (0.12 + 0.30 * feature.novelty + 0.08 * abs(feature.valence))
    )
    translation_y = clamp_signed(
        (audience_bias - 0.5) * 1.35
        + 0.18 * max(diff_valence, 0.0)
        - 0.12 * (feature.section_role == "release")
    )
    translation_bias = (translation_x, translation_y)

    rotation_mode = pick_rotation_mode(energy, stability, feature.novelty, motif_mode)
    rotation_gain = float(config.get("mind", {}).get("rotation_gain", 0.55))
    rotation_bias = 0.0
    if rotation_mode != "none":
        rotation_bias = clamp_signed(
            lateral_sign * rotation_gain * (0.40 + 0.60 * energy + 0.20 * feature.novelty)
        )

    motion_mode = motion_mode_from_bias(translation_bias)
    previous_z = previous_macro.z_base if previous_macro else None
    vertical_trend = vertical_trend_from_levels(z_base, previous_z)
    transition_style = transition_style_from_macro(feature, motif_mode, stability)

    return MacroState(
        phrase_index=feature.phrase_index,
        z_base=z_base,
        audience_bias=audience_bias,
        radius_base=radius_base,
        rotation_mode=rotation_mode,
        rotation_bias=rotation_bias,
        translation_bias=translation_bias,
        stability=stability,
        persistence=persistence,
        motif_mode=motif_mode,
        energy=energy,
        openness=openness,
        section_role=feature.section_role,
        motion_mode=motion_mode,
        vertical_trend=vertical_trend,
        transition_style=transition_style,
    )


def plan_macro_states(
    features: Sequence[StructureFeature],
    config: Optional[Dict[str, Any]] = None,
) -> List[MacroState]:
    config = config or {}
    planned: List[MacroState] = []
    previous_feature: Optional[StructureFeature] = None
    previous_macro: Optional[MacroState] = None

    for feature in features:
        macro = plan_macro_state(feature, previous_feature, previous_macro, config)
        planned.append(macro)
        previous_feature = feature
        previous_macro = macro

    return planned
