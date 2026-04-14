"""Persistent attractor state for the hierarchical choreography pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - import style depends on entry point
    from .mind_planner import MacroState
    from .structure_features import StructureFeature, clamp01
except ImportError:  # pragma: no cover
    from mind_planner import MacroState  # type: ignore
    from structure_features import StructureFeature, clamp01  # type: ignore


@dataclass
class _AttractorState:
    center: np.ndarray
    radius: float
    rotation: float


@dataclass(frozen=True)
class AttractorFrame:
    phrase_index: int
    start_center: tuple[float, float, float]
    end_center: tuple[float, float, float]
    target_center: tuple[float, float, float]
    start_radius: float
    end_radius: float
    start_rotation: float
    end_rotation: float
    rotation_mode: str
    persistence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phrase_index": self.phrase_index,
            "start_center": {
                "x": self.start_center[0],
                "y": self.start_center[1],
                "z": self.start_center[2],
            },
            "end_center": {
                "x": self.end_center[0],
                "y": self.end_center[1],
                "z": self.end_center[2],
            },
            "target_center": {
                "x": self.target_center[0],
                "y": self.target_center[1],
                "z": self.target_center[2],
            },
            "start_radius": self.start_radius,
            "end_radius": self.end_radius,
            "start_rotation": self.start_rotation,
            "end_rotation": self.end_rotation,
            "rotation_mode": self.rotation_mode,
            "persistence": self.persistence,
        }


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_vec(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


def margin_clamp(value: float, margin: float) -> float:
    return float(np.clip(value, margin, 1.0 - margin))


def target_center_from_macro(
    macro: MacroState,
    config: Dict[str, Any],
) -> np.ndarray:
    attractor_cfg = config.get("attractor", {})
    margin = float(attractor_cfg.get("center_margin", 0.12))
    x = margin_clamp(0.5 + 0.28 * macro.translation_bias[0], margin)
    y = margin_clamp(
        0.5
        + 0.28 * (macro.audience_bias - 0.5)
        + 0.12 * macro.translation_bias[1],
        margin,
    )
    z = margin_clamp(macro.z_base, margin * 0.8)
    return np.array([x, y, z], dtype=float)


def build_attractor_frames(
    features: Sequence[StructureFeature],
    macro_states: Sequence[MacroState],
    config: Optional[Dict[str, Any]] = None,
) -> List[AttractorFrame]:
    config = config or {}
    attractor_cfg = config.get("attractor", {})
    base_interp = float(attractor_cfg.get("base_interp", 0.34))
    inertia_gain = float(attractor_cfg.get("inertia_gain", 0.48))
    rotation_speed = float(attractor_cfg.get("rotation_radians_per_second", 0.75))

    state = _AttractorState(
        center=np.array([0.5, 0.5, 0.5], dtype=float),
        radius=0.42,
        rotation=0.0,
    )
    frames: List[AttractorFrame] = []

    for feature, macro in zip(features, macro_states):
        start_center = state.center.copy()
        start_radius = state.radius
        start_rotation = state.rotation
        target_center = target_center_from_macro(macro, config)

        adapt = clamp01(
            base_interp
            + (1.0 - macro.persistence) * inertia_gain
        )
        end_center = lerp_vec(state.center, target_center, adapt)
        end_radius = lerp(state.radius, macro.radius_base, adapt)

        rotation_target = start_rotation
        if macro.rotation_mode != "none":
            rotation_target += (
                macro.rotation_bias
                * rotation_speed
                * max(feature.duration, 0.25)
            )
        end_rotation = lerp(start_rotation, rotation_target, min(1.0, adapt + 0.12))

        frames.append(
            AttractorFrame(
                phrase_index=feature.phrase_index,
                start_center=tuple(float(value) for value in start_center),
                end_center=tuple(float(value) for value in end_center),
                target_center=tuple(float(value) for value in target_center),
                start_radius=float(start_radius),
                end_radius=float(end_radius),
                start_rotation=float(start_rotation),
                end_rotation=float(end_rotation),
                rotation_mode=macro.rotation_mode,
                persistence=macro.persistence,
            )
        )

        state = _AttractorState(
            center=end_center,
            radius=float(end_radius),
            rotation=float(end_rotation),
        )

    return frames
