"""End-to-end hierarchical choreography planning.

The new pipeline keeps the conceptual split explicit:
- structure/progression features define what is changing
- the mind planner chooses persistent macro attractor intent
- the attractor controller smooths that intent through time
- the body planner adds bounded rhythmic modulation around the macro frame

Existing timing helpers are reused from ``phrase_generator`` and the current
CLAP/beat outputs remain the source material.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # pragma: no cover - import style depends on entry point
    from .attractor_state import AttractorFrame, build_attractor_frames
    from .body_modulation import BodyState, plan_body_states
    from .mind_planner import MacroState, plan_macro_states
    from .phrase_generator import load_json, load_sections
    from .structure_features import StructureFeature, build_structure_features
except ImportError:  # pragma: no cover
    from attractor_state import AttractorFrame, build_attractor_frames  # type: ignore
    from body_modulation import BodyState, plan_body_states  # type: ignore
    from mind_planner import MacroState, plan_macro_states  # type: ignore
    from phrase_generator import load_json, load_sections  # type: ignore
    from structure_features import StructureFeature, build_structure_features  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "json/hierarchical_plan.json"
DEFAULT_CONFIG_PATH = "config/hierarchical_config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "history": {
        "decay": 0.82,
    },
    "mind": {
        "macro_update_interval_beats": 8,
        "rotation_gain": 0.55,
    },
    "motif": {
        "preservation_strength": 0.72,
        "return_threshold": 0.18,
        "variation_threshold": 0.34,
        "rupture_threshold": 0.58,
    },
    "attractor": {
        "base_interp": 0.34,
        "inertia_gain": 0.48,
        "rotation_radians_per_second": 0.75,
        "center_margin": 0.12,
    },
    "body": {
        "pulse_gain": 0.68,
        "breath_gain": 0.48,
        "sway_gain": 0.42,
        "swirl_gain": 0.46,
        "tangential_gain": 0.44,
        "accent_gain": 0.62,
        "dz_pulse_max": 0.10,
        "dr_pulse_max": 0.18,
        "dtheta_pulse_max": 0.65,
        "boid_delta_bounds": {
            "separation": 0.35,
            "alignment": 0.35,
            "cohesion": 0.35,
            "goal_weight": 0.25,
            "jitter": 0.35,
        },
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_hierarchical_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config_path = PROJECT_ROOT / path
    if not config_path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)

    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return deep_merge(DEFAULT_CONFIG, loaded)


def build_phrase_payload(
    feature: StructureFeature,
    macro: MacroState,
    attractor: AttractorFrame,
    body: BodyState,
) -> Dict[str, Any]:
    return {
        "phrase_index": feature.phrase_index,
        "start": feature.start,
        "end": feature.end,
        "duration": feature.duration,
        "beat_count": feature.beat_count,
        "section_index": feature.section_index,
        "section_role": feature.section_role,
        "motif_signature": feature.motif_signature,
        "motif_family": feature.motif_family,
        "motif_repeat_index": feature.motif_repeat_index,
        "motion_mode": macro.motion_mode,
        "height_level": macro.z_base,
        "depth_level": macro.audience_bias,
        "speed_level": macro.energy,
        "vertical_trend": macro.vertical_trend,
        "transition_style": macro.transition_style,
        "beat_plan": list(body.accent_events),
        "feature_summary": feature.to_dict(),
        "macro_state": macro.to_dict(),
        "attractor_frame": attractor.to_dict(),
        "body_state": body.to_dict(),
    }


def build_output_payload(
    features: Sequence[StructureFeature],
    macro_states: Sequence[MacroState],
    attractor_frames: Sequence[AttractorFrame],
    body_states: Sequence[BodyState],
    source_file: str,
    config_path: str,
) -> Dict[str, Any]:
    phrases = [
        build_phrase_payload(feature, macro, attractor, body)
        for feature, macro, attractor, body in zip(
            features,
            macro_states,
            attractor_frames,
            body_states,
        )
    ]
    return {
        "source_file": source_file,
        "config_file": config_path,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "planner": "hierarchical_choreography",
        "total_phrases": len(phrases),
        "phrases": phrases,
    }


def generate_hierarchical_plan(
    clap_segments: Optional[Sequence[Dict[str, Any]]] = None,
    beat_times: Optional[Sequence[float]] = None,
    sections: Optional[Sequence[Any]] = None,
    anchor_weight_segments: Optional[Sequence[Dict[str, Any]]] = None,
    source_file: str = "json/clap_results.json",
    beat_times_path: str = "json/beat_times.json",
    sections_path: str = "json/sections.json",
    anchor_weights_path: str = "json/clap_weights.json",
    output_path: str = DEFAULT_OUTPUT,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> Dict[str, Any]:
    config = load_hierarchical_config(config_path)
    clap_segments = list(clap_segments or load_json(source_file, default=[]) or [])
    beat_times = list(beat_times or load_json(beat_times_path, default=[]) or [])
    sections = list(sections or load_sections(sections_path))
    anchor_weight_segments = list(
        anchor_weight_segments
        or load_json(anchor_weights_path, default=[]) or []
    )

    if not clap_segments:
        raise RuntimeError(f"No phrase data found in {source_file}.")

    features = build_structure_features(
        clap_segments,
        beat_times=beat_times,
        sections=sections,
        anchor_weight_segments=anchor_weight_segments,
        history_decay=float(config.get("history", {}).get("decay", 0.82)),
    )
    macro_states = plan_macro_states(features, config=config)
    attractor_frames = build_attractor_frames(features, macro_states, config=config)
    body_states = plan_body_states(features, macro_states, config=config)
    payload = build_output_payload(
        features,
        macro_states,
        attractor_frames,
        body_states,
        source_file=source_file,
        config_path=config_path,
    )

    output_file = PROJECT_ROOT / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="json/clap_results.json")
    parser.add_argument("--beat-times", default="json/beat_times.json")
    parser.add_argument("--sections", default="json/sections.json")
    parser.add_argument("--anchor-weights", default="json/clap_weights.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    payload = generate_hierarchical_plan(
        source_file=args.input,
        beat_times_path=args.beat_times,
        sections_path=args.sections,
        anchor_weights_path=args.anchor_weights,
        output_path=args.output,
        config_path=args.config,
    )
    print(
        f"Wrote {args.output} with {payload['total_phrases']} hierarchical phrase plans"
    )


if __name__ == "__main__":
    main()
