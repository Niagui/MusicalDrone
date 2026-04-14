"""Macro attractor planning with a minimal LLM-owned attractor schema.

The macro planner intentionally does not infer attractor motion from hand-tuned
emotion formulas. The only non-LLM behavior in this module is:
- prompt construction from precomputed structure/progression features
- schema validation and clamping
- neutral fallback defaults when LLM planning is unavailable by choice
- batching/retry so long songs do not depend on one giant JSON response
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # pragma: no cover - import style depends on entry point
    from .phrase_generator import OPENAI_API_KEY, create_chat_completion, parse_json_strict
    from .structure_features import StructureFeature, clamp01, clamp_signed
except ImportError:  # pragma: no cover
    from phrase_generator import (  # type: ignore
        OPENAI_API_KEY,
        create_chat_completion,
        parse_json_strict,
    )
    from structure_features import StructureFeature, clamp01, clamp_signed  # type: ignore


ROTATION_MODES = {"none", "orbit", "swirl"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LLM_SYSTEM_PROMPT = """You are a JSON planner for phrase-scale attractor control.
Plan only the macro attractor layer.

Use the difference between current emotion and decayed past emotion as the main
driver of change. Similar repeated motifs can keep related local behavior while
still getting different macro posture because the emotional state is different
now.

Keep it minimal. Only decide the attractor frame.

Return JSON only in this format:
{
  "phrases": [
    {
      "phrase_index": 0,
      "z_base": 0.0,
      "audience_bias": 0.0,
      "radius_base": 0.0,
      "rotation_mode": "none",
      "rotation_bias": 0.0,
      "translation_bias_x": 0.0,
      "translation_bias_y": 0.0,
      "persistence": 0.0
    }
  ]
}

Rules:
- numeric fields must stay within [0, 1], except rotation_bias and translation
  biases which must stay within [-1, 1]
- `rotation_mode` must be one of: none, orbit, swirl
- do not output beat plans, boid parameters, drone coordinates, or framewise
  commands
"""


@dataclass(frozen=True)
class MacroState:
    phrase_index: int
    z_base: float
    audience_bias: float
    radius_base: float
    rotation_mode: str
    rotation_bias: float
    translation_bias: tuple[float, float]
    persistence: float
    section_role: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["translation_bias"] = {
            "x": self.translation_bias[0],
            "y": self.translation_bias[1],
        }
        return payload


@dataclass(frozen=True)
class MacroPlanResult:
    states: tuple[MacroState, ...]
    planner_mode: str
    model: Optional[str]


def clamp_level(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return clamp01(numeric)


def clamp_signed_level(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return clamp_signed(numeric)


def neutral_macro_directive(feature: StructureFeature) -> Dict[str, Any]:
    return {
        "phrase_index": feature.phrase_index,
        "z_base": 0.5,
        "audience_bias": 0.5,
        "radius_base": 0.5,
        "rotation_mode": "none",
        "rotation_bias": 0.0,
        "translation_bias_x": 0.0,
        "translation_bias_y": 0.0,
        "persistence": 0.5,
    }


def directive_to_macro_state(
    feature: StructureFeature,
    directive: Dict[str, Any],
) -> MacroState:
    z_base = clamp_level(directive.get("z_base"), 0.5)
    audience_bias = clamp_level(directive.get("audience_bias"), 0.5)
    radius_base = clamp_level(directive.get("radius_base"), 0.5)
    rotation_mode = (
        directive.get("rotation_mode")
        if isinstance(directive.get("rotation_mode"), str)
        and directive.get("rotation_mode") in ROTATION_MODES
        else "none"
    )
    rotation_bias = clamp_signed_level(directive.get("rotation_bias"), 0.0)
    if rotation_mode == "none":
        rotation_bias = 0.0

    translation_bias = (
        clamp_signed_level(directive.get("translation_bias_x"), 0.0),
        clamp_signed_level(directive.get("translation_bias_y"), 0.0),
    )
    persistence = clamp_level(directive.get("persistence"), 0.5)

    return MacroState(
        phrase_index=feature.phrase_index,
        z_base=z_base,
        audience_bias=audience_bias,
        radius_base=radius_base,
        rotation_mode=rotation_mode,
        rotation_bias=rotation_bias,
        translation_bias=translation_bias,
        persistence=persistence,
        section_role=feature.section_role,
    )


def build_llm_prompt(features: Sequence[StructureFeature]) -> str:
    lines = [
        f"song_phrases: {len(features)}",
        "Fields are [valence, arousal, tension].",
        "",
    ]

    for feature in features:
        current = [round(value, 3) for value in feature.current_emotion_vector[:3]]
        history = [round(value, 3) for value in feature.decayed_history_vector[:3]]
        diff = [round(value, 3) for value in feature.difference_vector[:3]]
        lines.extend(
            [
                f"phrase_index: {feature.phrase_index}",
                f"repeat_index: {feature.motif_repeat_index}",
                f"current: {current}",
                f"decayed_history: {history}",
                f"delta: {diff}",
                f"normalized_change: {feature.normalized_change:.3f}",
                "",
            ]
        )

    return "\n".join(lines).strip()


def normalize_llm_directives(
    obj: Dict[str, Any],
    default_directives: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    directives = {
        phrase_index: dict(directive)
        for phrase_index, directive in default_directives.items()
    }

    for record in obj.get("phrases", []):
        if not isinstance(record, dict):
            continue

        phrase_index = record.get("phrase_index")
        if not isinstance(phrase_index, int) or phrase_index not in directives:
            continue

        directives[phrase_index] = {
            "phrase_index": phrase_index,
            "z_base": clamp_level(record.get("z_base"), directives[phrase_index]["z_base"]),
            "audience_bias": clamp_level(
                record.get("audience_bias"),
                directives[phrase_index]["audience_bias"],
            ),
            "radius_base": clamp_level(
                record.get("radius_base"),
                directives[phrase_index]["radius_base"],
            ),
            "rotation_mode": (
                record.get("rotation_mode")
                if isinstance(record.get("rotation_mode"), str)
                and record.get("rotation_mode") in ROTATION_MODES
                else directives[phrase_index]["rotation_mode"]
            ),
            "rotation_bias": clamp_signed_level(
                record.get("rotation_bias"),
                directives[phrase_index]["rotation_bias"],
            ),
            "translation_bias_x": clamp_signed_level(
                record.get("translation_bias_x"),
                directives[phrase_index]["translation_bias_x"],
            ),
            "translation_bias_y": clamp_signed_level(
                record.get("translation_bias_y"),
                directives[phrase_index]["translation_bias_y"],
            ),
            "persistence": clamp_level(
                record.get("persistence"),
                directives[phrase_index]["persistence"],
            ),
        }

    return directives


def call_llm_macro_planner(
    features: Sequence[StructureFeature],
    default_directives: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    mind_cfg = config.get("mind", {})
    model = str(mind_cfg.get("llm_model", "gpt-4o-mini"))
    temperature = float(mind_cfg.get("llm_temperature", 0.2))
    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": build_llm_prompt(features)},
        ],
        "temperature": temperature,
        "max_tokens": 6000,
        "response_format": {"type": "json_object"},
    }
    raw = create_chat_completion(request_kwargs)
    try:
        parsed = parse_json_strict(raw)
    except Exception as err:
        debug_path = PROJECT_ROOT / "json" / "hierarchical_macro_last_response.txt"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as handle:
            handle.write(raw)
        raise RuntimeError(
            f"Failed to parse macro planner JSON. Saved raw response to {debug_path}."
        ) from err
    return normalize_llm_directives(parsed, default_directives)


def batched_feature_groups(
    features: Sequence[StructureFeature],
    max_batch_phrases: int,
) -> List[Sequence[StructureFeature]]:
    if max_batch_phrases <= 0:
        max_batch_phrases = len(features) or 1
    return [
        features[index : index + max_batch_phrases]
        for index in range(0, len(features), max_batch_phrases)
    ]


def plan_llm_batch_with_fallback(
    features: Sequence[StructureFeature],
    default_directives: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    try:
        return call_llm_macro_planner(features, default_directives, config)
    except Exception:
        if len(features) <= 1:
            raise

        midpoint = len(features) // 2
        left_features = features[:midpoint]
        right_features = features[midpoint:]

        left_defaults = {
            feature.phrase_index: default_directives[feature.phrase_index]
            for feature in left_features
        }
        right_defaults = {
            feature.phrase_index: default_directives[feature.phrase_index]
            for feature in right_features
        }

        left = plan_llm_batch_with_fallback(left_features, left_defaults, config)
        right = plan_llm_batch_with_fallback(right_features, right_defaults, config)
        return {**left, **right}


def plan_llm_directives(
    features: Sequence[StructureFeature],
    default_directives: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    max_batch_phrases = int(config.get("mind", {}).get("max_batch_phrases", 12))
    directives = {
        phrase_index: dict(directive)
        for phrase_index, directive in default_directives.items()
    }

    for group in batched_feature_groups(features, max_batch_phrases):
        group_defaults = {
            feature.phrase_index: directives[feature.phrase_index]
            for feature in group
        }
        directives.update(plan_llm_batch_with_fallback(group, group_defaults, config))

    return directives


def resolve_planner_mode(config: Dict[str, Any]) -> str:
    requested = str(config.get("mind", {}).get("planner_mode", "llm")).lower()
    if requested not in {"llm", "neutral", "auto", "heuristic"}:
        requested = "llm"

    if requested == "heuristic":
        return "neutral"

    if requested == "auto":
        return "llm" if OPENAI_API_KEY else "neutral"
    return requested


def plan_macro_states(
    features: Sequence[StructureFeature],
    config: Optional[Dict[str, Any]] = None,
) -> MacroPlanResult:
    config = config or {}
    default_directives = {
        feature.phrase_index: neutral_macro_directive(feature)
        for feature in features
    }

    resolved_mode = resolve_planner_mode(config)
    mind_cfg = config.get("mind", {})
    fallback_to_neutral = bool(
        mind_cfg.get(
            "fallback_to_neutral",
            mind_cfg.get("fallback_to_heuristic", False),
        )
    )
    directives = default_directives
    planner_mode = "neutral"
    model = None

    if resolved_mode == "llm":
        model = str(mind_cfg.get("llm_model", "gpt-4o-mini"))
        try:
            directives = plan_llm_directives(features, default_directives, config)
            planner_mode = "llm"
        except Exception:
            if not fallback_to_neutral:
                raise
            planner_mode = "neutral_fallback"
    elif resolved_mode == "neutral":
        planner_mode = "neutral"

    states: List[MacroState] = []
    for feature in features:
        macro = directive_to_macro_state(feature, directives[feature.phrase_index])
        states.append(macro)

    return MacroPlanResult(states=tuple(states), planner_mode=planner_mode, model=model)
