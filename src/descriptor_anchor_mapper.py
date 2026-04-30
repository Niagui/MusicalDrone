#!/usr/bin/env python3
"""Descriptor-anchor mapping for CLAP-driven motion weights.

This module keeps CLAP in its strongest role: matching audio against concrete
musical descriptors. User-facing anchors can be arbitrary emotion words, but the
output remains the seven motion anchors consumed by the simulator.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency

    def load_dotenv(*_args, **_kwargs):
        return False


try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENAI_CHAT_COMPLETIONS_URL = os.getenv(
    "OPENAI_CHAT_COMPLETIONS_URL", "https://api.openai.com/v1/chat/completions"
)
MODEL_NAME = os.getenv("OPENAI_DESCRIPTOR_ANCHOR_MODEL", "gpt-4o-mini")
MOTION_ANCHORS = ["happy", "sad", "sleepy", "brave", "grumpy", "scared", "shy"]
DEFAULT_SMOOTHING_ALPHA = 0.25
PROMPTS_PER_ANCHOR = 5
RETRIES = 3
BACKOFF_BASE_S = 0.8


DEFAULT_PROMPTS = {
    "happy": [
        "bright joyful music",
        "playful upbeat melody",
        "warm cheerful musical atmosphere",
        "light optimistic rhythm",
        "sunny positive instrumental music",
    ],
    "sad": [
        "melancholic soft music",
        "wistful lyrical melody",
        "low-energy sorrowful atmosphere",
        "tender reflective piano music",
        "gentle minor-key sadness",
    ],
    "sleepy": [
        "slow calm dreamy music",
        "soft floating musical texture",
        "quiet low-arousal atmosphere",
        "gentle sparse lullaby-like music",
        "relaxed drifting instrumental passage",
    ],
    "brave": [
        "bold confident music",
        "heroic rising melody",
        "strong determined rhythm",
        "majestic courageous atmosphere",
        "forward-moving triumphant instrumental music",
    ],
    "grumpy": [
        "harsh irritated music",
        "abrasive aggressive rhythm",
        "rough tense musical attack",
        "angry heavy dissonant music",
        "snappy frustrated musical gestures",
    ],
    "scared": [
        "anxious suspenseful music",
        "eerie uncertain atmosphere",
        "nervous high-tension texture",
        "ominous fearful instrumental music",
        "startled uneasy musical passage",
    ],
    "shy": [
        "quiet delicate music",
        "soft restrained melody",
        "fragile intimate musical atmosphere",
        "gentle hesitant instrumental passage",
        "small tender sparse piano music",
    ],
}


KEYWORD_MIXES: list[tuple[set[str], dict[str, float]]] = [
    (
        {"melancholy", "melancholic", "wistful", "lonely", "sorrow", "sorrowful"},
        {"sad": 0.6, "sleepy": 0.25, "shy": 0.15},
    ),
    (
        {"soft", "calm", "dreamy", "floating", "gentle", "lullaby", "relaxed"},
        {"sleepy": 0.55, "shy": 0.25, "sad": 0.2},
    ),
    (
        {"fragile", "delicate", "quiet", "restrained", "intimate", "tender"},
        {"shy": 0.45, "sleepy": 0.3, "sad": 0.25},
    ),
    (
        {"bright", "joyful", "playful", "cheerful", "optimistic", "euphoric"},
        {"happy": 0.75, "brave": 0.15, "shy": 0.1},
    ),
    (
        {"bold", "heroic", "majestic", "triumphant", "confident", "determined"},
        {"brave": 0.65, "happy": 0.25, "grumpy": 0.1},
    ),
    (
        {"harsh", "abrasive", "aggressive", "angry", "irritated", "frustrated"},
        {"grumpy": 0.7, "brave": 0.2, "scared": 0.1},
    ),
    (
        {"anxious", "eerie", "ominous", "fearful", "haunted", "nervous"},
        {"scared": 0.65, "sad": 0.2, "sleepy": 0.15},
    ),
    (
        {"chaotic", "frantic", "panic", "panicked", "unstable"},
        {"scared": 0.45, "grumpy": 0.35, "brave": 0.2},
    ),
]


@dataclass
class RawAnchorConfig:
    name: str
    description: str
    prompts: list[str]
    motion_mix: dict[str, float] | None


@dataclass
class ResolvedAnchor:
    name: str
    description: str
    prompts: list[str]
    motion_mix: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "prompts": self.prompts,
            "motion_mix": self.motion_mix,
        }


def _load_repo_env() -> None:
    env_path = PROJECT_ROOT / ".env"

    try:
        load_dotenv(dotenv_path=env_path, override=False)
    except TypeError:
        load_dotenv()

    if os.getenv("OPENAI_API_KEY") or not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def resolve_config_path(path: str | Path) -> Path:
    config_path = Path(path).expanduser()
    if config_path.is_absolute():
        return config_path
    if config_path.exists():
        return config_path.resolve()
    return (PROJECT_ROOT / config_path).resolve()


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def clamp_smoothing_alpha(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = DEFAULT_SMOOTHING_ALPHA
    return max(0.0, min(1.0, numeric))


def normalize_motion_mix(motion_mix: dict[str, Any]) -> dict[str, float]:
    if not isinstance(motion_mix, dict) or not motion_mix:
        raise ValueError("motion_mix must be a non-empty object.")

    weights = {anchor: 0.0 for anchor in MOTION_ANCHORS}
    for key, value in motion_mix.items():
        if key not in weights:
            raise ValueError(f"Unknown motion anchor '{key}' in motion_mix.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as err:
            raise ValueError(f"motion_mix value for '{key}' must be numeric.") from err
        if numeric < 0:
            raise ValueError(f"motion_mix value for '{key}' must be non-negative.")
        weights[key] = numeric

    total = sum(weights.values())
    if total <= np.finfo(float).eps:
        raise ValueError("motion_mix must contain at least one positive value.")
    return {key: value / total for key, value in weights.items()}


def _one_hot_motion_mix(anchor: str) -> dict[str, float]:
    return normalize_motion_mix({anchor: 1.0})


def fallback_prompts(name: str, description: str = "") -> list[str]:
    key = name.strip().lower()
    if key in DEFAULT_PROMPTS:
        return list(DEFAULT_PROMPTS[key])

    prompts = [
        f"{name} music",
        f"music that feels {name}",
        f"{name} musical atmosphere",
        f"{name} expressive instrumental passage",
    ]
    if description:
        prompts.extend(
            [
                f"music with {description}",
                f"{name} music with {description}",
            ]
        )
    return _dedupe_strings(prompts)[:PROMPTS_PER_ANCHOR]


def fallback_motion_mix(name: str, description: str = "") -> dict[str, float]:
    key = name.strip().lower()
    if key in MOTION_ANCHORS:
        return _one_hot_motion_mix(key)

    text = f"{key} {description.lower()}"
    mixed = {anchor: 0.0 for anchor in MOTION_ANCHORS}
    matched = False
    for keywords, motion_mix in KEYWORD_MIXES:
        if not any(keyword in text for keyword in keywords):
            continue
        matched = True
        for anchor, value in motion_mix.items():
            mixed[anchor] += value

    if matched:
        return normalize_motion_mix(mixed)

    return normalize_motion_mix(
        {"sleepy": 0.3, "shy": 0.25, "sad": 0.2, "happy": 0.15, "brave": 0.1}
    )


def load_anchor_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_config_path(path)
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Descriptor anchor config must be a JSON object.")
    if int(payload.get("version", 1)) != 1:
        raise ValueError("Only descriptor anchor config version 1 is supported.")
    if not isinstance(payload.get("anchors"), list) or not payload["anchors"]:
        raise ValueError("Descriptor anchor config must contain a non-empty anchors list.")
    return payload


def parse_raw_anchors(config: dict[str, Any]) -> list[RawAnchorConfig]:
    raw_anchors = []
    seen = set()
    for index, item in enumerate(config.get("anchors", [])):
        if not isinstance(item, dict):
            raise ValueError(f"anchors[{index}] must be an object.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"anchors[{index}].name must be a non-empty string.")
        key = name.lower()
        if key in seen:
            raise ValueError(f"Duplicate descriptor anchor name '{name}'.")
        seen.add(key)

        prompts = item.get("prompts", [])
        if prompts is None:
            prompts = []
        if not isinstance(prompts, list):
            raise ValueError(f"anchors[{index}].prompts must be a list when provided.")

        motion_mix = item.get("motion_mix")
        normalized_mix = None
        if motion_mix is not None:
            normalized_mix = normalize_motion_mix(motion_mix)

        raw_anchors.append(
            RawAnchorConfig(
                name=name,
                description=str(item.get("description", "")).strip(),
                prompts=_dedupe_strings(prompts),
                motion_mix=normalized_mix,
            )
        )
    return raw_anchors


def _strip_code_fences(raw: str) -> str:
    fence = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE)
    text = raw.strip()
    match = fence.match(text)
    return match.group(1) if match else text


def _balanced_brace_slice(raw: str) -> str | None:
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    last_end = None
    for index in range(start, len(raw)):
        char = raw[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                last_end = index
    if last_end is None:
        return None
    return raw[start : last_end + 1]


def parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    for candidate in (text, _strip_code_fences(text), _balanced_brace_slice(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return json.loads(text)


def _missing_llm_fields(anchor: RawAnchorConfig) -> bool:
    return not anchor.prompts or anchor.motion_mix is None


def _llm_enabled() -> bool:
    _load_repo_env()
    return bool(os.getenv("OPENAI_API_KEY"))


def _create_chat_completion(request_kwargs: dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if OpenAI is not None:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content or ""

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    payload = {
        name: value
        for name, value in request_kwargs.items()
        if name in {"model", "messages", "temperature", "max_tokens", "response_format"}
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=120) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI request failed ({err.code}): {detail[:500]}") from err
    except urllib_error.URLError as err:
        raise RuntimeError(f"OpenAI request failed: {err}") from err

    choices = response_json.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI response did not contain choices.")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def expand_anchors_with_llm(
    anchors: list[RawAnchorConfig],
    *,
    model: str = MODEL_NAME,
) -> dict[str, dict[str, Any]]:
    missing = [anchor for anchor in anchors if _missing_llm_fields(anchor)]
    if not missing:
        return {}

    prompt_payload = [
        {
            "name": anchor.name,
            "description": anchor.description,
            "existing_prompts": anchor.prompts,
            "existing_motion_mix": anchor.motion_mix,
        }
        for anchor in missing
    ]
    system_prompt = f"""You are a JSON generator for music-to-motion anchors.
Return strictly valid JSON in this format:
{{
  "anchors": [
    {{
      "name": "string",
      "prompts": ["string", "string", "string", "string", "string"],
      "motion_mix": {{
        "happy": number,
        "sad": number,
        "sleepy": number,
        "brave": number,
        "grumpy": number,
        "scared": number,
        "shy": number
      }}
    }}
  ]
}}
Prompts must be concrete musical/audio descriptors, not just emotion labels.
Motion mix values must be non-negative and should sum to 1.
Use grumpy only for harsh, aggressive, abrasive, irritated, or angry motion.
Return JSON only.
"""
    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Resolve these descriptor anchors:\n"
                + json.dumps(prompt_payload, indent=2),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1600,
        "response_format": {"type": "json_object"},
    }

    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            raw = _create_chat_completion(request_kwargs)
            obj = parse_json_response(raw)
            resolved: dict[str, dict[str, Any]] = {}
            for item in obj.get("anchors", []):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                prompts = _dedupe_strings(item.get("prompts", []))
                mix = item.get("motion_mix")
                if prompts or isinstance(mix, dict):
                    resolved[name.lower()] = {"prompts": prompts, "motion_mix": mix}
            return resolved
        except Exception as err:
            last_error = err
            print(f"[warn] descriptor anchor expansion failed ({attempt}/{RETRIES}): {err}")
            time.sleep(BACKOFF_BASE_S * attempt)

    print(f"[warn] descriptor anchor expansion fell back to deterministic mode: {last_error}")
    return {}


def infer_motion_mix_with_clap(anchor: RawAnchorConfig, clap: Any) -> dict[str, float]:
    if clap is None:
        return fallback_motion_mix(anchor.name, anchor.description)

    texts = [anchor.name]
    if anchor.description:
        texts.extend([anchor.description, f"{anchor.name}: {anchor.description}"])
    texts.extend(anchor.prompts[:2] or fallback_prompts(anchor.name, anchor.description)[:2])

    try:
        emb = clap.get_text_embedding(_dedupe_strings(texts)).numpy()
        weights = clap.classify_new_emotion(emb, k=min(3, len(MOTION_ANCHORS)))
        keys = list(clap.anchor_labels_emb.keys())
        averaged = np.asarray(weights, dtype=float).mean(axis=0)
        mix = {key: float(value) for key, value in zip(keys, averaged) if key in MOTION_ANCHORS}
        return normalize_motion_mix(mix)
    except Exception as err:
        print(f"[warn] CLAP motion mix inference failed for '{anchor.name}': {err}")
        return fallback_motion_mix(anchor.name, anchor.description)


def resolve_anchor_records(
    config_path: str | Path,
    *,
    clap: Any = None,
    use_llm: bool = True,
) -> tuple[list[ResolvedAnchor], dict[str, Any]]:
    config_file = resolve_config_path(config_path)
    config = load_anchor_config(config_file)
    raw_anchors = parse_raw_anchors(config)
    smoothing_alpha = clamp_smoothing_alpha(
        config.get("smoothing_alpha", DEFAULT_SMOOTHING_ALPHA)
    )

    llm_records: dict[str, dict[str, Any]] = {}
    if use_llm and _llm_enabled():
        llm_records = expand_anchors_with_llm(raw_anchors)

    resolved = []
    for raw_anchor in raw_anchors:
        llm_record = llm_records.get(raw_anchor.name.lower(), {})

        prompts = raw_anchor.prompts
        if not prompts:
            prompts = _dedupe_strings(llm_record.get("prompts", []))
        if not prompts:
            prompts = fallback_prompts(raw_anchor.name, raw_anchor.description)

        motion_mix = raw_anchor.motion_mix
        if motion_mix is None and isinstance(llm_record.get("motion_mix"), dict):
            try:
                motion_mix = normalize_motion_mix(llm_record["motion_mix"])
            except ValueError as err:
                print(f"[warn] Ignoring invalid LLM motion_mix for '{raw_anchor.name}': {err}")
        if motion_mix is None:
            enriched = RawAnchorConfig(
                name=raw_anchor.name,
                description=raw_anchor.description,
                prompts=prompts,
                motion_mix=None,
            )
            motion_mix = infer_motion_mix_with_clap(enriched, clap)

        resolved.append(
            ResolvedAnchor(
                name=raw_anchor.name,
                description=raw_anchor.description,
                prompts=prompts[:PROMPTS_PER_ANCHOR],
                motion_mix=motion_mix,
            )
        )

    metadata = {
        "version": 1,
        "source_config": str(config_file),
        "model": MODEL_NAME if llm_records else "deterministic-fallback",
        "smoothing_alpha": smoothing_alpha,
        "motion_anchors": MOTION_ANCHORS,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "anchors": [anchor.to_json() for anchor in resolved],
    }
    return resolved, metadata


def unique_prompt_labels(anchors: list[ResolvedAnchor]) -> list[str]:
    prompts = []
    for anchor in anchors:
        prompts.extend(anchor.prompts)
    return _dedupe_strings(prompts)


def aggregate_anchor_scores(
    prompt_scores: dict[str, float], anchors: list[ResolvedAnchor]
) -> dict[str, float]:
    scores = {}
    for anchor in anchors:
        values = [float(prompt_scores.get(prompt, 0.0)) for prompt in anchor.prompts]
        scores[anchor.name] = float(np.mean(values)) if values else 0.0

    total = sum(scores.values())
    if total <= np.finfo(float).eps:
        uniform = 1.0 / max(1, len(anchors))
        return {anchor.name: uniform for anchor in anchors}
    return {name: value / total for name, value in scores.items()}


def motion_weights_from_anchor_scores(
    anchor_scores: dict[str, float], anchors: list[ResolvedAnchor]
) -> list[float]:
    weights = {anchor: 0.0 for anchor in MOTION_ANCHORS}
    for anchor in anchors:
        anchor_score = float(anchor_scores.get(anchor.name, 0.0))
        for motion_anchor, mix_weight in anchor.motion_mix.items():
            weights[motion_anchor] += anchor_score * float(mix_weight)

    total = sum(weights.values())
    if total <= np.finfo(float).eps:
        return [0.0 for _ in MOTION_ANCHORS]
    return [float(weights[anchor] / total) for anchor in MOTION_ANCHORS]


def smooth_weight_rows(rows: list[list[float]], alpha: float) -> list[list[float]]:
    if not rows:
        return []

    carry = clamp_smoothing_alpha(alpha)
    smoothed = []
    previous = np.asarray(rows[0], dtype=float)
    previous = previous / (previous.sum() + np.finfo(float).eps)
    smoothed.append([float(value) for value in previous])

    for row in rows[1:]:
        current = np.asarray(row, dtype=float)
        current = current / (current.sum() + np.finfo(float).eps)
        previous = (1.0 - carry) * current + carry * previous
        previous = previous / (previous.sum() + np.finfo(float).eps)
        smoothed.append([float(value) for value in previous])
    return smoothed


def score_descriptor_audio(
    clap: Any,
    audio: str | Path,
    anchors: list[ResolvedAnchor],
    *,
    time_base: list[list[float]] | None = None,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    sr: int = 22050,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import librosa

    if time_base is None:
        time_base = clap.k_beats_segments

    prompts = unique_prompt_labels(anchors)
    if not prompts:
        raise ValueError("Descriptor anchor mode requires at least one prompt.")

    y, sr = librosa.load(audio, sr=sr)
    raw_results = []
    raw_weight_rows = []

    for start, end in time_base:
        chunk = y[int(round(start * sr)) : int(round(end * sr))]
        predictions = clap.audio_classifier(chunk, candidate_labels=prompts)
        prompt_scores = {
            str(item["label"]): float(item["score"])
            for item in predictions
            if isinstance(item, dict) and "label" in item and "score" in item
        }
        anchor_scores = aggregate_anchor_scores(prompt_scores, anchors)
        weights = motion_weights_from_anchor_scores(anchor_scores, anchors)
        raw_weight_rows.append(weights)
        raw_results.append(
            {
                "start": float(start),
                "end": float(end),
                "prompt_scores": prompt_scores,
                "anchor_scores": anchor_scores,
                "raw_weights": weights,
            }
        )

    smoothed_rows = smooth_weight_rows(raw_weight_rows, smoothing_alpha)
    clap_weights = []
    for index, weights in enumerate(smoothed_rows):
        raw_results[index]["weights"] = weights
        clap_weights.append(
            {
                "start": raw_results[index]["start"],
                "end": raw_results[index]["end"],
                "weights": weights,
            }
        )

    return clap_weights, raw_results


def build_descriptor_outputs(
    clap: Any,
    audio: str | Path,
    anchor_config_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    anchors, anchor_map = resolve_anchor_records(anchor_config_path, clap=clap)
    clap_weights, raw_results = score_descriptor_audio(
        clap,
        audio,
        anchors,
        smoothing_alpha=float(anchor_map["smoothing_alpha"]),
    )
    return clap_weights, raw_results, anchor_map
