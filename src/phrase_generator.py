#!/usr/bin/env python3
"""
Single-attractor phrase planner.

Reads phrase summaries from `clap_results.json`, asks an LLM for a compact
single-attractor phrase plan, then adds a heuristic beat plan in code.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

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
JSON_DIR_ENV_VAR = "DRONE_JSON_DIR"
MODEL_NAME = os.getenv("OPENAI_PHRASE_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_COMPLETIONS_URL = os.getenv(
    "OPENAI_CHAT_COMPLETIONS_URL", "https://api.openai.com/v1/chat/completions"
)

DEFAULT_PHRASE_BEAT_COUNT = 8
TOP_MOODS = 2
MAX_PROMPT_CHARS = 4500
MAX_BATCH_PHRASES = 8
MAX_COMPLETION_TOKENS = 2000
RETRIES = 3
BACKOFF_BASE_S = 0.8

SECTION_ROLES = {"stable", "buildup", "peak", "release", "transition"}
MOTION_MODES = {"hold", "advance", "retreat", "sweep_left", "sweep_right"}
VERTICAL_TRENDS = {"fall", "hold", "rise"}
TRANSITION_STYLES = {"smooth", "drift", "surge", "snap"}
BEAT_ACTIONS = {
    "hold",
    "advance",
    "retreat",
    "sweep_left",
    "sweep_right",
    "rise",
    "fall",
    "settle",
}


@dataclass(frozen=True)
class SectionSpan:
    index: int
    start: float
    end: float


@dataclass
class PhraseBlock:
    phrase_index: int
    start: float
    end: float
    beat_count: int
    prompt_text: str
    section_index: Optional[int] = None


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


_load_repo_env()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_generated_json_dir() -> Path:
    override = os.getenv(JSON_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / "json"


def get_generated_json_path(filename: str) -> Path:
    return get_generated_json_dir() / f"{filename}.json"


def get_pipeline_json_path(filename: str) -> Path:
    generated_path = get_generated_json_path(filename)
    if generated_path.exists():
        return generated_path
    return PROJECT_ROOT / "json" / f"{filename}.json"


def load_json(path: str | Path, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    if not file_path.exists():
        return default
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sections(path: str | Path) -> List[SectionSpan]:
    raw = load_json(path, default=[]) or []
    sections: List[SectionSpan] = []
    for index, item in enumerate(raw):
        if isinstance(item, list) and len(item) >= 2:
            sections.append(
                SectionSpan(index=index, start=float(item[0]), end=float(item[1]))
            )
    return sections


def resolve_section_index(
    start: float, end: float, sections: List[SectionSpan]
) -> Optional[int]:
    midpoint = 0.5 * (start + end)
    for section in sections:
        if section.start <= midpoint < section.end:
            return section.index
    return None


def dominant_label(feature: Dict[str, Any], category: str, default: str = "unknown") -> str:
    items = feature.get(category, [])
    if not items:
        return default
    return str(max(items, key=lambda item: item.get("score", 0.0)).get("label", default))


def top_labels(feature: Dict[str, Any], category: str, count: int) -> List[str]:
    items = sorted(feature.get(category, []), key=lambda item: item.get("score", 0.0), reverse=True)
    labels = []
    for item in items[:count]:
        label = item.get("label")
        if isinstance(label, str):
            labels.append(label)
    return labels


def count_phrase_beats(
    beat_times: List[float],
    start: float,
    end: float,
    default: int = DEFAULT_PHRASE_BEAT_COUNT,
) -> int:
    if not beat_times:
        return default

    tolerance = 0.03
    inner_beats = [t for t in beat_times if (start + tolerance) < t < (end - tolerance)]
    return max(1, len(inner_beats) + 1)


def phrase_to_block(
    phrase: Dict[str, Any],
    index: int,
    beat_times: List[float],
    sections: List[SectionSpan],
) -> PhraseBlock:
    start = float(phrase.get("start", 0.0))
    end = float(phrase.get("end", start))
    feature = phrase.get("feature", {})
    beat_count = count_phrase_beats(beat_times, start, end)
    section_index = resolve_section_index(start, end, sections)

    moods = ", ".join(top_labels(feature, "moods", TOP_MOODS)) or "unknown"
    valence = dominant_label(feature, "valence")
    arousal = dominant_label(feature, "arousal")
    tension = dominant_label(feature, "tension")

    lines = [
        f"phrase_index: {index}",
        f"time: {start:.2f}-{end:.2f}s",
        f"beats: {beat_count}",
    ]
    if section_index is not None:
        lines.append(f"section_index: {section_index}")
    lines.extend(
        [
            f"moods: {moods}",
            f"valence: {valence}",
            f"arousal: {arousal}",
            f"tension: {tension}",
        ]
    )

    return PhraseBlock(
        phrase_index=index,
        start=start,
        end=end,
        beat_count=beat_count,
        prompt_text="\n".join(lines),
        section_index=section_index,
    )


def build_phrase_blocks(
    phrases: List[Dict[str, Any]],
    beat_times: Optional[List[float]] = None,
    sections: Optional[List[SectionSpan]] = None,
) -> List[PhraseBlock]:
    beat_times = beat_times or []
    sections = sections or []
    return [
        phrase_to_block(phrase, index, beat_times, sections)
        for index, phrase in enumerate(phrases)
    ]


SYSTEM_PROMPT = f"""You are a JSON generator for single-attractor phrase planning.
Return strictly valid JSON in this format:
{{
  "phrases": [
    {{
      "phrase_index": integer,
      "section_role": "one of {sorted(SECTION_ROLES)}",
      "motion_mode": "one of {sorted(MOTION_MODES)}",
      "height_level": number,
      "depth_level": number,
      "speed_level": number,
      "vertical_trend": "one of {sorted(VERTICAL_TRENDS)}",
      "transition_style": "one of {sorted(TRANSITION_STYLES)}"
    }}
  ]
}}
Numeric fields must stay within [0, 1].
Do not output beat_plan. The program will add beat timing heuristically.
Do not output coordinates, trajectories, per-drone commands, or framewise control.
Return JSON only.
"""


def build_prompt(blocks: List[PhraseBlock]) -> str:
    body = "\n\n".join(block.prompt_text for block in blocks)
    return (
        "Plan one moving attractor at phrase level.\n"
        "Return exactly one record per phrase_index.\n"
        "Keep the plan compact and interpretable.\n\n"
        f"{body}"
    )


def strip_code_fences(raw: str) -> str:
    fence = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE)
    text = raw.strip()
    match = fence.match(text)
    return match.group(1) if match else text


def balanced_brace_slice(raw: str) -> Optional[str]:
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


def parse_json_strict(raw: str) -> Dict[str, Any]:
    text = raw.strip()

    for candidate in (text, strip_code_fences(text), balanced_brace_slice(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return json.loads(text)


def create_chat_completion(request_kwargs: Dict[str, Any]) -> str:
    if OpenAI is not None:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content or ""

    if not OPENAI_API_KEY:
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
            "Authorization": f"Bearer {OPENAI_API_KEY}",
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
        raise RuntimeError("OpenAI response did not contain any choices.")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def call_with_retries(prompt: str) -> Dict[str, Any]:
    request_kwargs = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "response_format": {"type": "json_object"},
    }

    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            raw = create_chat_completion(request_kwargs)
            return parse_json_strict(raw)
        except Exception as err:
            last_error = err
            print(f"[warn] phrase batch failed ({attempt}/{RETRIES}): {err}")
            time.sleep(BACKOFF_BASE_S * attempt)
    raise RuntimeError(last_error or "Phrase planning failed.")


def clamp_level(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric))


def pick_enum(value: Any, allowed: set[str], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def default_phrase_plan(block: PhraseBlock) -> Dict[str, Any]:
    return {
        "phrase_index": block.phrase_index,
        "section_role": "stable",
        "motion_mode": "hold",
        "height_level": 0.5,
        "depth_level": 0.5,
        "speed_level": 0.5,
        "vertical_trend": "hold",
        "transition_style": "smooth",
    }


def normalize_plan_records(
    obj: Dict[str, Any],
    batch: List[PhraseBlock],
) -> Dict[int, Dict[str, Any]]:
    by_index = {block.phrase_index: default_phrase_plan(block) for block in batch}
    for record in obj.get("phrases", []):
        if not isinstance(record, dict):
            continue
        phrase_index = record.get("phrase_index")
        if not isinstance(phrase_index, int) or phrase_index not in by_index:
            continue
        by_index[phrase_index] = {
            "phrase_index": phrase_index,
            "section_role": pick_enum(
                record.get("section_role"), SECTION_ROLES, "stable"
            ),
            "motion_mode": pick_enum(
                record.get("motion_mode"), MOTION_MODES, "hold"
            ),
            "height_level": clamp_level(record.get("height_level"), 0.5),
            "depth_level": clamp_level(record.get("depth_level"), 0.5),
            "speed_level": clamp_level(record.get("speed_level"), 0.5),
            "vertical_trend": pick_enum(
                record.get("vertical_trend"), VERTICAL_TRENDS, "hold"
            ),
            "transition_style": pick_enum(
                record.get("transition_style"), TRANSITION_STYLES, "smooth"
            ),
        }
    return by_index


def primary_action(plan: Dict[str, Any]) -> str:
    motion_mode = plan["motion_mode"]
    if motion_mode != "hold":
        return motion_mode
    if plan["vertical_trend"] != "hold":
        return plan["vertical_trend"]
    return "hold"


def build_beat_plan(block: PhraseBlock, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    beat_count = max(1, block.beat_count)
    primary = primary_action(plan)

    if beat_count == 1:
        return [{"beat": 1, "action": primary}]

    if (
        primary == "hold"
        and plan["speed_level"] < 0.35
        and plan["transition_style"] in {"smooth", "drift"}
    ):
        return [{"beat": 1, "action": "hold"}]

    if plan["section_role"] in {"buildup", "peak"}:
        accent_beat = max(2, min(beat_count - 1, round(beat_count * 0.75)))
    else:
        accent_beat = max(2, min(beat_count - 1, round(beat_count * 0.5)))

    events = [{"beat": 1, "action": "hold"}]

    if primary != "hold":
        events.append({"beat": accent_beat, "action": primary})

    if beat_count >= 3 and (
        plan["transition_style"] in {"smooth", "drift"} or plan["speed_level"] >= 0.6
    ):
        events.append({"beat": beat_count, "action": "settle"})

    deduped: List[Dict[str, Any]] = []
    seen_beats = set()
    for event in events:
        beat = int(event["beat"])
        action = event["action"]
        if beat in seen_beats:
            deduped[-1] = {"beat": beat, "action": action}
            continue
        seen_beats.add(beat)
        deduped.append({"beat": beat, "action": action})
    return deduped


def pack_batches(
    blocks: List[PhraseBlock],
    max_chars: int = MAX_PROMPT_CHARS,
    max_batch_phrases: int = MAX_BATCH_PHRASES,
) -> List[List[PhraseBlock]]:
    batches: List[List[PhraseBlock]] = []
    current: List[PhraseBlock] = []
    current_size = 0

    for block in blocks:
        block_size = len(block.prompt_text)
        if current and (
            current_size + block_size > max_chars
            or len(current) >= max_batch_phrases
        ):
            batches.append(current)
            current = []
            current_size = 0
        current.append(block)
        current_size += block_size

    if current:
        batches.append(current)
    return batches


def plan_batch(batch: List[PhraseBlock]) -> Dict[int, Dict[str, Any]]:
    response = call_with_retries(build_prompt(batch))
    return normalize_plan_records(response, batch)


def plan_batch_with_fallback(batch: List[PhraseBlock]) -> Dict[int, Dict[str, Any]]:
    try:
        return plan_batch(batch)
    except Exception as err:
        if len(batch) > 1:
            midpoint = len(batch) // 2
            print(
                f"[warn] Splitting phrase batch of {len(batch)} after parse failure: {err}"
            )
            left = plan_batch_with_fallback(batch[:midpoint])
            right = plan_batch_with_fallback(batch[midpoint:])
            return {**left, **right}

        block = batch[0]
        print(
            f"[warn] Falling back to default plan for phrase {block.phrase_index}: {err}"
        )
        return {block.phrase_index: default_phrase_plan(block)}


def build_output_payload(
    blocks: List[PhraseBlock],
    plan_records: Dict[int, Dict[str, Any]],
    source_file: str,
    model: str,
) -> Dict[str, Any]:
    phrases = []
    for block in blocks:
        plan = plan_records[block.phrase_index]
        phrase = {
            "phrase_index": block.phrase_index,
            "start": block.start,
            "end": block.end,
            "beat_count": block.beat_count,
            "section_role": plan["section_role"],
            "motion_mode": plan["motion_mode"],
            "height_level": plan["height_level"],
            "depth_level": plan["depth_level"],
            "speed_level": plan["speed_level"],
            "vertical_trend": plan["vertical_trend"],
            "transition_style": plan["transition_style"],
            "beat_plan": build_beat_plan(block, plan),
        }
        if block.section_index is not None:
            phrase["section_index"] = block.section_index
        phrases.append(phrase)

    return {
        "source_file": source_file,
        "model": model,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "total_phrases": len(phrases),
        "planner": "single_attractor_phrase_planner",
        "phrases": phrases,
    }


def main() -> None:
    input_json_path = get_generated_json_path("clap_results")
    beats_json_path = get_generated_json_path("beat_times")
    output_json_path = get_generated_json_path("phrase_plan")

    clap_phrases = load_json(input_json_path, default=[]) or []
    beat_times = load_json(beats_json_path, default=[]) or []

    if not clap_phrases:
        raise RuntimeError(f"No phrase data found in {input_json_path}.")

    blocks = build_phrase_blocks(clap_phrases, beat_times=beat_times, sections=[])
    batches = pack_batches(blocks)

    print(f"Loaded {len(blocks)} phrases")
    print(f"Created {len(batches)} prompt batches")

    plan_records: Dict[int, Dict[str, Any]] = {}
    llm_enabled = bool(OPENAI_API_KEY)

    if not llm_enabled:
        print("[warn] OPENAI_API_KEY missing. Using default phrase plans.")

    for batch_index, batch in enumerate(batches, start=1):
        print(f"Planning batch {batch_index}/{len(batches)} with {len(batch)} phrases")

        if llm_enabled:
            batch_records = plan_batch_with_fallback(batch)
        else:
            batch_records = {
                block.phrase_index: default_phrase_plan(block) for block in batch
            }

        plan_records.update(batch_records)

    payload = build_output_payload(blocks, plan_records, input_json_path.name, MODEL_NAME)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {output_json_path}")


if __name__ == "__main__":
    main()
