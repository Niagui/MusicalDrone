#!/usr/bin/env python3
"""
Label Variation Generator — Moods & Characteristics Only (with Segment Timing)

This script reads `clap_results.json`, generates variations **only for moods and
characteristics** (excluding valence, arousal, and tension), and writes
`llm_weights.json` with label variations **preserving the original time segments**.

Key features:
- Only processes moods and characteristics categories
- Preserves start/end time segments from input
- Uses adaptive batching based on prompt size
- Includes robust JSON parsing with retries
"""

import os
import json
import re
import time
import datetime
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

try:
    from openai import OpenAI

    SDK = "v1"
except Exception:
    OpenAI = None
    SDK = "legacy-or-missing"

# =============================================================================
# Configuration
# =============================================================================
load_dotenv()
INPUT_JSON = "json/clap_results.json"  # path to CLAP export
OUTPUT_JSON = "json/llm_weights.json"  # where we write variations
MODEL_NAME = "gpt-4o-mini"  # model to use
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)  # set your API key here or use os.environ['OPENAI_API_KEY']

# Sizing & batching
TARGET_PROMPT_TOKENS = 1400
TARGET_COMPLETION_TOKENS = 1600
MAX_TOKENS = TARGET_PROMPT_TOKENS + TARGET_COMPLETION_TOKENS

# Variations per label
VARIANTS_PER_LABEL = 2

# Retry policy
RETRIES = 3
BACKOFF_BASE_S = 0.8

# Optional pricing (set to enable cost estimates)
INPUT_USD_PER_1K = None
OUTPUT_USD_PER_1K = None

# REFACTORED: Only include moods
CATEGORIES = ["moods"]
TOP_PER_CATEGORY = 2

# =============================================================================
# Utility Functions
# =============================================================================


def estimate_tokens_from_text(s: str) -> int:
    """Rough token estimator: ~4 chars per token"""
    return max(1, int(len(s) / 4))


@dataclass
class UsageTally:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: Dict[str, Any]):
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)


def usd_cost(usage: UsageTally) -> Dict[str, Optional[float]]:
    inp = (
        None
        if INPUT_USD_PER_1K is None
        else usage.prompt_tokens / 1000.0 * INPUT_USD_PER_1K
    )
    out = (
        None
        if OUTPUT_USD_PER_1K is None
        else usage.completion_tokens / 1000.0 * OUTPUT_USD_PER_1K
    )
    tot = None if inp is None or out is None else inp + out
    return {"input_usd": inp, "output_usd": out, "total_usd": tot}


# =============================================================================
# JSON Parsing Helpers
# =============================================================================


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from JSON"""
    fence = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.IGNORECASE)
    m = fence.match(text.strip())
    return m.group(1) if m else text


def _balanced_brace_slice(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text"""
    s = text
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    last_valid_end = None
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_valid_end = i
    if last_valid_end is not None:
        return s[start : last_valid_end + 1]
    return None


def parse_json_strict(raw: str) -> Dict[str, Any]:
    """Parse JSON with multiple fallback strategies"""
    # Path 1: direct
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Path 2: strip fences
    try:
        return json.loads(_strip_code_fences(raw))
    except Exception:
        pass
    # Path 3: largest balanced slice
    slice_ = _balanced_brace_slice(raw)
    if slice_ is not None:
        return json.loads(slice_)
    # Give up
    return json.loads(raw)


def validate_variations_schema(obj: Dict[str, Any]):
    """Validate the expected JSON schema"""
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON must be an object.")
    if "variations" not in obj or not isinstance(obj["variations"], list):
        raise ValueError('JSON must contain key "variations" as a list.')
    for i, rec in enumerate(obj["variations"]):
        if not isinstance(rec, dict):
            raise ValueError(f"variations[{i}] must be an object.")
        for k in ("source", "variant", "weight"):
            if k not in rec:
                raise ValueError(f"Missing key {k} in variations[{i}].")
        if not isinstance(rec["source"], str):
            raise ValueError(f"variations[{i}].source must be a string.")
        if not isinstance(rec["variant"], str):
            raise ValueError(f"variations[{i}].variant must be a string.")
        if not isinstance(rec["weight"], (int, float)):
            raise ValueError(f"variations[{i}].weight must be a number.")


def normalize_variations(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize variation records to standard format"""
    out = []
    for rec in obj["variations"]:
        out.append(
            {
                "source": str(rec["source"]),
                "variant": str(rec["variant"]),
                "weight": float(rec["weight"]),
            }
        )
    return out


# =============================================================================
# Prompt Building
# =============================================================================


def topn(items, n=3, key=lambda x: x.get("score", 0.0)):
    """Get top N items by score"""
    return list(sorted(items, key=key, reverse=True))[:n]


@dataclass
class SegmentBlock:
    """Represents a segment with its prompt block and metadata"""

    segment_index: int
    start: float
    end: float
    prompt_text: str
    original_labels: List[
        Dict[str, Any]
    ]  # Store the original labels with their categories


def segment_to_block(seg: Dict[str, Any], index: int) -> SegmentBlock:
    """Convert a segment to a SegmentBlock (moods and characteristics only)"""
    feat = seg.get("feature", {})
    lines = []
    original_labels = []

    # Extract time window
    start = seg.get("start", 0.0)
    end = seg.get("end", 0.0)
    lines.append(f"segment_{index} (time: {start:.2f}-{end:.2f}s):")

    # Only process moods and characteristics
    for cat in CATEGORIES:
        arr = feat.get(cat, [])
        if not arr:
            continue
        lines.append(f"{cat}:")
        for it in topn(arr, TOP_PER_CATEGORY):
            lab = it.get("label", "")
            sc = float(it.get("score", 0.0))
            lines.append(f"- {lab} (weight={sc:.3f})")
            original_labels.append({"category": cat, "label": lab, "score": sc})

    prompt_text = "\n".join(lines)
    return SegmentBlock(
        segment_index=index,
        start=start,
        end=end,
        prompt_text=prompt_text,
        original_labels=original_labels,
    )


SYSTEM_JSON_SCHEMA = """You are a JSON generator. 
Return strictly valid JSON using UTF-8 with this schema:
{
  "variations": [
    {"source": "string", "variant": "string", "weight": number}
  ]
}
No extra text, no comments, no markdown fences.
"""


def build_prompt(blocks: List[SegmentBlock]) -> str:
    """Build the complete prompt from segment blocks"""
    prompt_texts = [block.prompt_text for block in blocks]
    return (
        "For each source label in the segments below, produce "
        f"{VARIANTS_PER_LABEL} diverse, non-redundant 'variant' strings that are realistic synonyms or near-synonyms. "
        "Propagate the input label weight into each variant's 'weight' (you may slightly jitter to reflect variety).\n\n"
        + "\n\n".join(prompt_texts)
    )


# =============================================================================
# OpenAI API Calls
# =============================================================================


def get_openai_client():
    """Initialize OpenAI client"""
    key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Set OPENAI_API_KEY in the config or as an environment variable."
        )
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not found. `pip install openai>=1.0.0`")
    client = OpenAI(api_key=key)
    return client


def call_json_mode(
    prompt: str, *, model: str, max_tokens: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Call OpenAI API in JSON mode"""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_JSON_SCHEMA},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": "Respond ONLY with a JSON object per the schema.",
            },
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    print(
        "RAW FROM MODEL >>>", repr(raw)[:200] + ("..." if len(repr(raw)) > 200 else "")
    )
    obj = parse_json_strict(raw)
    validate_variations_schema(obj)
    usage = getattr(resp, "usage", {}) or {}
    usage = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None)
        or usage.get("prompt_tokens"),
        "completion_tokens": getattr(usage, "completion_tokens", None)
        or usage.get("completion_tokens"),
        "total_tokens": getattr(usage, "total_tokens", None)
        or usage.get("total_tokens"),
    }
    return obj, usage


def call_with_retries(
    prompt: str, *, model: str, max_tokens: int, retries: int, backoff_base_s: float
):
    """Call OpenAI with exponential backoff retries"""
    last_err = None
    for i in range(1, retries + 1):
        try:
            return call_json_mode(prompt, model=model, max_tokens=max_tokens)
        except Exception as e:
            last_err = e
            print(f"[warn] JSON call failed (attempt {i}/{retries}): {e}")
            time.sleep(backoff_base_s * i)
            prompt = "STRICT JSON ONLY. " + prompt
    raise RuntimeError(f"Failed after {retries} retries: {last_err}")


# =============================================================================
# Batching Strategy
# =============================================================================


def load_clap(path: str) -> List[Dict[str, Any]]:
    """Load CLAP results from JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_segment_blocks(segments: List[Dict[str, Any]]) -> List[SegmentBlock]:
    """Build SegmentBlocks from segments"""
    return [segment_to_block(seg, i) for i, seg in enumerate(segments)]


def count_labels_in_block(block: SegmentBlock) -> int:
    """Count the number of labels in a SegmentBlock"""
    return sum(
        1 for line in block.prompt_text.splitlines() if line.strip().startswith("- ")
    )


def estimate_completion_tokens_for_block(block: SegmentBlock) -> int:
    """Estimate completion tokens needed for a block"""
    VAR_TOKENS = 24
    labels = count_labels_in_block(block)
    return labels * VARIANTS_PER_LABEL * VAR_TOKENS


def pack_batches(
    blocks: List[SegmentBlock], target_prompt_tokens: int, target_completion_tokens: int
) -> List[List[SegmentBlock]]:
    """Pack SegmentBlocks into batches respecting token limits"""
    batches = []
    cur, cur_p_tokens, cur_c_tokens = [], 0, 0
    for b in blocks:
        bp = estimate_tokens_from_text(b.prompt_text)
        bc = estimate_completion_tokens_for_block(b)
        if cur and (
            (cur_p_tokens + bp) > target_prompt_tokens
            or (cur_c_tokens + bc) > target_completion_tokens
        ):
            batches.append(cur)
            cur, cur_p_tokens, cur_c_tokens = [], 0, 0
        cur.append(b)
        cur_p_tokens += bp
        cur_c_tokens += bc
    if cur:
        batches.append(cur)
    return batches


def plan_calls(clap_segments: List[Dict[str, Any]]):
    """Plan API calls with batching strategy"""
    blocks = build_segment_blocks(clap_segments)
    batches = pack_batches(blocks, TARGET_PROMPT_TOKENS, TARGET_COMPLETION_TOKENS)
    est_prompt_tokens = sum(estimate_tokens_from_text(b.prompt_text) for b in blocks)
    return {
        "num_segments": len(clap_segments),
        "num_batches": len(batches),
        "estimated_prompt_tokens_total": est_prompt_tokens,
        "batches": batches,
    }


def distribute_variations_to_segments(
    batch: List[SegmentBlock], variations: List[Dict[str, Any]]
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Distribute variations back to their source segments.
    Matches variations to segments based on source labels.
    """
    segment_variations = {block.segment_index: [] for block in batch}

    # Create a mapping of source labels to segment indices
    label_to_segments = {}
    for block in batch:
        for label_info in block.original_labels:
            label = label_info["label"]
            if label not in label_to_segments:
                label_to_segments[label] = []
            label_to_segments[label].append(block.segment_index)

    # Distribute variations to segments
    for var in variations:
        source = var["source"]
        if source in label_to_segments:
            # Add this variation to all segments that have this source label
            for seg_idx in label_to_segments[source]:
                segment_variations[seg_idx].append(var)

    return segment_variations


# =============================================================================
# Main Execution
# =============================================================================


def main():
    """Main execution function"""
    print("=" * 70)
    print("Label Variation Generator — Moods & Characteristics Only")
    print("(Preserving Time Segments)")
    print("=" * 70)

    print("\nLoading CLAP results...")
    clap = load_clap(INPUT_JSON)
    print(f"Loaded {len(clap)} segments")

    print("\nPlanning batches...")
    plan = plan_calls(clap)
    print(f"Segments: {plan['num_segments']}")
    print(f"Batches: {plan['num_batches']}")
    print(f"Estimated prompt tokens (total): {plan['estimated_prompt_tokens_total']}")

    usage_total = UsageTally()

    # Initialize segment variations storage
    all_segment_variations = {i: [] for i in range(len(clap))}

    print("\nProcessing batches...")
    for bi, batch in enumerate(plan["batches"], start=1):
        print(f"\n=== Batch {bi}/{plan['num_batches']} (segments={len(batch)}) ===")
        prompt = build_prompt(batch)
        obj, usage = call_with_retries(
            prompt,
            model=MODEL_NAME,
            max_tokens=TARGET_COMPLETION_TOKENS,
            retries=RETRIES,
            backoff_base_s=BACKOFF_BASE_S,
        )
        usage_total.add(usage or {})

        # Get normalized variations
        variations = normalize_variations(obj)
        print(f"Received {len(variations)} variations")

        # Distribute variations to their source segments
        segment_vars = distribute_variations_to_segments(batch, variations)
        for seg_idx, vars_list in segment_vars.items():
            all_segment_variations[seg_idx].extend(vars_list)

    # Build output structure with segments
    print("\nBuilding output structure...")
    output_segments = []
    for i, seg in enumerate(clap):
        segment_out = {
            "start": seg.get("start"),
            "end": seg.get("end"),
            "original_labels": {
                cat: [
                    {"label": item["label"], "score": item["score"]}
                    for item in seg.get("feature", {}).get(cat, [])[:TOP_PER_CATEGORY]
                ]
                for cat in CATEGORIES
                if seg.get("feature", {}).get(cat)
            },
            "variations": all_segment_variations[i],
        }
        output_segments.append(segment_out)

    # Write output
    print("\nWriting output...")
    out = {
        "source_file": os.path.basename(INPUT_JSON),
        "model": MODEL_NAME,
        "generated_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat()
        + "Z",
        "categories_processed": CATEGORIES,
        "total_segments": len(output_segments),
        "segments": output_segments,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Calculate statistics
    # total_variations = sum(len(seg["variations"]) for seg in output_segments)
    segments_with_variations = sum(1 for seg in output_segments if seg["variations"])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Output file: {OUTPUT_JSON}")
    print(f"Total segments: {len(output_segments)}")
    print(f"Segments with variations: {segments_with_variations}")
    print(f"Usage: {usage_total}")

    cost = usd_cost(usage_total)
    if cost["total_usd"] is not None:
        print(f"✓ Estimated cost: ${cost['total_usd']:.4f}")
    else:
        print("✓ Set INPUT_USD_PER_1K and OUTPUT_USD_PER_1K for cost estimates")

    print(f"\n✓ Categories processed: {', '.join(CATEGORIES)}")
    print("✓ Categories excluded: valence, arousal, tension")
    print("=" * 70)


if __name__ == "__main__":
    main()
