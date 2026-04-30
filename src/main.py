import json
import argparse
import numpy as np
import beat_track as bt
import clap as cp
import descriptor_anchor_mapper as dam
import utils
import logger_config as logger
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_audio_path(audio_arg: str) -> Path:
    audio_folder = PROJECT_ROOT / "audio"

    # First: treat it as a direct path
    candidate = Path(audio_arg)
    if candidate.is_file():
        return candidate.resolve()

    # Second: treat it as a filename inside audio/
    candidate_in_audio = audio_folder / audio_arg
    if candidate_in_audio.is_file():
        return candidate_in_audio.resolve()

    raise FileNotFoundError(
        f"Could not find audio file '{audio_arg}'. "
        f"Tried '{candidate}' and '{candidate_in_audio}'."
    )


def load_json_file(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_generated_json_path(filename: str) -> Path:
    return utils.get_generated_json_path(filename)


def load_cached_clap_results():
    clap_results_path = get_generated_json_path("clap_results")
    if not clap_results_path.exists():
        return None

    logger.log_info(f"Reusing cached clap results from {clap_results_path}")
    return load_json_file(clap_results_path)


def load_llm_segments() -> list:
    llm_weights_path = get_generated_json_path("llm_weights")
    if not llm_weights_path.exists():
        raise FileNotFoundError(
            f"Could not find llm_weights.json at '{llm_weights_path}'. "
            "Run the LLM generation step first."
        )

    payload = load_json_file(llm_weights_path)
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError(
            f"Expected 'segments' list in '{llm_weights_path}', got {type(segments)}."
        )
    return segments


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        default="testSong.mp3",
        help="Audio file path, or filename inside the audio/ folder",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Apply llm_weights.json from the current generated json directory",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate reusable analysis JSON and stop before clap_weights",
    )
    parser.add_argument(
        "--descriptor-anchors",
        action="store_true",
        help="Use descriptor-anchor CLAP mapping instead of raw mood-anchor mapping",
    )
    parser.add_argument(
        "--anchor-config",
        default="json/descriptor_anchor_config.json",
        help="Descriptor anchor config JSON path",
    )
    args = parser.parse_args()
    audio = resolve_audio_path(args.audio)

    beat_times, _ = bt.beat_track(audio, save_to_json=True)
    bt.group_beats(beat_times, save_to_json=True)
    clap = None
    result = load_cached_clap_results()

    if result is None:
        utils.check_environment()
        clap = cp.Clap()
        clap.retrieve_info()
        result = clap.analyze_audio(audio)
        utils.save_as_json("clap_results", result)

    if args.prepare_only:
        logger.log_info("Prepare-only mode finished after caching analysis JSON.")
        return result

    if clap is None:
        utils.check_environment()
        clap = cp.Clap()
        clap.retrieve_info()

    if args.descriptor_anchors:
        if args.use_llm:
            logger.log_warning(
                "--use-llm label variations are ignored by --descriptor-anchors. "
                "Descriptor anchors use their own LLM expansion/fallback path."
            )

        logger.log_info("Building descriptor-anchor CLAP weights")
        clap_weights, descriptor_results, anchor_map = dam.build_descriptor_outputs(
            clap=clap,
            audio=audio,
            anchor_config_path=args.anchor_config,
        )
        utils.save_as_json("descriptor_anchor_map", anchor_map)
        utils.save_as_json("descriptor_clap_results", descriptor_results)
        utils.save_as_json("clap_weights", clap_weights)
        return [segment["weights"] for segment in clap_weights]

    llm_segments = None
    if args.use_llm:
        logger.log_debug("reading from llm variations")
        llm_segments = load_llm_segments()
        if len(llm_segments) != len(result):
            raise ValueError(
                "llm_weights.json segment count does not match clap_results.json. "
                "Regenerate the cached LLM outputs for this audio."
            )

    ##put it through llm and replace updated labels. main should generate a clap_weights.json that could be
    ##used for the boids.cpp
    ##-----------------------------------------------------------------
    w = []  # weights
    js = []  # data in json format

    logger.log_debug("Begin calculating emotion embeddings")
    for i, seg in enumerate(result):
        print("\n------------------------------------------------")
        print(f"Segment {seg['start']:.2f}-{seg['end']:.2f}s")

        moods = seg["feature"].get("moods", [])
        labels = [m["label"] for m in moods]  # don't cast to nparray
        scores = np.array([m["score"] for m in moods], dtype=float)

        if llm_segments is not None:
            variations = llm_segments[i].get("variations", [])
            if variations:
                top = max(variations, key=lambda x: x["weight"])
                labels.append(top["variant"])
                scores = np.append(scores / 4, float(top["weight"]))
            else:
                logger.log_warning(
                    f"No LLM variations found for segment {i}; using base CLAP labels."
                )

        # check labels and weights
        print("labels:", labels)
        print("scores", scores)

        emb = clap.get_text_embedding(labels).numpy()
        weights = clap.classify_new_emotion(emb, k=3)
        final_weights = (weights * scores[:, None]).sum(axis=0)
        w.append(final_weights)
        js.append(
            {"start": seg["start"], "end": seg["end"], "weights": list(final_weights)}
        )

        print("weights", final_weights)

    utils.save_as_json("clap_weights", js)
    return w


if __name__ == "__main__":
    main()
