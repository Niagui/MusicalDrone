import json
import argparse
import numpy as np
import beat_track as bt
import clap as cp
import utils
import logger_config as logger
from pathlib import Path
import hierarchical_choreography as hc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO = PROJECT_ROOT / "audio" / "oldTownRoad.mp3"

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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        default="testSong.mp3",
        help="Audio file path, or filename inside the audio/ folder",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Also generate json/hierarchical_plan.json for the new mind/body pipeline",
    )
    parser.add_argument(
        "--hierarchical-output",
        default="json/hierarchical_plan.json",
        help="Output path for the hierarchical choreography plan JSON",
    )
    parser.add_argument(
        "--hierarchical-config",
        default="config/hierarchical_config.json",
        help="Config file for the hierarchical choreography planner",
    )
    args = parser.parse_args()
    audio = resolve_audio_path(args.audio)

    beat_times, _ = bt.beat_track(audio)
    bt.group_beats(beat_times, save_to_json=True)
    utils.check_environment()
    clap = cp.Clap()
    clap.retrieve_info()
    result = clap.analyze_audio(audio)
    utils.save_as_json("clap_results", result, folder="")

    logger.log_debug("reading from llm variations")
    with open("json/llm_weights.json", "r") as f:
        llm_weights = json.load(f)
    llm_segments = llm_weights["segments"]

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
        scores = np.array([m["score"] for m in moods])

        ## switch out the label and the score here:
        top = max(llm_segments[i]["variations"], key=lambda x: x["weight"])
        labels.append(top["variant"])
        scores = np.append(scores / 4, top["weight"])

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

    utils.save_as_json("clap_weights", js, folder="")

    if args.hierarchical:
        # Reuse the fresh CLAP segments and beat tracker output we already computed
        # so the new planner stays aligned with the current analysis path.
        hc.generate_hierarchical_plan(
            clap_segments=result,
            beat_times=list(beat_times),
            output_path=args.hierarchical_output,
            config_path=args.hierarchical_config,
        )
    return w


if __name__ == "__main__":
    main()
    
