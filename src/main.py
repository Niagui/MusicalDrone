import json
import numpy as np
import beat_track as bt
import clap as cp
import utils
import logger_config as logger

AUDIO = 'audio/testSong.mp3'

def main():
    beat_times, _ = bt.beat_track(AUDIO)
    bt.group_beats(beat_times, save_to_json=True)
    utils.check_environment()
    clap = cp.Clap()
    clap.retrieve_info()
    result = clap.analyze_audio(AUDIO)

    logger.log_debug("reading from llm variations")
    with open('json/llm_weights.json', 'r') as f:
        llm_weights = json.load(f)
    llm_segments = llm_weights["segments"]


    ##put it through llm and replace updated labels. main should generate a clap_weights.json that could be
    ##used for the boids.cpp
    ##-----------------------------------------------------------------
    w = []  #weights
    js = [] #data in json format

    logger.log_debug("Begin calculating emotion embeddings")
    for i, seg in enumerate(result):
        print("\n------------------------------------------------")
        print(f"Segment {seg['start']:.2f}-{seg['end']:.2f}s")

        moods = seg["feature"].get("moods", [])
        labels = [m["label"] for m in moods]    #don't cast to nparray
        scores = np.array([m["score"] for m in moods])

        ## switch out the label and the score here:
        top = max(llm_segments[i]["variations"], key=lambda x: x["weight"])
        labels.append(top["variant"])
        scores = np.append(scores/4, top["weight"])

        # check labels and weights
        print("labels:", labels)
        print("scores", scores)

        emb = clap.get_text_embedding(labels).numpy()
        weights = clap.classify_new_emotion(emb, k=3)
        final_weights = (weights * scores[:, None]).sum(axis=0)
        w.append(final_weights)
        js.append(
            {
                "start": seg['start'],
                "end": seg['end'],
                "weights": list(final_weights)
            }
        )

        print("weights", final_weights)

    utils.save_as_json("clap_weights", js, folder="")
    return w

if __name__ == "__main__":
    main()