import beat_track
from clap import *
from utils import *

AUDIO = 'audio/testSong.mp3'

def main():
    beat_times, _ = beat_track.beat_track(AUDIO)
    beat_track.group_beats(beat_times, save_to_json=True)
    check_environment()
    clap = Clap()
    clap.retrieve_info()
    result = clap.analyze_audio(AUDIO)

    ##put it through llm and replace updated labels. main should generate a clap_weights.json that could be
    ##used for the boids.cpp
    ##-----------------------------------------------------------------
    w = []  #weights
    js = [] #
    for seg in result:
        print(f"\n------------------------------------------------")
        print(f"Segment {seg['start']:.2f}-{seg['end']:.2f}s")

        moods = seg["feature"].get("moods", [])
        labels = [m["label"] for m in moods]    #don't cast to nparray
        scores = np.array([m["score"] for m in moods])

        print("labels:", labels)
        print("scores", scores)
        ## switch out the label and the score here:
        

        emb = clap.get_text_embedding(labels).numpy()
        weights = clap.classify_new_emotion(emb)
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

    print(np.array(w))
    save_as_json("clap_weights", js, folder="")
    return w

if __name__ == "__main__":
    main()