import numpy as np
import librosa
import torch
from transformers import pipeline, ClapProcessor, ClapModel
import json
import logger_config as logger
import utils
from pathlib import Path


class Clap:
    
    def __init__(self, model_name="laion/larger_clap_general", device=None):
        logger.log_info("initiating clap")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained("laion/larger_clap_general")
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.audio_classifier = pipeline(
            task="zero-shot-audio-classification",
            model="laion/larger_clap_general",
            batch=8,
            device=device,
        )
        logger.log_info("clap initiation successful")

    def _emb(self, text):
        """
        returns a single pyTorch array.
        """
        with torch.no_grad():
            inputs = self.processor(
                text=text, return_tensors="pt", padding=True, truncation=True
            )
            emb = self.model.get_text_features(**inputs)
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

    def get_text_embedding(self, text: list):
        """
        with cache logic. Still return a singleton
        """
        return self._emb(text)

    def retrieve_info(self):
        logger.log_info("retrieving clap info")
        clap_label_json = utils.get_shared_json_path("clap_labels")
        with open(clap_label_json, "r") as f:
            self.music_labels = json.load(f)

        anchor_labels_json = utils.get_shared_json_path("anchor_labels")
        with open(anchor_labels_json, "r") as f:
            self.anchor_labels = json.load(f)
            self.anchor_labels_set = set(self.anchor_labels)  # fast lookup

        time_segments_json = utils.get_pipeline_json_path("k_beat_segments")
        with open(time_segments_json, "r") as f:
            self.k_beats_segments = json.load(f)

        self.anchor_labels_emb = {}
        emb = self._emb(self.anchor_labels)
        for i, label in enumerate(self.anchor_labels):
            self.anchor_labels_emb[label] = emb[i].tolist()

        logger.log_info("retrieving clap info successful")

    def analyze_audio(
        self, audio, time_base=None, labels=None, sr=22050, k=3, threshold=0.1
    ):
        if time_base is None:
            time_base = self.k_beats_segments
        if labels is None:
            labels = self.music_labels

        logger.log_info("analyzing audio")
        result = []
        y, sr = librosa.load(audio, sr=sr)
        for [start, end] in time_base:
            chunk = y[int(round(start * sr)) : int(round(end * sr))]  # select chunk
            features = {}
            for label in labels:
                classes = labels[label]
                predictions = self.audio_classifier(chunk, candidate_labels=classes)

                top_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)
                filtered = top_preds[:k]
                features[label] = filtered

            result.append({"start": start, "end": end, "feature": features})
        logger.log_info("retrieving clap info")
        return result

    def get_cosine_similarity(
        self, new_label, anchor_labels, center=True, remove_pcs=0
    ):
        new = np.array(new_label)
        anchor = np.array(anchor_labels)

        new = new / np.linalg.norm(new)
        anchor = anchor / np.linalg.norm(anchor, axis=1, keepdims=True)

        if center:
            mu = anchor.mean(axis=0, keepdims=True)
            anchor = anchor - mu
            new = new - mu.squeeze(0)

        ##idk what this is but chat said it can help remove "directionness"
        if remove_pcs and remove_pcs > 0:
            # SVD computes principal components
            U, S, VT = np.linalg.svd(anchor, full_matrices=False)
            P = VT[:remove_pcs].T  # (d, k)

            # Project out dominant PCs
            anchor = anchor - (anchor @ P) @ P.T
            new = new - (new @ P) @ P.T

        similarities = anchor @ new
        return similarities

    def classify_new_emotion(self, emotion_vector, k=3, temperature=0.05):
        # pick top k anchors to compare with
        # controls how sharp on the weighting
        val = list(self.anchor_labels_emb.values())
        key = list(self.anchor_labels_emb.keys())
        # emotion_vector = np.array(self.get_text_embedding(emotion_vector).tolist()[0])
        sims = np.array(
            [self.get_cosine_similarity(vec, val) for vec in emotion_vector]
        )
        logger.log_debug(f"sims:{sims.shape}")

        top_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        w = np.zeros_like(sims)
        row_idx = np.arange(sims.shape[0])[:, None]
        w[row_idx, top_idx] = np.exp(sims[row_idx, top_idx] / temperature)
        w /= w.sum(axis=1, keepdims=True) + np.finfo(float).eps
        logger.log_debug(f"weights:{w.shape}")

        for i, (s, idxs) in enumerate(zip(sims, top_idx)):
            print(f"\n--- Emotion vector {i} ---")
            for j in idxs:
                print(
                    f"Anchor #{j:2d} ({key[j]}): sim={s[j]:.3f}, weight={w[i, j]:.3f}"
                )
        return w


def clap_pipeline(audio_file):
    clap = Clap()
    clap.retrieve_info()
    result = clap.analyze_audio(audio_file)

    ## variation
    w = []
    js = []
    for seg in result:
        print("\n------")
        print(f"Segment {seg['start']:.2f}-{seg['end']:.2f}s")
        moods = seg["feature"].get("moods", [])
        labels = [m["label"] for m in moods]  # don't cast to nparray
        scores = np.array([m["score"] for m in moods])

        ## replace the

        emb = clap.get_text_embedding(labels).numpy()
        weights = clap.classify_new_emotion(emb)
        final_weights = (weights * scores[:, None]).sum(axis=0)
        w.append(final_weights)
        js.append(
            {"start": seg["start"], "end": seg["end"], "weights": list(final_weights)}
        )

        print("Labels:", labels)
        print("Scores:", scores)
        print("weights", final_weights)

    print(np.array(w))
    utils.save_as_json("clap_weights", js)
    return w


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    AUDIO = PROJECT_ROOT / "audio" / "testSong.mp3"
    final_weights = clap_pipeline(AUDIO)
