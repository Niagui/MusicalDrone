import librosa
import numpy as np
import utils
import logger_config as logger


def beat_track(audio_file, save_to_json=False) -> tuple:
    """
    Extract beat tracking information from an audio file using librosa.
    fftsize = 1024
    window = 1024
    hop = 512
    melBin = 128
    sr = 22050

    Args:
        audio_file (_type_): path to audio file

    Returns:
        beat_times (np.ndarray): Array of beat times in seconds.
        tempo (float): Estimated tempo in beats per minute (BPM).
    """
    logger.log_info("Running beat tracker...")
    y, sr = librosa.load(audio_file, sr=22050)
    tempo, beattrack = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beattrack, sr=sr)
    if save_to_json:
        utils.save_as_json("beat_times", list(beat_times), folder="")
    return np.array(beat_times), np.round(tempo)  # cast bpm to int for convenience


def group_beats(beat_times, k=8, save_to_json=False, destination=None) -> list:
    """_summary_

    Args:
        beat_times (_type_): _description_
        k (int, optional): _description_. Defaults to 4.
        save_to_json (bool, optional): _description_. Defaults to False.

    Returns:
        list: _description_
    """
    b = list(beat_times)
    segments = []
    for i in range(0, len(b) - k, k):
        segments.append((b[i], b[i + k]))
    segments = list(segments)

    if save_to_json:
        utils.save_as_json("k_beat_segments", segments, folder="")
    return segments


if __name__ == "__main__":
    # demo
    audiofile = "audio/testSong.mp3"
    beat_times, tempo = beat_track(audiofile, save_to_json=True)
    print(len(beat_times))
    print(tempo)
    group_beats(beat_times, save_to_json=True)
    print(f"Estimated tempo: {tempo} BPM")
    print("Beat times (s):", beat_times)
