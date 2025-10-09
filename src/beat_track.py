import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import json
from utils import save_as_json


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
    y, sr = librosa.load(audio_file, sr=22050)
    tempo, beattrack = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beattrack, sr=sr)
    if save_to_json:
        save_as_json('beat_times', beat_times)
    return np.array(beat_times), np.round(tempo)     #cast bpm to int for convenience



if __name__ == '__main__':
    #demo
    audiofile = 'audio/testSong.mp3'
    beat_times, tempo = beat_track(audiofile)
    print(f"Estimated tempo: {tempo} BPM")
    print("Beat times (s):", beat_times)