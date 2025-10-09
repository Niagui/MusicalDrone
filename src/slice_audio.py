import soundfile as sf
import librosa
import json
import numpy as np
from pathlib import Path

def parse_segments(json_file, level=12) -> tuple:
    """select the level of detail we want from different levels of segmentations

    Args:
        json_file (_type_): the json file that documents the splitting points of different sections of a song (verse, chorus, etc).
            This can be genereated from the musicseg_deepemb model used in segment.ipynb
        level (int, optional): level of detail extracted from the segmentation files. default to deepest level with most amount of detail. Defaults to 12.

    Returns:
        timestamps: _description_
        theme: 
    """
    with open(json_file, 'r') as f:
        details = json.load(f)

    return details[level-1][0], details[level-1][1]


def slice_audio(time_stamps, audio_file, output_dir=None, sr=22050) -> None:
    """slice audio based on the time stamps

    Args:
        time_stamps (_type_): list of [start, end] time pairs in seconds
        audio_file (_type_): path to the audio file to be sliced
        output_dir (_type_, optional): directory to save the sliced audio files. If None, will create a directory. Defaults to None.
        sr (int, optional): sample rate. Defaults to 22050.
    """

    y, sr = librosa.load(audio_file, sr=None)

    if output_dir is None:
        audio_file = audio_file.split('/')[-1].split('.')[0]
        output_dir = f"segments/{audio_file}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, [start, end] in enumerate(time_stamps):
        clip = y[int(start * sr) :int(end * sr)]
        sf.write(f"{output_dir}/{i:02d}_{start:.2f}-{end:.2f}.wav", clip, sr)
    return


if __name__ == '__main__':

    # example usage
    timestamps, themes = parse_segments("json/segmentation.json", level=12)
    print(timestamps)
    print(themes)
    slice_audio(timestamps, "audio/testSong.mp3")