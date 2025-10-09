import librosa
import soundfile as sf
import json
import numpy as np
from numpy.typing import ArrayLike

def get_duration(audio_file) -> float:

    """Get the duration of an audio file in seconds

    Args:
        audio_file: path to audio file
    Returns:
        duration (float): duration in seconds

    """
    info = sf.info(audio_file)
    return info.frames / info.samplerate

def save_as_json(filename, data) -> None:

    """
    Save data as a json file

    Args:
        filename: file name for the json file to be store (without extension)
        data (list): data stored as array
    """
    with open(f'json/{filename}.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    return

def smooth_step(t: float) -> float:

    """Smoothstep function to smooth the transition between two values
    Args:
        t (float): input value between 0 and 1
    Returns:
        float: smoothed value between 0 and 1
    """
    t = np.clip(t, 0.0, 1.0)
    return t*t*(3.0 - 2.0*t)

def normalize(values: ArrayLike) -> np.ndarray:

    """arousal and valences are ranged from [1,9] and normalize to [-1,1]
    Args:
        values (ArrayLike): input values between 1 and 9
    Returns:
        np.ndarray: normalized values between -1 and 1
    """
    v = np.asarray(values, dtype=float)
    return 2 * ((values - 1) / 8) - 1
