import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import logger_config as logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_JSON_DIR = PROJECT_ROOT / "json"
JSON_DIR_ENV_VAR = "DRONE_JSON_DIR"


def get_shared_json_dir() -> Path:
    return SHARED_JSON_DIR


def get_generated_json_dir(folder: str | None = "../") -> Path:
    override = os.getenv(JSON_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    if folder in {None, "", ".", "./", "../"}:
        return SHARED_JSON_DIR

    folder_path = Path(folder).expanduser()
    if folder_path.is_absolute():
        return folder_path / "json"
    return (Path.cwd() / folder_path / "json").resolve()


def get_shared_json_path(filename: str) -> Path:
    return get_shared_json_dir() / f"{filename}.json"


def get_generated_json_path(filename: str, folder: str | None = "../") -> Path:
    return get_generated_json_dir(folder) / f"{filename}.json"


def get_pipeline_json_path(filename: str) -> Path:
    generated_path = get_generated_json_path(filename)
    if generated_path.exists():
        return generated_path
    return get_shared_json_path(filename)


def get_duration(audio_file) -> float:
    """Get the duration of an audio file in seconds

    Args:
        audio_file: path to audio file
    Returns:
        duration (float): duration in seconds

    """
    info = sf.info(audio_file)
    return info.frames / info.samplerate


def save_as_json(filename, data, folder="../") -> None:
    """
    Save data as a json file

    Args:
        filename: file name for the json file to be store (without extension)
        data (list): data stored as array
    """
    output_path = get_generated_json_path(filename, folder)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
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
    return t * t * (3.0 - 2.0 * t)


def normalize(values) -> np.ndarray:
    """arousal and valences are ranged from [1,9] and normalize to [-1,1]
    Args:
        values (ArrayLike): input values between 1 and 9
    Returns:
        np.ndarray: normalized values between -1 and 1
    """
    v = np.asarray(values, dtype=float)
    return 2 * ((v - 1) / 8) - 1


def check_environment():
    torch.backends.cudnn.enabled = True
    if torch.cuda.is_available():
        logger.log_info(
            f"GPU(s):{torch.cuda.device_count()} {torch.cuda.get_device_name(0)}"
        )
    else:
        logger.log_info("Running on cpu")
    return
