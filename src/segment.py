import os
import warnings
import torch, tensorflow as tf
import importlib.util
from pathlib import Path
import musicsections
import librosa
import librosa.display
import numpy as np

AUDIO = "../audio/oldTownRoad.mp3"
PROJECT_ROOT = Path.cwd()
helper_path = PROJECT_ROOT / ".." /"src" / "utils.py"
spec = importlib.util.spec_from_file_location("utils", helper_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

torch.backends.cudnn.enabled = False
print("torch:", torch.__version__, "cuda?", torch.cuda.is_available())
print("tf:", tf.__version__)
if torch.cuda.is_available():
    print("GPU(s):", torch.cuda.device_count(), torch.cuda.get_device_name(0))


print("Torch build:", torch.__version__)
print("CUDA compiled:", torch.version.cuda)
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("Device count:", torch.cuda.device_count())
print(torch.__version__)

print(Path.cwd())
deepsim_model_folder = PROJECT_ROOT/".."/"external"/"musicseg_deepemb"/"models"/"deepsim"
fewshot_model_folder = PROJECT_ROOT/".."/"external"/"musicseg_deepemb"/"models"/"fewshot"

model_deepsim = musicsections.load_deepsim_model(deepsim_model_folder)
model_fewshot = musicsections.load_fewshot_model(fewshot_model_folder)

audiofile = AUDIO
segmentations, features = musicsections.segment_file(
    audiofile, 
    deepsim_model=model_deepsim,
    fewshot_model=model_fewshot,
    min_duration=8,
    mu=0.5,
    gamma=0.5,
    beats_alg="librosa",
    beats_file=None)

musicsections.plot_segmentation(segmentations)
utils.save_as_json('segmentation', segmentations)


ya, sra = librosa.load(audiofile)

y = librosa.resample(ya, orig_sr=sra, target_sr=22050)
hop_length = 512
tempo, beattrack = librosa.beat.beat_track(y=y, sr=22050, hop_length=hop_length)
beat_times = librosa.frames_to_time(beattrack, sr=22050, hop_length=hop_length)
utils.save_as_json('beatTrack', beat_times.tolist())