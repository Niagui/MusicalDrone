from transformers import pipeline
import librosa, json
from logger_config import *

def clap_generate_features():

    audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/larger_clap_general", batch=8, device='cuda')