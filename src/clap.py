from transformers import pipeline
from logger_config import *

def clap_generate_features():
    audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/larger_clap_general", batch=8, device='cuda')
    audio_classifier.check_model_type()

if __name__ == "__main__":
    clap_generate_features()