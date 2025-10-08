import librosa
import soundfile as sf
import json
import numpy as np

def getDurtion(audioFile):
    info = sf.info(audioFile)
    return info.frames / info.samplerate

def saveAsJson(filename, data):
    with open(f'json/{filename}.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t*t*(3.0 - 2.0*t)