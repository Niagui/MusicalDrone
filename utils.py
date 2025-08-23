import librosa
import soundfile as sf

def getDurtion(audioFile):
    info = sf.info(audioFile)
    return info.frames / info.samplerate