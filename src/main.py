import beat_track, clap
from utils import *

AUDIO = 'audio/testSong.mp3'

def main():
    beat_times, _ = beat_track.beat_track(AUDIO)
    beat_track.group_beats(beat_times, save_to_json=True)
    check_environment()
    
    return

if __name__ == "__main__":
    main()