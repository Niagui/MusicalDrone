from src.beat_track import beat_track

AUDIO = "audio/testSong.mp3"


def test_beat_track():
    beat_times, tempo = beat_track(AUDIO, save_to_json=False)
    assert len(beat_times) == 494
    assert tempo == [152.0]
