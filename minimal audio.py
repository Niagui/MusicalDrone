import pygame
import time

AUDIO_PATH = "audio/testSong.mp3"

def test_audio():
    """Minimal audio playback test"""
    try:
        print("Initializing pygame mixer...")
        pygame.mixer.init()
        
        print(f"Loading audio: {AUDIO_PATH}")
        pygame.mixer.music.load(AUDIO_PATH)
        
        print("Starting playback...")
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.5)
            print(".", end="", flush=True)
        
        print("\nPlayback complete!")
        pygame.mixer.quit()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_audio()