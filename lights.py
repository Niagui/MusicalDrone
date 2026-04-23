## lights.py
## This file handles all LED light behavior for drones
## 1. Loads timed emotion weights from clap_weights.json
## 2. Waits for shared sequence start time from monitor_CBF.py
## 3. Switches drone LED lights based on strongest emotion weight

import json
import time
import threading


class LightController:

    def __init__(self):
        # Tracks whether lighting was successfully initialized per drone
        self.enabled = {}

        # Used to stop the light thread in case of shutdown/emergency
        self.stop_event = threading.Event()

        # Signals that the shared sequence start time has been published
        self.sequence_start_event = threading.Event()

        # Shared show start time, set by monitor_CBF.py after takeoff barrier
        self.sequence_start_time = None

        # Loaded JSON segments from clap_weights.json
        self.segments = []

        self.beat_times = []          # beat timestamps in seconds
        self.beat_flash_s = 0.08      # how long the white flash holds
        # RGB color per emotion index (0-6)
        self.emotion_colors = {
            0: (255, 255, 0),    # happy -> yellow
            1: (0, 100, 255),    # sad -> blue
            2: (150, 100, 255),  # sleepy -> purple
            3: (255, 0, 0),      # brave -> red
            4: (255, 120, 0),    # grumpy -> orange
            5: (0, 255, 255),    # scared -> light blue
            6: (255, 120, 180),  # shy -> pink
        }

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def load_weights(self, path="json/clap_weights.json"):
        with open(path, "r") as f:
            self.segments = json.load(f)
        print(f"[LIGHTS] Loaded {len(self.segments)}")

    def load_beat_times(self, path="json/beat_times.json"):
      with open(path, "r") as f:
          self.beat_times = json.load(f)   # list of floats
      print(f"[LIGHTS] Loaded {len(self.beat_times)} beats")

    @staticmethod
    def _brighten(rgb, factor=1.6):
        """Scale an RGB tuple up, clamped to 255."""
        return tuple(min(255, int(c * factor)) for c in rgb)

    def _param_exists(self, scf, name):
        """Return True if *name* (e.g. 'ring.effect') is present in the TOC."""
        try:
            toc = scf.cf.param.toc.toc
            group, var = name.split(".")
            return group in toc and var in toc[group]
        except Exception:
            return False

    def init_drone_lights(self, scf):
        """Initialize LED deck for one drone; disables lights if deck absent."""
        uri = scf.cf.link_uri

        required = ["ring.effect", "ring.solidRed", "ring.solidGreen", "ring.solidBlue"]
        if not all(self._param_exists(scf, p) for p in required):
            self.enabled[uri] = False
            print(f"[{uri}] LED ring params not found, disabling lights")
            return  

        try:
            scf.cf.param.set_value("ring.solidRed",   "20")
            scf.cf.param.set_value("ring.solidGreen", "20")
            scf.cf.param.set_value("ring.solidBlue",  "20")
            scf.cf.param.set_value("ring.effect", "7")   # solid color mode
            self.enabled[uri] = True
            print(f"[{uri}] Lights initialized (solid color mode)")
        except Exception as e:
            self.enabled[uri] = False
            print(f"[{uri}] Light init failed: {e}")

    # -------------------------------------------------------------------------
    # Timing sync
    # -------------------------------------------------------------------------

    def set_sequence_start(self, t0):
        """Record shared sequence start time and unblock waiting light threads."""
        self.sequence_start_time = t0
        self.sequence_start_event.set()

    def set_sequence(self, t0):
        """Backward-compatible alias for set_sequence_start."""
        self.set_sequence_start(t0)

    # -------------------------------------------------------------------------
    # Emotion helpers
    # -------------------------------------------------------------------------

    def strongest_weight_index(self, weights):
        """Return the index of the highest weight (0-6)."""
        return max(range(len(weights)), key=lambda i: weights[i])

    def emotion_index_to_rgb(self, idx):
        """Map emotion index to (R, G, B) tuple."""
        return self.emotion_colors.get(idx, (0, 0, 0))

    # -------------------------------------------------------------------------
    # Low-level LED commands
    # -------------------------------------------------------------------------

    def set_rgb(self, scf, r, g, b):
        """Set a solid RGB color immediately."""
        uri = scf.cf.link_uri
        if not self.enabled.get(uri, False):
            return
        try:
            scf.cf.param.set_value("ring.solidRed",   str(int(r)))
            scf.cf.param.set_value("ring.solidGreen", str(int(g)))
            scf.cf.param.set_value("ring.solidBlue",  str(int(b)))
            scf.cf.param.set_value("ring.effect", "7")   # solid color mode
        except Exception as e:
            self.enabled[uri] = False   # suppress future calls after link loss
            print(f"[{uri}] Failed to set RGB ({r},{g},{b}): {e}")

    def rgb_to_fadecolor_value(self, r, g, b):
        """Encode (R, G, B) as the 0x00RRGGBB integer expected by ring.fadeColor."""
        return (int(r) << 16) | (int(g) << 8) | int(b)

    def set_fade_rgb(self, scf, r, g, b, fade_time=0.25):
        """Cross-fade to an RGB color over fade_time seconds.
        Falls back to set_rgb if fade params are unavailable."""
        uri = scf.cf.link_uri
        if not self.enabled.get(uri, False):
            return

        if not (self._param_exists(scf, "ring.fadeColor") and
                self._param_exists(scf, "ring.fadeTime")):
            self.set_rgb(scf, r, g, b)
            return

        try:
            color_value = self.rgb_to_fadecolor_value(r, g, b)
            scf.cf.param.set_value("ring.fadeColor", str(color_value))
            scf.cf.param.set_value("ring.fadeTime",  str(float(fade_time)))
            scf.cf.param.set_value("ring.effect", "14")  # fade color mode
        except Exception as e:
            self.enabled[uri] = False   # suppress future calls after link loss
            print(f"[{uri}] Failed to fade RGB ({r},{g},{b}): {e}")

    def lights_off(self, scf):
        """Turn all LEDs off (black in solid color mode)."""
        uri = scf.cf.link_uri
        if not self.enabled.get(uri, False):
            return
        try:
            scf.cf.param.set_value("ring.solidRed",   "0")
            scf.cf.param.set_value("ring.solidGreen", "0")
            scf.cf.param.set_value("ring.solidBlue",  "0")
            scf.cf.param.set_value("ring.effect", "7")
        except Exception as e:
            self.enabled[uri] = False   # suppress future calls after link loss
            print(f"[{uri}] Failed to turn lights off: {e}")

    # -------------------------------------------------------------------------
    # Main lighting thread
    # -------------------------------------------------------------------------

    def run_emotion_sync(self, scf):
        uri = scf.cf.link_uri
        if not self.enabled.get(uri, False):
            print(f"[{uri}] EMOTION SYNC FAILED: LIGHTS NOT ENABLED")
            return

        while not self.stop_event.is_set():
            if self.sequence_start_event.wait(timeout=0.05):
                break
        if self.stop_event.is_set() or self.sequence_start_time is None:
            return

        sequence_start_time = self.sequence_start_time

        # --- Build merged timeline ---
        events = []
        for seg in self.segments:
            try:
                events.append({
                    "time":    float(seg["start"]),
                    "type":    "segment",
                    "weights": seg["weights"],
                })
            except (KeyError, TypeError, ValueError) as e:
                print(f"[{uri}] Bad segment skipped: {e}")

        for bt in self.beat_times:
            t = float(bt)
            events.append({"time": t,                        "type": "beat_on"})
            events.append({"time": t + self.beat_flash_s,    "type": "beat_off"})

        events.sort(key=lambda e: e["time"])

        current_rgb = (0, 0, 0)
        last_emotion_idx = None

        for event in events:
            if self.stop_event.is_set():
                break

            sleep_s = (sequence_start_time + event["time"]) - time.perf_counter()
            if sleep_s > 0:
                if self.stop_event.wait(timeout=sleep_s):
                    break

            etype = event["type"]

            if etype == "segment":
                idx = self.strongest_weight_index(event["weights"])
                r, g, b = self.emotion_index_to_rgb(idx)
                current_rgb = (r, g, b)
                if idx != last_emotion_idx:
                    self.set_fade_rgb(scf, r, g, b, fade_time=0.25)
                    print(f"[{uri}] Emotion → {idx} RGB({r},{g},{b}) "
                          f"@ {event['time']:.2f}s")
                    last_emotion_idx = idx

            elif etype == "beat_on":
                # White flash — always fires, no duplicate check needed
                self.set_rgb(scf, 255, 255, 255)

            elif etype == "beat_off":
                # Restore to current emotion color
                r, g, b = current_rgb
                self.set_rgb(scf, r, g, b)

        self.lights_off(scf)

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    def stop(self):
        """Signal all light threads to exit cleanly."""
        self.stop_event.set()
        self.sequence_start_event.set()   # unblock any thread still waiting
