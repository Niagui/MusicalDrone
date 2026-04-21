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

        # RGB color per emotion index (0-6)
        self.emotion_colors = {
            0: (255,  80,  80),   # red
            1: (255, 170,  60),   # orange
            2: (255, 230,  80),   # yellow
            3: ( 80, 255, 120),   # green
            4: ( 80, 170, 255),   # blue
            5: (170,  80, 255),   # purple
            6: (255,  80, 200),   # pink
        }

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def load_weights(self, path="json/clap_weights.json"):
        with open(path, "r") as f:
            self.segments = json.load(f)
        print(f"[LIGHTS] Loaded {len(self.segments)}")

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
            scf.cf.param.set_value("ring.solidRed",   "0")
            scf.cf.param.set_value("ring.solidGreen", "0")
            scf.cf.param.set_value("ring.solidBlue",  "0")
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
        """Lighting thread for one drone: follows emotion weights in real time."""
        uri = scf.cf.link_uri

        if not self.enabled.get(uri, False):
            return

        # Wait until the sequence start time is published (or stop is requested)
        while not self.stop_event.is_set():
            if self.sequence_start_event.wait(timeout=0.05):
                break

        if self.stop_event.is_set() or self.sequence_start_time is None:
            return

        sequence_start_time = self.sequence_start_time
        last_idx = None

        for seg in self.segments:
            if self.stop_event.is_set():
                break

            try:
                start   = float(seg["start"])
                weights = seg["weights"]
            except (KeyError, TypeError, ValueError) as e:
                print(f"[{uri}] Bad light segment skipped: {e}")
                continue

            # Sleep until this segment's scheduled wall-clock time
            sleep_s = (sequence_start_time + start) - time.perf_counter()
            if sleep_s > 0:
                if self.stop_event.wait(timeout=sleep_s):
                    break

            idx = self.strongest_weight_index(weights)

            # Only send a command when the dominant emotion actually changes
            if idx != last_idx:
                r, g, b = self.emotion_index_to_rgb(idx)
                # self.set_rgb(scf, r, g, b)        # hard cut
                self.set_fade_rgb(scf, r, g, b)     # smooth fade
                print(f"[{uri}] Light change at {start:.2f}s: "
                      f"emotion {idx} -> RGB ({r},{g},{b})")
                last_idx = idx

        # Sequence finished (or stop_event set) — turn off LEDs
        self.lights_off(scf)

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    def stop(self):
        """Signal all light threads to exit cleanly."""
        self.stop_event.set()
        self.sequence_start_event.set()   # unblock any thread still waiting