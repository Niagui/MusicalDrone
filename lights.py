## lights.py
## This file handles all LED light behavior for drones
## 1. Loads timed emotion weights from clap_weights.json
## 2. waits for share sequences start time from monitor_CBF.py and finds strongest emotion weight from each
## 3. Switches drone LED lights base on strongest weight

import json
import time
import threading

class LightController:

    def __init__(self):
        # Tracks whether lighting was successfully initialized per drone
        self.enabled = {}

        # Used to stop the light thread in case of shutdown/emergency
        self.stop_event = threading.Event()

        # Signals that the shared sequence start time has been published.
        self.sequence_start_event = threading.Event()

        # Shared show start time, set by monitor_CBF.py after takeoff barrier
        self.sequence_start_time = None

        # Loads JSON segments from clap_weights.json
        self.segments = []

    def load_weights(self, path="json/clap_weights.json"):

        with open(path, "r") as f:
            self.segments = json.load(f)

        print(f"[LIGHTS] Loaded {len(self.segments)}")

    def init_drone_lights(self, scf):
        #initializes LED deck for one drone

        uri = scf.cf.link_uri
        try:
            #ring.effect = 0 is typically "off" or base mode
            scf.cf.param.set_value("ring.effect", "0")
            self.enabled[uri] = True
            print(f"[{uri}] Lights initialized")
        except Exception as e:
            self.enabled[uri] = False
            print(f"[{uri}] Light init failed: {e}")

    def set_sequence_start(self, t0):
        #drones use same start time so light changes are synched
        self.sequence_start_time = t0
        self.sequence_start_event.set()

    def set_sequence(self, t0):
        # Backward-compatible alias for older callers.
        self.set_sequence_start(t0)

    def strongest_weight_index(self, weights):
        #returns largest weight index, returns value 0-6
        return max(range(len(weights)), key=lambda i: weights[i])
    
    def weight_index_to_effect(self, idx):
        #map weight index to LED effect ID
        effect_map = {
            0:1,
            1:2,
            2:3,
            3:4,
            4:5,
            5:6,
            6:7,
        }
        return effect_map.get(idx, 0)
    
    def set_effect(self, scf, effect_id):
        #send light command to one drone
        uri = scf.cf.link_uri

        #skip if lights weren't initialized successfully
        if not self.enabled.get(uri, False):
            return

        try:
            scf.cf.param.set_value("ring.effect", str(effect_id))
        except Exception as e:
            print(f"[{uri}] Failed to set effect {effect_id}: {e}")

    def run_emotion_sync(self, scf):
        #lighting thread for one drone

        uri = scf.cf.link_uri

        #do nothing if night init failed
        if not self.enabled.get(uri, False):
            return
        
        while not self.stop_event.is_set():
            if self.sequence_start_event.wait(timeout=0.05):
                break

        if self.stop_event.is_set() or self.sequence_start_time is None:
            return

        sequence_start_time = self.sequence_start_time
        
        #track dominant weight index so light only changes when winning emotion changes
        last_idx = None

        for seg in self.segments:
            if self.stop_event.is_set():
                break
            
            try:
                start = float(seg["start"])
                weights = seg["weights"] 
            except (KeyError, TypeError, ValueError) as e:
                print(f"[{uri}] Bad light segment skipped: {e}")
                continue

            #sleep until segments begins
            scheduled = sequence_start_time + start
            sleep_s = scheduled - time.perf_counter()

            if sleep_s > 0:
                if self.stop_event.wait(timeout=sleep_s):
                    break

            #find which weight is strongest for segment
            idx = self.strongest_weight_index(weights)

            #wait to update light effect until strongest emotion weight changes
            if idx != last_idx:
                effect_id = self.weight_index_to_effect(idx)
                self.set_effect(scf, effect_id)
                print(
                    f"[{uri}] Light change at {start:.2f}s: "
                    f"dominant weight index {idx} -> effect {effect_id}"
                )
                last_idx = idx

            #turn lights off when done
            self.set_effect(scf, 0)

    def lights_off(self, scf):
        #turn lights off
        self.set_effect(scf, 0)

    def stop(self):
        #stop all light threads
        self.stop_event.set()
        self.sequence_start_event.set()
