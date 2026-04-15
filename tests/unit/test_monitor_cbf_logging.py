import csv
import importlib
import sys
import threading
import tempfile
import time as real_time
import types
import unittest
from unittest import mock

import numpy as np


def load_monitor_module():
    sys.modules.pop("monitor_CBF", None)

    cflib = types.ModuleType("cflib")
    cflib.crtp = types.ModuleType("cflib.crtp")
    cflib.crtp.init_drivers = lambda: None

    crazyflie = types.ModuleType("cflib.crazyflie")
    log_mod = types.ModuleType("cflib.crazyflie.log")
    sync_mod = types.ModuleType("cflib.crazyflie.syncCrazyflie")
    swarm_mod = types.ModuleType("cflib.crazyflie.swarm")
    positioning = types.ModuleType("cflib.positioning")
    pos_cmd_mod = types.ModuleType("cflib.positioning.position_hl_commander")

    class DummyLogConfig:
        def __init__(self, *args, **kwargs):
            self.data_received_cb = types.SimpleNamespace(add_callback=lambda cb: None)

        def add_variable(self, *args, **kwargs):
            return None

        def start(self):
            return None

        def stop(self):
            return None

    class DummySyncCrazyflie:
        pass

    class DummyCachedCfFactory:
        def __init__(self, *args, **kwargs):
            pass

    class DummySwarm:
        def __init__(self, *args, **kwargs):
            pass

    class DummyPositionHlCommander:
        CONTROLLER_PID = 1

        def __init__(self, *args, **kwargs):
            pass

        def take_off(self, *args, **kwargs):
            return None

        def land(self, *args, **kwargs):
            return None

    log_mod.LogConfig = DummyLogConfig
    sync_mod.SyncCrazyflie = DummySyncCrazyflie
    swarm_mod.CachedCfFactory = DummyCachedCfFactory
    swarm_mod.Swarm = DummySwarm
    pos_cmd_mod.PositionHlCommander = DummyPositionHlCommander

    pygame = types.ModuleType("pygame")
    pygame.mixer = types.SimpleNamespace(
        init=lambda: None,
        music=types.SimpleNamespace(
            load=lambda *args, **kwargs: None,
            play=lambda *args, **kwargs: None,
            get_busy=lambda: False,
            stop=lambda *args, **kwargs: None,
            fadeout=lambda *args, **kwargs: None,
        ),
    )

    sys.modules["cflib"] = cflib
    sys.modules["cflib.crtp"] = cflib.crtp
    sys.modules["cflib.crazyflie"] = crazyflie
    sys.modules["cflib.crazyflie.log"] = log_mod
    sys.modules["cflib.crazyflie.syncCrazyflie"] = sync_mod
    sys.modules["cflib.crazyflie.swarm"] = swarm_mod
    sys.modules["cflib.positioning"] = positioning
    sys.modules["cflib.positioning.position_hl_commander"] = pos_cmd_mod
    sys.modules["pygame"] = pygame

    return importlib.import_module("monitor_CBF")


class FakeCommander:
    def __init__(self):
        self.takeoff_calls = []
        self.land_calls = []

    def take_off(self, height, velocity):
        self.takeoff_calls.append((height, velocity))

    def land(self, velocity):
        self.land_calls.append(velocity)


class FakeHighLevelCommander:
    def __init__(self):
        self.go_to_calls = []
        self.land_calls = []

    def go_to(self, x, y, z, yaw, duration_s):
        self.go_to_calls.append((x, y, z, yaw, duration_s))

    def land(self, z, duration_s):
        self.land_calls.append((z, duration_s))


class FakeCf:
    def __init__(self, uri):
        self.link_uri = uri
        self.high_level_commander = FakeHighLevelCommander()


class FakeScf:
    def __init__(self, uri):
        self.cf = FakeCf(uri)


class FakeCBF:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.dt = 0.2

    def get_position(self, uri):
        return self.position.copy()

    def update_position(self, uri, pos):
        self.position = np.asarray(pos, dtype=float)

    def filter(self, uri, x_des, y_des, z_des, dt=None):
        return x_des, y_des, z_des


class StopDuringWaitEvent:
    def __init__(self):
        self._is_set = False

    def is_set(self):
        return self._is_set

    def set(self):
        self._is_set = True

    def wait(self, timeout=None):
        if timeout and timeout > 0:
            self._is_set = True
            return True
        return self._is_set


class MonitorWaypointLoggingTests(unittest.TestCase):
    def test_fly_sequence_logs_stop_during_wait(self):
        monitor = load_monitor_module()
        uri = "radio://test"

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/actual_waypoints.log"
            monitor.WAYPOINT_LOG_PATH = log_path
            monitor.init_waypoint_log()
            monitor.commanders.clear()
            monitor.loggers.clear()
            monitor.light_threads.clear()
            monitor.stop_event = StopDuringWaitEvent()
            monitor.audio_started = threading.Event()
            monitor.uris = [uri]
            monitor._takeoff_barrier = None
            monitor._sequence_start_time = real_time.perf_counter() + 10.0
            monitor.cbf_filter = FakeCBF([0.0, 0.0, monitor.TAKEOFF_HEIGHT])
            monitor.light_controller = None
            monitor.monitor = None

            commander = FakeCommander()
            monitor.commanders[uri] = commander
            scf = FakeScf(uri)

            with mock.patch.object(monitor.time, "sleep", return_value=None):
                monitor.fly_sequence(scf, [(0.0, 0.1, 0.2, monitor.TAKEOFF_HEIGHT)])

            with open(log_path, newline="") as f:
                rows = list(csv.reader(f))

        self.assertEqual(rows[0][-1], "event")
        self.assertEqual(rows[1][0], uri)
        self.assertEqual(rows[1][1], "0")
        self.assertEqual(
            rows[1][7:10],
            ["0.1000", "0.2000", f"{monitor.TAKEOFF_HEIGHT:.4f}"],
        )
        self.assertEqual(rows[1][-1], "stop_during_wait")
        self.assertEqual(scf.cf.high_level_commander.go_to_calls, [])
        self.assertEqual(commander.land_calls, [monitor.DEFAULT_VELOCITY])

    def test_emergency_land_logs_snapshot_when_no_waypoint_was_sent(self):
        monitor = load_monitor_module()
        uri = "radio://test"

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/actual_waypoints.log"
            monitor.WAYPOINT_LOG_PATH = log_path
            monitor.init_waypoint_log()
            monitor.cbf_filter = FakeCBF([0.2, -0.1, 0.8])
            monitor.light_controller = None
            monitor.commanders.clear()

            scf = FakeScf(uri)

            with mock.patch.object(monitor.time, "sleep", return_value=None):
                monitor.emergency_land(scf)

            with open(log_path, newline="") as f:
                rows = list(csv.reader(f))

        self.assertEqual(rows[1][0], uri)
        self.assertEqual(rows[1][1], "-1")
        self.assertEqual(rows[1][4:7], ["0.2000", "-0.1000", "0.8000"])
        self.assertEqual(rows[1][-1], "emergency_land")
        self.assertEqual(scf.cf.high_level_commander.land_calls, [(0.0, 2.0)])


if __name__ == "__main__":
    unittest.main()
