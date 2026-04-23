import time
import csv
import os
import numpy as np
from collections import defaultdict
import traceback
import threading
import psutil
import statistics
from collections import deque

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.positioning.position_hl_commander import PositionHlCommander

import pygame

from lights import LightController

PATH = "trajectories.csv"
AUDIO_PATH = "audio/oldTownRoad.mp3"
WAYPOINT_LOG_PATH = "logs/actual_waypoints.log"
TAKEOFF_HEIGHT = 1.0
DEFAULT_VELOCITY = 0.3

X_WALL_MIN = -1.05
X_WALL_MAX = 1.05
Y_WALL_MIN = -1.05
Y_WALL_MAX = 1.05
Z_WALL_MIN = 0.5
Z_WALL_MAX = 1.2

uris = [
    'radio://0/80/2M/E7E7E7E702',
]

commanders = {}
loggers = {}
audio_started = threading.Event()
AUDIO_LEAD_S = 3.0
waypoint_log_lock = threading.Lock()
stop_log_lock = threading.Lock()

stop_event = threading.Event()
stop_logged_uris = set()

# Barrier used so every drone enters its waypoint loop at the same instant.
# Reset in __main__ once the number of drones is known.
# The barrier's action= callback fires after all threads arrive but before any
# thread is released, so _sequence_start_time is always written before it is read.
_takeoff_barrier: threading.Barrier | None = None
_sequence_start_time: float = 0.0  # perf_counter timestamp at barrier release

def _record_sequence_start():
    global _sequence_start_time
    _sequence_start_time = time.perf_counter() + AUDIO_LEAD_S


def init_waypoint_log():
    log_dir = os.path.dirname(WAYPOINT_LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with stop_log_lock:
        stop_logged_uris.clear()

    with waypoint_log_lock, open(WAYPOINT_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "uri", "wp_idx", "target_time_s", "duration_s",
            "measured_x", "measured_y", "measured_z",
            "nominal_x", "nominal_y", "nominal_z",
            "command_x", "command_y", "command_z",
            "event",
        ])


def log_waypoint(uri: str, idx: int, target_time: float, duration_s: float,
                 measured_pos: np.ndarray,
                 nominal_pos: tuple[float, float, float],
                 command_pos: tuple[float, float, float],
                 event: str = "waypoint"):
    with waypoint_log_lock, open(WAYPOINT_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            uri, idx, f"{target_time:.3f}", f"{duration_s:.3f}",
            f"{measured_pos[0]:.4f}", f"{measured_pos[1]:.4f}", f"{measured_pos[2]:.4f}",
            f"{nominal_pos[0]:.4f}", f"{nominal_pos[1]:.4f}", f"{nominal_pos[2]:.4f}",
            f"{command_pos[0]:.4f}", f"{command_pos[1]:.4f}", f"{command_pos[2]:.4f}",
            event,
        ])


def _get_measured_position(uri: str) -> np.ndarray:
    if cbf_filter is not None:
        return cbf_filter.get_position(uri)
    return np.array([0.0, 0.0, TAKEOFF_HEIGHT], dtype=float)


def _current_sequence_elapsed_s() -> float:
    if _sequence_start_time <= 0.0:
        return 0.0
    return max(0.0, time.perf_counter() - _sequence_start_time)


def log_stop_event(uri: str, idx: int,
                   nominal_pos: tuple[float, float, float] | None = None,
                   command_pos: tuple[float, float, float] | None = None,
                   event: str = "emergency_stop"):
    with stop_log_lock:
        if uri in stop_logged_uris:
            return
        stop_logged_uris.add(uri)

    measured_pos = _get_measured_position(uri)
    nominal = nominal_pos if nominal_pos is not None else tuple(measured_pos.tolist())
    command = command_pos if command_pos is not None else tuple(measured_pos.tolist())

    log_waypoint(
        uri,
        idx,
        _current_sequence_elapsed_s(),
        0.0,
        measured_pos,
        nominal,
        command,
        event=event,
    )

# ---------------------------------------------------------------------------
# CBF Performance Monitor
# ---------------------------------------------------------------------------

class CBFMonitor:
    """Lightweight rolling stats for CBF performance."""

    def __init__(self, window: int = 100):
        self._lock = threading.Lock()
        self._cbf_times:  dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._wait_times: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._wp_slips:   dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._last_log_time: dict[str, float] = {}

    def record_cbf(self, uri: str, wait_s: float, compute_s: float):
        with self._lock:
            self._cbf_times[uri].append(compute_s)
            self._wait_times[uri].append(wait_s)

    def record_waypoint_slip(self, uri: str, actual_dt: float, expected_dt: float):
        with self._lock:
            self._wp_slips[uri].append(actual_dt - expected_dt)

    def record_log_callback(self, uri: str):
        """Call at the top of every telemetry callback to detect silent drones."""
        now = time.perf_counter()
        with self._lock:
            last = self._last_log_time.get(uri)
            if last is not None:
                gap = now - last
                if gap > 0.05:   # >50 ms — drone has gone quiet
                    print(f'[MONITOR] {uri} telemetry gap {gap*1000:.1f} ms')
            self._last_log_time[uri] = now

    def report(self):
        proc    = psutil.Process()
        cpu_pct = psutil.cpu_percent(interval=None)
        mem_mb  = proc.memory_info().rss / 1e6
        threads = proc.num_threads()

        print(f'\n[MONITOR] System  cpu={cpu_pct:.1f}%  mem={mem_mb:.1f} MB'
              f'  threads={threads}')

        with self._lock:
            for uri in self._cbf_times:
                short = uri[-4:]
                ct = list(self._cbf_times[uri])
                wt = list(self._wait_times[uri])
                sl = list(self._wp_slips[uri])
                if not ct:
                    continue

                # Build slip portion separately so the ternary only controls
                # that segment, not the entire print statement.
                slip_str = (f'  wp_slip={statistics.mean(sl)*1e3:+.1f}ms'
                            if sl else '')

                print(
                    f'[MONITOR] {short}'
                    f'  cbf_compute={statistics.mean(ct)*1e3:.2f}ms'
                    f'(max {max(ct)*1e3:.2f})'
                    f'  lock_wait={statistics.mean(wt)*1e3:.2f}ms'
                    f'(max {max(wt)*1e3:.2f})'
                    + slip_str
                )


# ---------------------------------------------------------------------------
# Control Barrier Function (CBF) Safety Filter
# ---------------------------------------------------------------------------
#
# Safety constraint between drone i and drone j:
#   h(pᵢ, pⱼ) = ‖pᵢ − pⱼ‖² − d_safe²  ≥  0
#
# CBF condition (first-order):
#   ḣ + γ · h ≥ 0
#   2(pᵢ−pⱼ)·(vᵢ−vⱼ) + γ · (‖pᵢ−pⱼ‖² − d_safe²) ≥ 0
#
# Decentralised approximation: each drone treats the other's velocity as its
# last telemetry-measured velocity.  If the desired velocity vᵢ_des violates
# the CBF inequality, the minimal correction (1-D projection onto the
# constraint normal) is applied:
#
#   vᵢ_safe = vᵢ_des + λ · n̂ᵢⱼ    where  n̂ᵢⱼ = (pᵢ−pⱼ)/‖pᵢ−pⱼ‖
#   λ = deficit / (2 · ‖pᵢ−pⱼ‖)   where  deficit = cbf_rhs − ḣ_before
#
# The safe velocity (m/s) is converted back to a position target:
#   p_safe = pᵢ + vᵢ_safe · Δt
# ---------------------------------------------------------------------------

class CBFSafetyFilter:
    """
    Thread-safe CBF filter for a swarm of drones.

    Parameters
    ----------
    uris : list[str]
        Radio URIs of all drones in the swarm.
    d_safe : float
        Minimum allowed centre-to-centre separation (metres).
    gamma : float
        CBF decay rate γ — larger = more aggressive correction.
    dt : float
        Control timestep (seconds).  Must match the actual interval between
        successive go_to calls so that the velocity ↔ position conversion
        is dimensionally correct.
    """

    def __init__(self, uris: list[str], d_safe: float = 0.4,
                gamma: float = 0.5, dt: float = 0.2,
                z_floor: float = Z_WALL_MIN,
                bounds_lo: tuple = (X_WALL_MIN, Y_WALL_MIN, Z_WALL_MIN),
                bounds_hi: tuple = (X_WALL_MAX, Y_WALL_MAX, Z_WALL_MAX)):
        self.d_safe  = d_safe
        self.gamma   = gamma
        self.dt      = dt
        self.z_floor = z_floor
        self.uris    = list(uris)

        self._bounds_lo = np.array(list(bounds_lo))
        self._bounds_hi = np.array(list(bounds_hi))

        self._lock = threading.Lock()
        self._positions:  dict[str, np.ndarray] = {uri: np.zeros(3) for uri in self.uris}
        self._velocities: dict[str, np.ndarray] = {uri: np.zeros(3) for uri in self.uris}

    def update_state(self, uri: str, pos: np.ndarray,
                     vel: np.ndarray | None = None):
        """
        Ingest a fresh telemetry measurement for *uri*.

        Parameters
        ----------
        pos : array-like (3,) — [x, y, z] in metres
        vel : array-like (3,) or None — [vx, vy, vz] in m/s
        """
        with self._lock:
            self._positions[uri] = np.asarray(pos, dtype=float)
            if vel is not None:
                self._velocities[uri] = np.asarray(vel, dtype=float)

    def update_position(self, uri: str, pos: np.ndarray):
        """Convenience alias for update_state with position only."""
        self.update_state(uri, pos)

    def get_position(self, uri: str) -> np.ndarray:
        with self._lock:
            return self._positions[uri].copy()

    def filter(self, uri: str, x_des: float, y_des: float,
           z_des: float, dt: float | None = None) -> tuple[float, float, float]:
        """
        Return a CBF-safe target position for *uri* given its desired target.
        """
        t_wait_start = time.perf_counter()

        with self._lock:
            t_compute_start = time.perf_counter()
            dt_cmd = dt if dt is not None else self.dt

            p_i   = self._positions[uri].copy()
            p_des = np.array([x_des, y_des, z_des], dtype=float)

            v_des  = (p_des - p_i) / dt_cmd   # m/s
            v_safe = v_des.copy()

            for other_uri, p_j in self._positions.items():
                if other_uri == uri:
                    continue

                diff    = p_i - p_j
                dist_sq = float(diff @ diff)
                dist    = np.sqrt(dist_sq)
                h       = dist_sq - self.d_safe ** 2

                if dist < 1e-4:
                    v_safe += np.array([0.0, 0.0, 0.3])
                    print(f'[CBF] WARNING: {uri} and {other_uri} nearly coincident!')
                    continue

                v_j     = self._velocities[other_uri].copy()
                lie_h   = 2.0 * diff @ (v_safe - v_j)
                cbf_rhs = -self.gamma * h

                if lie_h < cbf_rhs:
                    n       = diff / dist
                    deficit = cbf_rhs - lie_h
                    lam     = deficit / (2.0 * dist)
                    v_safe  = v_safe + lam * n
                    print(
                        f'[CBF] {uri} ↔ {other_uri} | '
                        f'dist={dist:.3f}m  h={h:.3f}  correction λ={lam:.4f}'
                    )

            p_safe = p_i + v_safe * dt_cmd

            # ---- Floor clamp ------------------------------------------------
            if p_safe[2] < self.z_floor:
                print(f'[CBF] {uri} z-floor clamp: {p_safe[2]:.3f}m → {self.z_floor}m')
                p_safe[2] = self.z_floor
                self._velocities[uri][2] = (self.z_floor - p_i[2]) / dt_cmd

            # ---- Wall clamp -------------------------------------------------
            # The CBF repulsion vector can push p_safe outside room bounds.
            axis_names = ("x", "y", "z")
            for axis, lo, hi in zip(range(3), self._bounds_lo, self._bounds_hi):
                if p_safe[axis] < lo:
                    print(f'[CBF] {uri} {axis_names[axis]} lo-clamp: {p_safe[axis]:.3f}m → {lo}m')
                    p_safe[axis] = lo
                    self._velocities[uri][axis] = (lo - p_i[axis]) / dt_cmd
                elif p_safe[axis] > hi:
                    print(f'[CBF] {uri} {axis_names[axis]} hi-clamp: {p_safe[axis]:.3f}m → {hi}m')
                    p_safe[axis] = hi
                    self._velocities[uri][axis] = (hi - p_i[axis]) / dt_cmd
            # -----------------------------------------------------------------

            wait_s    = t_compute_start - t_wait_start
            compute_s = time.perf_counter() - t_compute_start

        if monitor is not None:
            monitor.record_cbf(uri, wait_s, compute_s)

        return float(p_safe[0]), float(p_safe[1]), float(p_safe[2])


# Shared instances — both created in __main__
cbf_filter: CBFSafetyFilter | None = None
monitor:    CBFMonitor | None      = None

#shared lighting controller created in main
light_controller: LightController | None = None

#keep one light thread per drone URI
light_threads = {}

# ---------------------------------------------------------------------------
# Telemetry logging
# ---------------------------------------------------------------------------

def setup_logging(scf: SyncCrazyflie):
    """
    Register a 10 ms state-estimate log and feed results into the CBF filter
    and the performance monitor.
    """
    uri = scf.cf.link_uri

    logconf = LogConfig(name='state', period_in_ms=10)
    logconf.add_variable('stateEstimate.x',  'FP16')
    logconf.add_variable('stateEstimate.y',  'FP16')
    logconf.add_variable('stateEstimate.z',  'FP16')
    logconf.add_variable('stateEstimate.vx', 'FP16')
    logconf.add_variable('stateEstimate.vy', 'FP16')
    logconf.add_variable('stateEstimate.vz', 'FP16')

    def _cb(_ts, data, _lc):
        if monitor is not None:
            monitor.record_log_callback(uri)

        if cbf_filter is not None:
            cbf_filter.update_state(
                uri,
                pos=np.array([data['stateEstimate.x'],
                               data['stateEstimate.y'],
                               data['stateEstimate.z']]),
                vel=np.array([data['stateEstimate.vx'],
                               data['stateEstimate.vy'],
                               data['stateEstimate.vz']]),
            )

    scf.cf.log.add_config(logconf)
    logconf.data_received_cb.add_callback(_cb)
    logconf.start()
    loggers[uri] = logconf
    print(f'[{uri}] Telemetry logging started')


def stop_logging(scf: SyncCrazyflie):
    uri = scf.cf.link_uri
    lc  = loggers.get(uri)
    if lc is not None:
        try:
            lc.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def setup_lights(scf: SyncCrazyflie):
    #initialize LED deck for one drone

    if light_controller is not None:
        light_controller.init_drone_lights(scf)
# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def play_audio(audio_file):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        audio_started.wait()
        print(f'[AUDIO] Starting playback: {audio_file}')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        print('[AUDIO] Playback finished')
    except Exception as e:
        print(f'[AUDIO] Error: {e}')


# ---------------------------------------------------------------------------
# CSV / waypoint loading
# ---------------------------------------------------------------------------

def read_csv(path):
    waypoint_map = defaultdict(list)
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            id_, t, x, y, z, vx, vy, vz, ax, ay, az = row
            waypoint_map[int(id_)].append((
                float(t),
                np.clip(float(x), -1.1, 1.1),
                np.clip(float(y), -1.1, 1.1),
                np.clip(float(z), Z_WALL_MIN, Z_WALL_MAX),
            ))

    for id_ in waypoint_map:
        waypoint_map[id_].sort(key=lambda p: p[0])

    print(f"Loaded waypoints for drone IDs: {list(waypoint_map.keys())}")
    for id_ in waypoint_map:
        print(f"  Drone {id_}: {len(waypoint_map[id_])} waypoints")

    return waypoint_map


def init_data(path):
    waypoints = read_csv(path)
    seq = {}
    for i, uri in enumerate(sorted(uris)):
        if i in waypoints:
            slice_ = waypoints[i][150:200]
            if slice_:
                t0 = slice_[0][0]  # time of first waypoint in slice
                # shift all times so the slice starts at t=0
                slice_ = [(t - t0, x, y, z) for (t, x, y, z) in slice_]
            seq[uri] = [slice_]
        else:
            print(f"WARNING: No waypoints found for drone ID {i} (URI: {uri})")
            seq[uri] = [[]]
    return seq


# ---------------------------------------------------------------------------
# Swarm callbacks
# ---------------------------------------------------------------------------

def make_commander(scf: SyncCrazyflie):
    uri = scf.cf.link_uri
    commander = PositionHlCommander(
        scf,
        default_height=TAKEOFF_HEIGHT,
        default_velocity=DEFAULT_VELOCITY,
        controller=PositionHlCommander.CONTROLLER_PID,
    )
    commanders[uri] = commander
    print(f'[{uri}] Commander created')


def fly_sequence(scf: SyncCrazyflie, sequence):
    uri       = scf.cf.link_uri
    commander = commanders.get(uri)
    if commander is None:
        raise RuntimeError(f"No commander for {uri} — did make_commander run?")

    print(f'[{uri}] Taking off...')
    commander.take_off(height=TAKEOFF_HEIGHT, velocity=DEFAULT_VELOCITY)
    time.sleep(1.0)

    if cbf_filter is not None and uri not in loggers:
        cbf_filter.update_position(uri, np.array([0.0, 0.0, TAKEOFF_HEIGHT]))

    # Trigger audio before the barrier so music plays during the lead window
    if uri == sorted(uris)[0]:
        print(f'[{uri}] Triggering audio playback ({AUDIO_LEAD_S}s lead)')
        audio_started.set()

    print(f'[{uri}] Airborne — waiting at barrier')

    global _sequence_start_time
    
    if _takeoff_barrier is not None:
        try:
            _takeoff_barrier.wait(timeout=15.0)
            light_controller.set_sequence_start(_sequence_start_time)
            
        except threading.BrokenBarrierError:
            print(f'[{uri}] Barrier broken — aborting')
            return
        
        light_controller.set_sequence_start(_sequence_start_time)
        if uri not in light_threads:
            t = threading.Thread(
                target=light_controller.run_emotion_sync,
                args=(scf,),
                daemon=True
            )
            light_threads[uri] = t
            t.start()

    if not sequence:
        print(f'[{uri}] No waypoints — hovering briefly')
        if stop_event.wait(timeout=2.0):
            log_stop_event(uri, -1, event="stop_while_hovering")
    else:
        print(f'[{uri}] Running {len(sequence)} waypoints (CBF active)...')
        for idx, (target_time, x_des, y_des, z_des) in enumerate(sequence):

            if stop_event.is_set():
                print(f'[{uri}] Stop event — aborting sequence at wp {idx}')
                log_stop_event(
                    uri,
                    idx,
                    nominal_pos=(x_des, y_des, z_des),
                    event="stop_before_waypoint",
                )
                break

            # ---- Wait until this waypoint's scheduled time ----------------
            # target_time is in seconds relative to sequence start (t=0).
            # Sleep until that wall-clock moment so all drones stay in step.
            now = time.perf_counter()
            scheduled_at = _sequence_start_time + target_time
            sleep_s = scheduled_at - now
            if sleep_s > 0:
                if stop_event.wait(timeout=sleep_s):   # wakes early on stop
                    print(f'[{uri}] Stop event during wait — aborting at wp {idx}')
                    log_stop_event(
                        uri,
                        idx,
                        nominal_pos=(x_des, y_des, z_des),
                        event="stop_during_wait",
                    )
                    break
            elif sleep_s < -0.1:
                print(f'[{uri}] wp {idx} late by {-sleep_s*1000:.1f} ms')
            # ---------------------------------------------------------------

            measured_pos = (
                cbf_filter.get_position(uri)
                if cbf_filter is not None else
                np.array([0.0, 0.0, TAKEOFF_HEIGHT], dtype=float)
            )

            if idx + 1 < len(sequence):
                duration_s = sequence[idx + 1][0] - target_time
            else:
                duration_s = cbf_filter.dt if cbf_filter else 0.5
            duration_s = max(0.1, duration_s)   # never send a zero-duration cmd

            # ---- CBF safety filter ----------------------------------------
            if cbf_filter is not None:
                x_safe, y_safe, z_safe = cbf_filter.filter(
                    uri, x_des, y_des, z_des, dt=duration_s
                )
                if (abs(x_safe - x_des) > 0.01 or
                        abs(y_safe - y_des) > 0.01 or
                        abs(z_safe - z_des) > 0.01):
                    print(
                        f'[CBF] {uri} wp{idx}: '
                        f'({x_des:.2f},{y_des:.2f},{z_des:.2f}) → '
                        f'({x_safe:.2f},{y_safe:.2f},{z_safe:.2f})'
                    )
            else:
                x_safe, y_safe, z_safe = x_des, y_des, z_des
            # ---------------------------------------------------------------

            if idx % 10 == 0:
                print(f'[{uri}] wp {idx}/{len(sequence)} → '
                      f'({x_safe:.2f}, {y_safe:.2f}, {z_safe:.2f})')

            # ---- Non-blocking go_to with explicit duration --------------------
            # PositionHlCommander.go_to() computes travel time from velocity
            # and then *sleeps* for that duration, which blocks this thread and
            # causes per-drone skew that compounds over a long sequence.
            #
            # Instead, use the low-level high_level_commander directly — it
            # sends the setpoint packet and returns immediately.  The drone
            # executes the trajectory in firmware; the outer absolute-time
            # sleep above is already handling the inter-waypoint pacing on the
            # Python side.
            #
            # duration_s: how long the drone should take to reach this target.
            # Set it to the gap until the *next* waypoint so the firmware
            # trajectory matches the schedule.  For the final waypoint use
            # cbf_filter.dt (or 0.5 s) as a sensible hover duration.
            log_waypoint(
                uri,
                idx,
                target_time,
                duration_s,
                measured_pos,
                (x_des, y_des, z_des),
                (x_safe, y_safe, z_safe),
            )

            t_wp_start = time.perf_counter()
            scf.cf.high_level_commander.go_to(
                x_safe, y_safe, z_safe, 0.0, duration_s
            )
            actual_dt = time.perf_counter() - t_wp_start   # should be ~packet RTT
            if monitor is not None:
                monitor.record_waypoint_slip(
                    uri, actual_dt, duration_s
                )

            # Periodic report every 20 waypoints
            if monitor is not None and idx % 20 == 0 and idx > 0:
                monitor.report()

        print(f'[{uri}] Sequence complete')

    # Signal light thread to stop cleanly before landing begins
    if light_controller is not None:
        light_controller.stop()                     # sets light_controller.stop_event

    t_light = light_threads.get(uri)
    if t_light is not None:
        t_light.join(timeout=2.0)                   # wait up to 2s for clean exit
        if t_light.is_alive():
            print(f'[{uri}] Light thread still alive after join — proceeding anyway')

    print(f'[{uri}] Landing...')
    commander.land(velocity=DEFAULT_VELOCITY)

    #turn lightd off after landing
    if light_controller is not None:
        light_controller.lights_off(scf)


    # Only the first drone handles audio fadeout to avoid multiple threads
    # calling stop simultaneously.
    if uri == sorted(uris)[0]:
        try:
            pygame.mixer.music.fadeout(2000)
        except Exception:
            pass


def emergency_land(scf: SyncCrazyflie):
    """
    Best-effort emergency land.  Tries the PositionHlCommander first, falls
    back to the raw high-level commander.
    """
    uri = scf.cf.link_uri
    print(f'[{uri}] EMERGENCY LAND')
    log_stop_event(uri, -1, event="emergency_land")

    #stop lighting if emergency landing enacted
    if light_controller is not None:
        light_controller.stop()

    try:
        commander = commanders.get(uri)
        if commander is not None:
            commander.land(velocity=DEFAULT_VELOCITY)
        else:
            scf.cf.high_level_commander.land(0.0, 2.0)
        time.sleep(2.5)
    except Exception as e:
        print(f'[{uri}] Emergency land failed: {e}')
        try:
            scf.cf.high_level_commander.land(0.0, 2.0)
            time.sleep(2.5)
        except Exception:
            pass

    if light_controller is not None:
        light_controller.lights_off(scf)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    seq_args = init_data(PATH)
    init_waypoint_log()
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')

    monitor = CBFMonitor(window=100)
    print('[MONITOR] Initialised')

    cbf_filter = CBFSafetyFilter(uris, d_safe=0.5, gamma=0.5, dt=0.2, z_floor=Z_WALL_MIN)
    print(
        f'[CBF] Initialised | d_safe={cbf_filter.d_safe}m  γ={cbf_filter.gamma}  '
        f'z_floor={cbf_filter.z_floor}m'
    )

    #create shared light controller and load clap weights
    light_controller = LightController()
    light_controller.load_weights("json/clap_weights.json")
    light_controller.load_beat_times("json/beat_times.json")
    print('[LIGHTS] Initialized')

    _takeoff_barrier = threading.Barrier(len(uris), action=_record_sequence_start)
    print(f'[SYNC] Takeoff barrier initialised for {len(uris)} drones')

    audio_thread = threading.Thread(
        target=play_audio, args=(AUDIO_PATH,), daemon=True
    )
    audio_thread.start()

    with Swarm(uris, factory=factory) as swarm:
        try:
            print('Resetting estimators...')
            swarm.reset_estimators()
            time.sleep(2)

            print('Starting telemetry logs...')
            swarm.parallel_safe(setup_logging)
            time.sleep(0.5)

            print('Creating commanders...')
            swarm.parallel_safe(make_commander)

            print('Initializing lights...')
            swarm.parallel_safe(setup_lights)

            print('Flying sequences...')
            swarm.parallel_safe(fly_sequence, args_dict=seq_args)

            audio_thread.join(timeout=250)

        except (KeyboardInterrupt, Exception) as e:
            if isinstance(e, KeyboardInterrupt):
                print('\n[MAIN] KeyboardInterrupt — initiating safe shutdown')
            else:
                print(f'[MAIN] Swarm error: {e}')
                traceback.print_exc()

            stop_event.set()

            if light_controller is not None:
                light_controller.stop()

            time.sleep(1.5)

            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

            swarm.parallel_safe(emergency_land)

        finally:
            if monitor is not None:
                print('\n--- Final performance report ---')
                monitor.report()
            swarm.parallel_safe(stop_logging)
