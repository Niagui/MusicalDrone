import time
import csv
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

PATH = "trajectories.csv"
AUDIO_PATH = "audio/testSong.mp3"
TAKEOFF_HEIGHT = 1.0
DEFAULT_VELOCITY = 0.5

uris = [
    'radio://0/80/2M/E7E7E7E701',
    'radio://0/80/2M/E7E7E7E702',
]

commanders = {}
loggers = {}
audio_started = threading.Event()

stop_event = threading.Event()

# Barrier used so every drone enters its waypoint loop at the same instant.
# Reset in __main__ once the number of drones is known.
_takeoff_barrier: threading.Barrier | None = None
_sequence_start_time: float = 0.0  # perf_counter timestamp at barrier release


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
                 gamma: float = 2.0, dt: float = 0.5,
                 z_floor: float = 0.3):
        self.d_safe  = d_safe
        self.gamma   = gamma
        self.dt      = dt
        self.z_floor = z_floor   # metres — safe position output never goes below this
        self.uris    = list(uris)

        self._lock = threading.Lock()

        self._positions: dict[str, np.ndarray] = {
            uri: np.zeros(3) for uri in self.uris
        }
        self._velocities: dict[str, np.ndarray] = {
            uri: np.zeros(3) for uri in self.uris
        }

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

    def filter(self, uri: str, x_des: float, y_des: float,
               z_des: float) -> tuple[float, float, float]:
        """
        Return a CBF-safe target position for *uri* given its desired target.
        """
        t_wait_start = time.perf_counter()

        with self._lock:
            t_compute_start = time.perf_counter()

            p_i   = self._positions[uri].copy()
            p_des = np.array([x_des, y_des, z_des], dtype=float)

            v_des  = (p_des - p_i) / self.dt   # m/s
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

            self._velocities[uri] = v_safe.copy()
            p_safe = p_i + v_safe * self.dt

            # ---- Floor clamp -------------------------------------------------
            # The 3-D repulsion vector can push z downward when another drone
            # is above this one.  Enforce a hard minimum altitude so the drone
            # is never commanded into the ground.
            if p_safe[2] < self.z_floor:
                print(f'[CBF] {uri} z-floor clamp: {p_safe[2]:.3f}m → {self.z_floor}m')
                p_safe[2] = self.z_floor
                # Back-propagate so stored velocity stays consistent.
                self._velocities[uri][2] = (self.z_floor - p_i[2]) / self.dt
            # ------------------------------------------------------------------

            # Compute timings inside the lock so both values are always
            # defined together before record_cbf is called.
            wait_s    = t_compute_start - t_wait_start
            compute_s = time.perf_counter() - t_compute_start

        # Call record_cbf outside the CBF lock to avoid holding it longer
        # than necessary.
        if monitor is not None:
            monitor.record_cbf(uri, wait_s, compute_s)

        return float(p_safe[0]), float(p_safe[1]), float(p_safe[2])


# Shared instances — both created in __main__
cbf_filter: CBFSafetyFilter | None = None
monitor:    CBFMonitor | None      = None


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
                np.clip(float(y), -1.8, 1.8),
                np.clip(float(z), 0.1,  1.2),
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
            seq[uri] = [waypoints[i][:50]]
            print(f"Assigned {len(seq[uri][0])} waypoints to {uri} (drone ID {i})")
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
    """
    Take off, execute CBF-filtered waypoint sequence, then land.

    Checks stop_event between every waypoint so that a Ctrl+C in the main
    thread can signal all fly_sequence threads to abort cleanly before
    emergency_land is called.
    """
    uri       = scf.cf.link_uri
    commander = commanders.get(uri)
    if commander is None:
        raise RuntimeError(f"No commander for {uri} — did make_commander run?")

    print(f'[{uri}] Taking off...')
    commander.take_off(height=TAKEOFF_HEIGHT, velocity=DEFAULT_VELOCITY)
    time.sleep(1.0)

    if cbf_filter is not None and uri not in loggers:
        cbf_filter.update_position(uri, np.array([0.0, 0.0, TAKEOFF_HEIGHT]))

    print(f'[{uri}] Airborne — waiting at barrier')

    # ------------------------------------------------------------------
    # Synchronisation barrier: every drone waits here until ALL drones
    # are airborne, then they all start counting time from the same t=0.
    # The first thread through records the shared start time.
    # ------------------------------------------------------------------
    global _sequence_start_time
    if _takeoff_barrier is not None:
        try:
            idx_at_barrier = _takeoff_barrier.wait(timeout=15.0)
            if idx_at_barrier == 0:                   # first thread through
                _sequence_start_time = time.perf_counter()
        except threading.BrokenBarrierError:
            print(f'[{uri}] Barrier broken — aborting')
            return

    # Only the first (alphabetically) drone triggers audio so that it
    # starts in sync with the sequence.
    if uri == sorted(uris)[0]:
        print(f'[{uri}] Triggering audio playback')
        audio_started.set()

    if not sequence:
        print(f'[{uri}] No waypoints — hovering briefly')
        stop_event.wait(timeout=2.0)
    else:
        print(f'[{uri}] Running {len(sequence)} waypoints (CBF active)...')
        for idx, (target_time, x_des, y_des, z_des) in enumerate(sequence):

            if stop_event.is_set():
                print(f'[{uri}] Stop event — aborting sequence at wp {idx}')
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
                    break
            elif sleep_s < -0.1:
                print(f'[{uri}] wp {idx} late by {-sleep_s*1000:.1f} ms')
            # ---------------------------------------------------------------

            # ---- CBF safety filter ----------------------------------------
            if cbf_filter is not None:
                x_safe, y_safe, z_safe = cbf_filter.filter(
                    uri, x_des, y_des, z_des
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

            # Time the actual go_to duration and record slip against expected dt.
            t_wp_start = time.perf_counter()
            commander.go_to(x_safe, y_safe, z_safe, velocity=DEFAULT_VELOCITY)
            actual_dt = time.perf_counter() - t_wp_start
            if monitor is not None:
                monitor.record_waypoint_slip(
                    uri, actual_dt, cbf_filter.dt if cbf_filter else 0.5
                )

            # Periodic report every 20 waypoints
            if monitor is not None and idx % 20 == 0 and idx > 0:
                monitor.report()

        print(f'[{uri}] Sequence complete')

    print(f'[{uri}] Landing...')
    commander.land(velocity=DEFAULT_VELOCITY)
    print(f'[{uri}] Landed')

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    seq_args = init_data(PATH)
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')

    monitor = CBFMonitor(window=100)
    print('[MONITOR] Initialised')

    cbf_filter = CBFSafetyFilter(uris, d_safe=0.5, gamma=2.0, dt=0.5, z_floor=0.3)
    print(f'[CBF] Initialised | d_safe={cbf_filter.d_safe}m  γ={cbf_filter.gamma}  z_floor={cbf_filter.z_floor}m')

    _takeoff_barrier = threading.Barrier(len(uris))
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