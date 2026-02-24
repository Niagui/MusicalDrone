import time
import csv
import numpy as np
from collections import defaultdict
import traceback
import threading

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

# Set by the main thread when a Ctrl+C or fatal error occurs.
# fly_sequence checks this between waypoints to exit cleanly before
# emergency_land is issued — avoiding the parallel_safe conflict.
stop_event = threading.Event()


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

    def __init__(self, uris: list[str], d_safe: float = 0.5,
                 gamma: float = 1.0, dt: float = 0.5):
        self.d_safe = d_safe
        self.gamma = gamma
        self.dt = dt
        self.uris = list(uris)

        self._lock = threading.Lock()

        # Best-known position of every drone, updated from telemetry callbacks.
        # Initialised to zero; call update_position() as soon as you have
        # a real estimate (e.g. immediately after take_off).
        self._positions: dict[str, np.ndarray] = {
            uri: np.zeros(3) for uri in self.uris
        }

        # Latest MEASURED velocity (m/s) of every drone, updated from the
        # state-estimate log callback.  Used as the "other" drone's velocity
        # in the decentralised CBF Lie-derivative term.
        self._velocities: dict[str, np.ndarray] = {
            uri: np.zeros(3) for uri in self.uris
        }

    # ------------------------------------------------------------------
    # Public helpers called from outside the filter
    # ------------------------------------------------------------------

    def update_state(self, uri: str, pos: np.ndarray,
                     vel: np.ndarray | None = None):
        """
        Ingest a fresh telemetry measurement for *uri*.

        Call this from the log callback (10 ms cadence) so that the filter
        always reasons about real positions rather than commanded targets.

        Parameters
        ----------
        uri : str
        pos : array-like, shape (3,) — [x, y, z] in metres
        vel : array-like, shape (3,) or None — [vx, vy, vz] in m/s.
              Pass None if velocity is not available; the filter will keep
              its previous estimate.
        """
        with self._lock:
            self._positions[uri] = np.asarray(pos, dtype=float)
            if vel is not None:
                self._velocities[uri] = np.asarray(vel, dtype=float)

    # Convenience alias kept for backwards-compatibility
    def update_position(self, uri: str, pos: np.ndarray):
        self.update_state(uri, pos)

    def filter(self, uri: str, x_des: float, y_des: float,
               z_des: float) -> tuple[float, float, float]:
        """
        Return a CBF-safe target position for *uri* given its desired target.

        Internally works in velocity space (m/s), then converts back to a
        position command via p_safe = p_i + v_safe * dt.

        Parameters
        ----------
        uri      : URI of the drone being commanded.
        x_des, y_des, z_des : desired target position in metres.

        Returns
        -------
        (x_safe, y_safe, z_safe) — position target that satisfies all
        pairwise CBF constraints.
        """
        with self._lock:
            p_i = self._positions[uri].copy()
            p_des = np.array([x_des, y_des, z_des], dtype=float)

            # FIX: divide by dt to get a true velocity in m/s.
            # The original code used the raw displacement (metres) as if it
            # were a velocity, making the p_safe = p_i + v * dt conversion
            # only reach halfway to the target each step.
            v_des = (p_des - p_i) / self.dt   # m/s

            v_safe = v_des.copy()

            for other_uri, p_j in self._positions.items():
                if other_uri == uri:
                    continue

                diff = p_i - p_j          # vector from j → i
                dist_sq = float(diff @ diff)
                dist = np.sqrt(dist_sq)

                h = dist_sq - self.d_safe ** 2

                if dist < 1e-4:
                    # Drones nearly coincident — push upward to recover
                    v_safe += np.array([0.0, 0.0, 0.3])
                    print(f'[CBF] WARNING: {uri} and {other_uri} nearly coincident!')
                    continue

                # FIX: use the MEASURED velocity of the other drone (set by
                # update_state from telemetry), not a stale command-derived
                # displacement.  In the original code _velocities stored the
                # previous v_safe of that drone (a displacement, not m/s),
                # which made the Lie-derivative term dimensionally wrong.
                v_j = self._velocities[other_uri].copy()   # m/s

                # CBF Lie derivative: ḣ = 2(pᵢ−pⱼ)·(vᵢ−vⱼ)
                lie_h = 2.0 * diff @ (v_safe - v_j)
                cbf_rhs = -self.gamma * h     # required lower bound on ḣ

                if lie_h < cbf_rhs:
                    # Constraint violated — compute minimal correction along
                    # the outward unit normal n̂ = diff / ‖diff‖
                    n = diff / dist

                    deficit = cbf_rhs - lie_h
                    # Δ(lie_h) = 2 * diff · (λ * n) = 2 * ‖diff‖ * λ
                    lam = deficit / (2.0 * dist)
                    v_safe = v_safe + lam * n

                    print(
                        f'[CBF] {uri} ↔ {other_uri} | '
                        f'dist={dist:.3f}m  h={h:.3f}  correction λ={lam:.4f}'
                    )

            # FIX: store the corrected velocity in m/s (not displacement).
            # This is what the next filter() call will read back as v_j for
            # THIS drone when another drone queries it.
            self._velocities[uri] = v_safe.copy()   # m/s

            # Convert safe velocity back to a position target
            p_safe = p_i + v_safe * self.dt

        return float(p_safe[0]), float(p_safe[1]), float(p_safe[2])


# Shared CBF filter instance (created in __main__ after URI list is known)
cbf_filter: CBFSafetyFilter | None = None


# ---------------------------------------------------------------------------
# Telemetry logging
# ---------------------------------------------------------------------------

def setup_logging(scf: SyncCrazyflie):
    """
    Register a 10 ms state-estimate log on *scf* and feed the results into
    the CBF filter so it always reasons about real positions.
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
    lc = loggers.get(uri)
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
            id_, t, x, y, z = row
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
    thread can signal all fly_sequence threads to abort cleanly *before*
    emergency_land is called — preventing the parallel_safe collision that
    occurred in the original code.
    """
    uri = scf.cf.link_uri
    commander = commanders.get(uri)
    if commander is None:
        raise RuntimeError(f"No commander for {uri} — did make_commander run?")

    print(f'[{uri}] Taking off...')
    commander.take_off(height=TAKEOFF_HEIGHT, velocity=DEFAULT_VELOCITY)
    time.sleep(1.0)

    # Seed the CBF filter with the real post-takeoff position from telemetry.
    # By this point setup_logging has been running for >1 s, so the position
    # reported by the onboard estimator is already in cbf_filter via the
    # log callback.  The explicit seed below is just a safety fallback in
    # case the log hasn't fired yet.
    if cbf_filter is not None and uri not in loggers:
        cbf_filter.update_position(uri, np.array([0.0, 0.0, TAKEOFF_HEIGHT]))

    print(f'[{uri}] Airborne')

    # First drone triggers audio
    if uri == sorted(uris)[0]:
        print(f'[{uri}] Triggering audio playback')
        audio_started.set()

    if not sequence:
        print(f'[{uri}] No waypoints — hovering briefly')
        # Still honour stop_event during the hover
        stop_event.wait(timeout=2.0)
    else:
        print(f'[{uri}] Running {len(sequence)} waypoints (CBF active)...')
        for idx, (target_time, x_des, y_des, z_des) in enumerate(sequence):

            # Abort cleanly if the main thread has signalled a stop
            if stop_event.is_set():
                print(f'[{uri}] Stop event — aborting sequence at wp {idx}')
                break

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

            commander.go_to(x_safe, y_safe, z_safe, velocity=DEFAULT_VELOCITY)

        print(f'[{uri}] Sequence complete')

    print(f'[{uri}] Landing...')
    commander.land(velocity=DEFAULT_VELOCITY)
    print(f'[{uri}] Landed')


def emergency_land(scf: SyncCrazyflie):
    """
    Best-effort emergency land.  Tries the PositionHlCommander first, falls
    back to the raw high-level commander.  Does NOT check _is_flying (private
    attribute, not reliable under exception conditions).
    """
    uri = scf.cf.link_uri
    print(f'[{uri}] EMERGENCY LAND')
    try:
        commander = commanders.get(uri)
        if commander is not None:
            commander.land(velocity=DEFAULT_VELOCITY)
        else:
            # Commander was never created — use the raw HL interface
            scf.cf.high_level_commander.land(0.0, 2.0)
        time.sleep(2.5)
    except Exception as e:
        print(f'[{uri}] Emergency land failed: {e}')
        # Last resort: send a direct land via raw HL commander
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

    # CBF filter
    # d_safe : 0.5 m minimum separation
    # gamma  : 1.0 — increase for more aggressive barrier enforcement
    # dt     : must match the actual interval between successive go_to calls
    cbf_filter = CBFSafetyFilter(uris, d_safe=0.5, gamma=1.0, dt=0.5)
    print(f'[CBF] Initialised | d_safe={cbf_filter.d_safe}m  γ={cbf_filter.gamma}')

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
            time.sleep(0.5)   # let the first log callbacks fire

            print('Creating commanders...')
            swarm.parallel_safe(make_commander)

            print('Flying sequences...')
            swarm.parallel_safe(fly_sequence, args_dict=seq_args)

            audio_thread.join(timeout=300)

        except (KeyboardInterrupt, Exception) as e:
            if isinstance(e, KeyboardInterrupt):
                print('\n[MAIN] KeyboardInterrupt — initiating safe shutdown')
            else:
                print(f'[MAIN] Swarm error: {e}')
                traceback.print_exc()

            # Signal all fly_sequence threads to exit their waypoint loop.
            # They will call commander.land() themselves and then return,
            # so when parallel_safe(emergency_land) runs below there are no
            # competing commander calls in flight.
            stop_event.set()
            time.sleep(1.5)   # give fly_sequence threads time to break out

            # Stop audio before issuing land commands
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

            # Now it is safe to issue emergency land — fly_sequence threads
            # have either finished their own land or broken out of the loop
            swarm.parallel_safe(emergency_land)

        finally:
            # Always stop logs, even if everything went smoothly
            swarm.parallel_safe(stop_logging)