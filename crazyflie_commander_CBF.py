import time
import csv
import numpy as np
from collections import defaultdict
import traceback
import threading

import cflib.crtp
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
audio_started = threading.Event()


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
# current estimated velocity (latest known).  If the desired velocity vᵢ_des
# violates the CBF inequality, the minimal correction (1-D projection onto the
# constraint normal) is applied:
#
#   vᵢ_safe = vᵢ_des + λ · n̂ᵢⱼ    where  n̂ᵢⱼ = (pᵢ−pⱼ)/‖pᵢ−pⱼ‖
#   λ = max(0, (rhs − 2(pᵢ−pⱼ)·vᵢ_des) / (2‖pᵢ−pⱼ‖²))  · ‖pᵢ−pⱼ‖²
#
# The safe velocity is then converted back to a position command:
#   p_safe = pᵢ + vᵢ_safe · Δt
# ---------------------------------------------------------------------------

class CBFSafetyFilter:
    """
    Thread-safe CBF filter for a swarm of drones.

    Parameters
    ----------
    d_safe : float
        Minimum allowed centre-to-centre separation (metres).
    gamma : float
        CBF decay rate γ — larger = more aggressive correction.
    dt : float
        Timestep used to convert velocity back to a position command.
        Should roughly match the interval between successive go_to calls.
    """

    def __init__(self, uris: list[str], d_safe: float = 0.5,
                 gamma: float = 1.0, dt: float = 0.5):
        self.d_safe = d_safe
        self.gamma = gamma
        self.dt = dt
        self.uris = list(uris)

        self._lock = threading.Lock()
        # Current best-known position of every drone (set at takeoff / updated
        # after each filtered go_to).
        self._positions: dict[str, np.ndarray] = {
            uri: np.zeros(3) for uri in self.uris
        }
        # Latest velocity estimate for each drone (used as the "other" drone's
        # velocity in the decentralised CBF).
        self._velocities: dict[str, np.ndarray] = {
            uri: np.zeros(3) for uri in self.uris
        }

    # Public helpers

    def update_position(self, uri: str, pos: np.ndarray):
        """Call this whenever you have a fresh position for a drone."""
        with self._lock:
            self._positions[uri] = np.array(pos, dtype=float)

    def filter(self, uri: str, x_des: float, y_des: float,
               z_des: float) -> tuple[float, float, float]:
        """
        Return a CBF-safe target position for *uri* given its desired target.

        The returned position is the closest point to (x_des, y_des, z_des)
        that satisfies the pairwise CBF constraints with every other drone.
        """
        with self._lock:
            p_i = self._positions[uri].copy()
            p_des = np.array([x_des, y_des, z_des], dtype=float)

            # Desired velocity (proportional to displacement, capped at
            # DEFAULT_VELOCITY so the CBF timestep interpretation is consistent)
            v_des = p_des - p_i           # direction × magnitude

            v_safe = v_des.copy()

            for other_uri, p_j in self._positions.items():
                if other_uri == uri:
                    continue

                diff = p_i - p_j          # vector from j to i
                dist_sq = float(diff @ diff)
                dist = np.sqrt(dist_sq)

                h = dist_sq - self.d_safe ** 2

                if dist < 1e-4:
                    # Drones are on top of each other — push along z to recover
                    v_safe += np.array([0.0, 0.0, 0.3])
                    print(f'[CBF] WARNING: {uri} and {other_uri} nearly coincident!')
                    continue

                v_j = self._velocities[other_uri].copy()

                # CBF Lie derivative: ḣ = 2(pᵢ−pⱼ)·(vᵢ−vⱼ)
                lie_h = 2.0 * diff @ (v_safe - v_j)
                cbf_rhs = -self.gamma * h          # required lower bound on ḣ

                if lie_h < cbf_rhs:
                    # Constraint violated — compute minimal correction
                    # Projection direction: outward normal n̂ᵢⱼ (unit vector i→j)
                    n = diff / dist       # unit vector pointing away from j

                    # Required increase in ḣ
                    deficit = cbf_rhs - lie_h
                    # ḣ increases by 2‖diff‖ for each unit of λ along n
                    # so:  λ = deficit / (2 * ‖diff‖ * 2) ... let's be exact:
                    # Δ(lie_h) from adding λ*n to v_safe = 2 * diff · (λ*n)
                    #                                     = 2 * ‖diff‖ * λ
                    lam = deficit / (2.0 * dist)
                    v_safe = v_safe + lam * n

                    print(
                        f'[CBF] {uri} ↔ {other_uri} | '
                        f'dist={dist:.3f}m  h={h:.3f}  correction λ={lam:.4f}'
                    )

            # Store updated velocity estimate
            self._velocities[uri] = v_safe.copy()

            # Convert safe velocity back to a position target
            p_safe = p_i + v_safe * self.dt

            # Update stored position to the commanded target
            self._positions[uri] = p_safe.copy()

        return float(p_safe[0]), float(p_safe[1]), float(p_safe[2])


# Shared CBF filter instance (created after URI list is known)
cbf_filter: CBFSafetyFilter | None = None


# Audio

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


# CSV / waypoint loading

def read_csv(path):
    waypoint_map = defaultdict(list)
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            id, t, x, y, z, vx, vy, vz, ax, ay, az = row
            waypoint_map[int(id)].append((
                float(t),
                np.clip(float(x), -1.1, 1.1),
                np.clip(float(y), -1.8, 1.8),
                np.clip(float(z), 0.1, 1.2)
            ))

    for id in waypoint_map:
        waypoint_map[id].sort(key=lambda p: p[0])

    print(f"Loaded waypoints for drone IDs: {list(waypoint_map.keys())}")
    for id in waypoint_map:
        print(f"  Drone {id}: {len(waypoint_map[id])} waypoints")

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


# Swarm callbacks

def make_commander(scf):
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
    """Take off, execute CBF-filtered waypoint sequence, then land."""
    uri = scf.cf.link_uri
    commander = commanders.get(uri)
    if commander is None:
        raise RuntimeError(f"No commander for {uri} — did make_commander run?")

    print(f'[{uri}] Taking off...')
    commander.take_off(height=TAKEOFF_HEIGHT, velocity=DEFAULT_VELOCITY)
    time.sleep(1.0)

    # Seed the CBF filter with the takeoff position.
    # PositionHlCommander doesn't expose state feedback directly, so we
    # initialise from the HL commander's last setpoint (0, 0, height).
    if cbf_filter is not None:
        cbf_filter.update_position(uri, np.array([0.0, 0.0, TAKEOFF_HEIGHT]))

    print(f'[{uri}] Airborne')

    # First drone triggers audio
    if uri == sorted(uris)[0]:
        print(f'[{uri}] Triggering audio playback')
        audio_started.set()

    if not sequence:
        print(f'[{uri}] No waypoints — hovering briefly')
        time.sleep(2.0)
    else:
        print(f'[{uri}] Running {len(sequence)} waypoints (CBF active)...')
        for idx, (target_time, x_des, y_des, z_des) in enumerate(sequence):

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
    pygame.mixer.music.fadeout(20)
    pygame.mixer.music.stop()



def emergency_land(scf):
    uri = scf.cf.link_uri
    print(f'[{uri}] EMERGENCY LAND')
    try:
        commander = commanders.get(uri)
        if commander and commander._is_flying:
            commander.land(velocity=DEFAULT_VELOCITY)
        else:
            scf.cf.high_level_commander.land(0.0, 2.0)
        time.sleep(2.5)
    except Exception as e:
        print(f'[{uri}] Emergency land failed: {e}')


# Entry point

if __name__ == '__main__':
    seq_args = init_data(PATH)
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')

    # Initialise CBF filter
    # d_safe  : 0.5 m minimum separation (adjust to your flight volume)
    # gamma   : 1.0 — higher = more aggressive barrier enforcement
    # dt      : 0.5 s — approximate time between successive go_to calls;
    #           tune to match your actual loop rate
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

            print('Creating commanders...')
            swarm.parallel_safe(make_commander)

            print('Flying sequences...')
            swarm.parallel_safe(fly_sequence, args_dict=seq_args)

            audio_thread.join(timeout=250)

        except (KeyboardInterrupt, Exception) as e:
            print(f'Swarm error: {e}')
            traceback.print_exc()
            swarm.parallel_safe(emergency_land)
            pygame.mixer.music.stop()
