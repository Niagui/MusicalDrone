import time
import csv
import numpy as np
from collections import defaultdict
import traceback

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.positioning.position_hl_commander import PositionHlCommander

PATH = "trajectories.csv"
TAKEOFF_HEIGHT = 1.0
DEFAULT_VELOCITY = 0.5

uris = [
    'radio://0/80/2M/E7E7E7E701',
]

# Global dict to store PositionHlCommander instances (no context manager)
commanders = {}


def read_csv(path):
    """Read waypoints from CSV file with format: id, t, x, y, z"""
    waypoint_map = defaultdict(list)

    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            id, t, x, y, z = row
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
            seq[uri] = waypoints[i][:50]
            print(f"Assigned {len(seq[uri])} waypoints to {uri} (drone ID {i})")
        else:
            print(f"WARNING: No waypoints found for drone ID {i} (URI: {uri})")
            seq[uri] = []

    return seq


def make_commander(scf):
    """
    Instantiate a PositionHlCommander and store it — does NOT take off.

    We deliberately avoid __enter__ because it auto-calls take_off().
    Calling the constructor directly only sets up params and activates the
    controller; the drone stays on the ground until fly_sequence() is called.
    """
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
    """Take off, execute waypoint sequence, then land. Each waypoint: (target_time, x, y, z)."""
    uri = scf.cf.link_uri
    commander = commanders.get(uri)
    if commander is None:
        raise RuntimeError(f"No commander for {uri} — did make_commander run?")

    print(f'[{uri}] Taking off...')
    commander.take_off(height=TAKEOFF_HEIGHT, velocity=DEFAULT_VELOCITY)
    time.sleep(1.0)  # Stabilise after ascent
    print(f'[{uri}] Airborne')

    if not sequence:
        print(f'[{uri}] No waypoints — hovering briefly')
        time.sleep(2.0)
    else:
        print(f'[{uri}] Running {len(sequence)} waypoints...')
        for idx, (target_time, x, y, z) in enumerate(sequence):
            if idx % 10 == 0:
                print(f'[{uri}] Waypoint {idx}/{len(sequence)} → ({x:.2f}, {y:.2f}, {z:.2f})')
            # go_to internally sleeps for the travel duration, no extra sleep needed.
            commander.go_to(x, y, z, velocity=DEFAULT_VELOCITY)
        print(f'[{uri}] Sequence complete')

    print(f'[{uri}] Landing...')
    commander.land(velocity=DEFAULT_VELOCITY)
    print(f'[{uri}] Landed')


def emergency_land(scf):
    """Best-effort landing — tries stored commander first, then raw HL commander."""
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


if __name__ == '__main__':
    seq_args = init_data(PATH)
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')

    with Swarm(uris, factory=factory) as swarm:
        try:
            # Optional: uncomment for Lighthouse deck
            # swarm.parallel_safe(configure_lighthouse)

            print('Resetting estimators...')
            swarm.reset_estimators()
            time.sleep(2)

            print('Creating commanders...')
            swarm.parallel_safe(make_commander)

            print('Flying sequences...')
            swarm.parallel_safe(fly_sequence, args_dict=seq_args)

        except (KeyboardInterrupt, Exception) as e:
            print(f'Swarm error: {e}')
            traceback.print_exc()
            swarm.parallel_safe(emergency_land)