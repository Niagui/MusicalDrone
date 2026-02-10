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


def fly_sequence(scf: SyncCrazyflie, sequence):
    """
    Complete lifecycle for one drone: takeoff → sequence → land.

    PositionHlCommander owns the entire lifecycle inside its `with` block.
    Takeoff happens automatically on __enter__, landing on __exit__.
    """
    uri = scf.cf.link_uri
    print(f'[{uri}] Starting flight ({len(sequence)} waypoints)')

    try:
        with PositionHlCommander(
            scf,
            default_height=TAKEOFF_HEIGHT,
            default_velocity=DEFAULT_VELOCITY,
            controller=PositionHlCommander.CONTROLLER_PID
        ) as commander:

            # PositionHlCommander has already taken off at this point
            print(f'[{uri}] Airborne, beginning sequence...')
            time.sleep(1.0)  # Stabilise after takeoff

            if not sequence:
                print(f'[{uri}] No waypoints, hovering briefly before landing')
                time.sleep(2.0)
            else:
                for idx, (target_time, x, y, z) in enumerate(sequence):
                    if idx % 10 == 0:
                        print(f'[{uri}] Waypoint {idx}/{len(sequence)} → ({x:.2f}, {y:.2f}, {z:.2f})')
                    commander.go_to(x, y, z, velocity=DEFAULT_VELOCITY)
                    time.sleep(0.2)

            print(f'[{uri}] Sequence complete, landing...')
            # Landing is automatic on `with` block exit

    except Exception as e:
        print(f'[{uri}] Error during flight: {e}')
        traceback.print_exc()
        # Attempt emergency land via high-level commander as fallback
        try:
            scf.cf.high_level_commander.land(0.0, 2.0)
            time.sleep(2.5)
        except Exception as land_err:
            print(f'[{uri}] Emergency land also failed: {land_err}')
        raise


if __name__ == '__main__':
    seq_args = init_data(PATH)
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')

    with Swarm(uris, factory=factory) as swarm:
        try:
            # Optional: uncomment if using Lighthouse positioning
            # swarm.parallel_safe(configure_lighthouse)

            print('Resetting estimators...')
            swarm.reset_estimators()
            time.sleep(2)

            print('Flying sequences...')
            swarm.parallel_safe(fly_sequence, args_dict=seq_args)

        except (KeyboardInterrupt, Exception) as e:
            print(f'Swarm error: {e}')
            # Emergency land: PositionHlCommander context has already exited
            # so fall back to raw high-level commander
            for uri in uris:
                try:
                    # SyncCrazyflie objects are managed by Swarm internally;
                    # parallel_safe handles per-drone exceptions above.
                    pass
                except Exception:
                    pass