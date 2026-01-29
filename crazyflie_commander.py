import time
import csv
import numpy as np
from collections import defaultdict

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.positioning.position_hl_commander import PositionHlCommander

PATH = "trajectories.csv"
TAKEOFF_HEIGHT = 1.0
DEFAULT_VELOCITY = 0.5

uris = {
    'radio://0/20/2M/E7E7E7E701',
}


def read_csv(path):
    waypoint_map = defaultdict(list)

    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            id, t, x, y, z = row
            waypoint_map[int(id)].append((float(t), 
                                          np.clip(float(x), -1.1, 1.1),
                                          np.clip(float(y), -1.8, 1.8),
                                          np.clip(float(z), 0.1, 1.2),
                                        0.))

            for id in waypoint_map:
                waypoint_map[id].sort(key=lambda p: p[0])

    # print(waypoint_map)
    return waypoint_map


def init_data(path):
    waypoints = read_csv(path)
    seq = {}
    for i, uri in enumerate(sorted(uris)):
        seq[uri] = waypoints.get(i, [])[:50]
    return seq


def configure_lighthouse(scf):
    cf = scf.cf
    cf.param.set_value('deck.bcLighthouse4', '1')
    cf.param.set_value('stabilizer.estimator', '2')
    cf.param.set_value('stabilizer.controller', '2')
    time.sleep(0.5)


def take_off(scf):
    with PositionHlCommander(scf, default_height=TAKEOFF_HEIGHT, 
                             default_velocity=DEFAULT_VELOCITY) as pc:
        pc.take_off(height=TAKEOFF_HEIGHT, velocity=DEFAULT_VELOCITY)
        time.sleep(2.0)


def land(scf):
    with PositionHlCommander(scf, default_height=TAKEOFF_HEIGHT,
                             default_velocity=DEFAULT_VELOCITY) as pc:
        pc.land(velocity=DEFAULT_VELOCITY)
        time.sleep(2.0)

def run_sequence(scf: SyncCrazyflie, sequence):
    if not sequence:
        return

    with PositionHlCommander(scf, default_height=TAKEOFF_HEIGHT,
                             default_velocity=DEFAULT_VELOCITY) as pc:
        for target_time, x, y, z, yaw in sequence:
            duration = 0.2  # Fixed duration per waypoint
            
            print('Setting position {} to cf {}'.format((x, y, z), scf.cf.link_uri))
            pc.go_to(x, y, z, velocity=DEFAULT_VELOCITY)
            time.sleep(duration)


def emergency_land(scf):
    try:
        commander = scf.cf.high_level_commander
        commander.land(0.0, 2.0)
        time.sleep(2.5)
        commander.stop()
    except Exception as e:
        print(f'Emergency land failed: {e}')


if __name__ == '__main__':
    seq_args = init_data(PATH)
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    
    with Swarm(uris, factory=factory) as swarm:
        try:
            swarm.parallel_safe(configure_lighthouse)
            swarm.reset_estimators()
            time.sleep(2)
            swarm.parallel_safe(take_off)
            swarm.parallel_safe(run_sequence, args_dict=seq_args)
            swarm.parallel_safe(land)
            
        except (KeyboardInterrupt, Exception) as e:
            print(f'Error: {e}')
            swarm.parallel_safe(emergency_land)