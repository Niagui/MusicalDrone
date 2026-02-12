# import logging
import time
import csv
import numpy as np
from collections import defaultdict

import cflib.crtp
# from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm

from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.utils import uri_helper

PATH = "trajectories.csv"

uris = {
    'radio://0/20/2M/E7E7E7E702',
    'radio://0/20/2M/E7E7E7E703'
    # Add more URIs if you want more copters in the swarm
    # URIs in a swarm using the same radio must also be on the same channel
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


def write_timed_waypoints(id_, samples, out_path):
    """
    samples: list of (t, x, y, z, yaw)
    """
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        for t, x, y, z, yaw in samples:
            w.writerow([f"{t:.4f}", f"{x:.2f}", f"{y:.2f}", f"{z:.2f}", yaw])


def init_data(path):
    """
    make a data structure like:
    seq_args = {
        uris[0]: [sequence0],
        uris[1]: [sequence1],
        uris[2]: [sequence2],
        uris[3]: [sequence3],
    }
    """
    waypoints = read_csv(path)
    seq = defaultdict(list)

    #use goto
    for i, uri in enumerate(uris):
        seq[uri] = [waypoints[i][:300]]

    #use to generate trajectories
    for id, waypoint in waypoints.items():
        write_timed_waypoints(id, waypoint, f"tmp_{id}.csv")
        print()
    # print(seq["radio://0/20/2M/E7E7E7E701"])
    return seq


# def activate_led_bit_mask(scf):
#     scf.cf.param.set_value('led.bitmask', 255)

# def deactivate_led_bit_mask(scf):
#     scf.cf.param.set_value('led.bitmask', 0)

# def light_check(scf):
#     activate_led_bit_mask(scf)
#     time.sleep(2)
#     deactivate_led_bit_mask(scf)


def take_off(scf):
    commander= scf.cf.high_level_commander

    commander.takeoff(1.0, 2.0)
    time.sleep(3)


def land(scf):
    commander= scf.cf.high_level_commander

    commander.land(0.0, 2.0)
    time.sleep(2)

    commander.stop()



def run_sequence(scf: SyncCrazyflie, sequence):
    cf = scf.cf

    for arguments in sequence:
        commander = scf.cf.high_level_commander

        #add np.clip
        x, y, z = arguments[1], arguments[2], arguments[3]
        duration = 0.2

        print('Setting position {} to cf {}'.format((x, y, z), cf.link_uri))
        commander.go_to(x, y, z, 0, duration, relative=False)
        time.sleep(duration)




if __name__ == '__main__':
    seq_args = init_data(PATH)
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        
    # wrapper
    # add acceleration bound
        try:
            # swarm.parallel_safe(light_check)
            swarm.reset_estimators()
            swarm.parallel_safe(take_off)
            swarm.parallel_safe(run_sequence, args_dict=seq_args)
            swarm.parallel_safe(land)

        except(ValueError, KeyboardInterrupt, IndexError):
            print("error: try landing")
            swarm.parallel_safe(land)
