# import logging
import time
import csv
from collections import defaultdict

import cflib.crtp
# from cflib.crazyflie import Crazyflie
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem.trajectory_memory import Poly4D
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig



PATH = "trajectories.csv"
uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')


def read_csv(path):
    waypoint_map = defaultdict(list)

    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            t, id, x, y, z = row
            waypoint_map[int(id)].append((float(x),float(y),float(z),float(t)))

    # print(waypoint_map)
    return waypoint_map


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
    for i, uri in enumerate(uris):
        seq[uri] = [waypoints[i]]
    return seq



def _stab_log_data(timestamp, data, logconf):
    print('x: {}, y: {}, z: {}'.format(x = data['stateEstimate.x'],
                                        y = data['stateEstimate.y'],
                                        z = data['stateEstimate.z']))
    
def _log_error(logconf, msg):
    print(f"[LOG ERROR] {logconf.name}: {msg}")


def set_up_logging(scf: SyncCrazyflie, period_ms: int = 100):
    _lg_stab = LogConfig(name='pos', period_in_ms=period_ms)
    _lg_stab.add_variable('stateEstimate.x', 'float')
    _lg_stab.add_variable('stateEstimate.y', 'float')
    _lg_stab.add_variable('stateEstimate.z', 'float')

    scf.cf.log.add_config(_lg_stab)
    _lg_stab.data_received_cb.add_callback(_stab_log_data)
    _lg_stab.error_cb.add_callback(_log_error)

    scf._pos_logconf = _lg_stab
    _lg_stab.start()



def activate_led_bit_mask(scf):
    scf.cf.param.set_value('led.bitmask', 255)

def deactivate_led_bit_mask(scf):
    scf.cf.param.set_value('led.bitmask', 0)

def light_check(scf):
    activate_led_bit_mask(scf)
    time.sleep(2)
    deactivate_led_bit_mask(scf)


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

        x, y, z = arguments[0], arguments[1], arguments[2]
        duration = 0.2

        print('Setting position {} to cf {}'.format((x, y, z), cf.link_uri))
        commander.go_to(x, y, z, 0, duration, relative=False)
        time.sleep(duration)


uris = {
    'radio://0/20/2M/E7E7E7E701',
    # Add more URIs if you want more copters in the swarm
    # URIs in a swarm using the same radio must also be on the same channel
}


if __name__ == '__main__':
    seq_args = init_data(PATH)
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        swarm.parallel_safe(set_up_logging)
        swarm.parallel_safe(light_check)
        swarm.reset_estimators()
        swarm.parallel_safe(take_off)
        swarm.parallel_safe(run_sequence, args_dict=seq_args)
        swarm.parallel_safe(land)
