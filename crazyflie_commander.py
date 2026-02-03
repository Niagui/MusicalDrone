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

# Global dict to store PositionHlCommander objects for each drone
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
    
    # Sort waypoints by timestamp for each drone
    for id in waypoint_map:
        waypoint_map[id].sort(key=lambda p: p[0])

    print(f"Loaded waypoints for drone IDs: {list(waypoint_map.keys())}")
    for id in waypoint_map:
        print(f"  Drone {id}: {len(waypoint_map[id])} waypoints")
    
    return waypoint_map


def init_data(path):
    waypoints = read_csv(path)
    seq = {}
    
    # Match URIs to drone IDs in the CSV
    # Assuming drone ID in CSV corresponds to index in sorted URIs
    for i, uri in enumerate(sorted(uris)):
        if i in waypoints:
            seq[uri] = waypoints[i][:50]
            print(f"Assigned {len(seq[uri])} waypoints to {uri} (drone ID {i})")
        else:
            print(f"WARNING: No waypoints found for drone ID {i} (URI: {uri})")
            seq[uri] = []
    
    return seq


def make_commander(scf):
    """Create and store PositionHlCommander for this drone."""
    try:
        print(f'Creating commander for {scf.cf.link_uri}')
        commander = PositionHlCommander(
            scf, 
            default_velocity=DEFAULT_VELOCITY, 
            controller=PositionHlCommander.CONTROLLER_PID
        )
        # Enter the context manager
        commander.__enter__()
        commanders[scf.cf.link_uri] = commander
        print(f'Commander created and initialized for {scf.cf.link_uri}')
    except Exception as e:
        print(f'Error creating commander for {scf.cf.link_uri}: {e}')
        traceback.print_exc()
        raise


def configure_lighthouse(scf):
    cf = scf.cf
    cf.param.set_value('deck.bcLighthouse4', '1')
    cf.param.set_value('stabilizer.estimator', '2')
    cf.param.set_value('stabilizer.controller', '2')
    time.sleep(0.5)


def take_off(scf):
    """Take off using the stored PositionHlCommander."""
    try:
        print(f'Taking off {scf.cf.link_uri}')
        commander = commanders.get(scf.cf.link_uri)
        if commander is None:
            raise RuntimeError(f"No commander found for {scf.cf.link_uri}")
        
        commander.take_off(height=TAKEOFF_HEIGHT, velocity=DEFAULT_VELOCITY)
        time.sleep(2.0)
        print(f'Take off complete for {scf.cf.link_uri}')
    except Exception as e:
        print(f'Error during takeoff for {scf.cf.link_uri}: {e}')
        traceback.print_exc()
        raise


def land(scf):
    """Land using the stored PositionHlCommander."""
    try:
        print(f'Landing {scf.cf.link_uri}')
        commander = commanders.get(scf.cf.link_uri)
        if commander is None:
            raise RuntimeError(f"No commander found for {scf.cf.link_uri}")
        
        commander.land(velocity=DEFAULT_VELOCITY)
        time.sleep(2.0)
        print(f'Landing complete for {scf.cf.link_uri}')
    except Exception as e:
        print(f'Error during landing for {scf.cf.link_uri}: {e}')
        traceback.print_exc()
        raise


def run_sequence(scf: SyncCrazyflie, sequence):
    """Execute waypoint sequence using stored PositionHlCommander.
    
    Each waypoint is (target_time, x, y, z).
    """
    try:
        print(f'Running sequence for {scf.cf.link_uri} with {len(sequence)} waypoints')
        
        if not sequence:
            print(f'No sequence for {scf.cf.link_uri}, skipping')
            return

        commander = commanders.get(scf.cf.link_uri)
        if commander is None:
            raise RuntimeError(f"No commander found for {scf.cf.link_uri}")

        for idx, (target_time, x, y, z) in enumerate(sequence):
            duration = 0.2  # Fixed duration per waypoint
            
            if idx % 10 == 0:  # Print every 10th waypoint to reduce spam
                print(f'{scf.cf.link_uri}: Waypoint {idx}/{len(sequence)} - position ({x:.2f}, {y:.2f}, {z:.2f})')
            
            commander.go_to(x, y, z, velocity=DEFAULT_VELOCITY)
            time.sleep(duration)
        
        print(f'Sequence complete for {scf.cf.link_uri}')
    except Exception as e:
        print(f'Error during sequence for {scf.cf.link_uri}: {e}')
        traceback.print_exc()
        raise


def emergency_land(scf):
    try:
        print(f'Emergency landing {scf.cf.link_uri}')
        # Try using stored commander first
        commander = commanders.get(scf.cf.link_uri)
        if commander:
            commander.land(velocity=DEFAULT_VELOCITY)
        else:
            # Fallback to high level commander
            commander = scf.cf.high_level_commander
            commander.land(0.0, 2.0)
        time.sleep(2.5)
    except Exception as e:
        print(f'Emergency land failed for {scf.cf.link_uri}: {e}')


def cleanup_commanders():
    """Exit context managers for all commanders."""
    for uri, commander in commanders.items():
        try:
            print(f'Cleaning up commander for {uri}')
            commander.__exit__(None, None, None)
        except Exception as e:
            print(f'Error cleaning up commander for {uri}: {e}')


if __name__ == '__main__':
    seq_args = init_data(PATH)
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    
    with Swarm(uris, factory=factory) as swarm:
        try:
            #swarm.parallel_safe(configure_lighthouse)
            print('Resetting estimators...')
            swarm.reset_estimators()
            time.sleep(2)
            
            # Create and store commanders for all drones
            print('Creating commanders...')
            swarm.parallel_safe(make_commander)
            
            print('Taking off...')
            swarm.parallel_safe(take_off)
            
            print('Running sequences...')
            swarm.parallel_safe(run_sequence, args_dict=seq_args)
            
            print('Landing...')
            swarm.parallel_safe(land)
            
        except (KeyboardInterrupt, Exception) as e:
            print(f'Error in main: {e}')
            traceback.print_exc()
            swarm.parallel_safe(emergency_land)
        finally:
            print('Cleaning up...')
            cleanup_commanders()
