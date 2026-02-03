import time
import csv
import numpy as np
from collections import defaultdict

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.positioning.motion_commander import MotionCommander

PATH = "trajectories.csv"
TAKEOFF_HEIGHT = 1.0
DEFAULT_VELOCITY = 0.5

uris = {
    'radio://0/20/2M/E7E7E7E701',
    # Add more URIs if you want more copters in the swarm
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

    return waypoint_map


def init_data(path):
    """
    Create sequence dictionary mapping URIs to waypoint lists
    """
    waypoints = read_csv(path)
    seq = {}
    
    for i, uri in enumerate(sorted(uris)):
        seq[uri] = waypoints[i][:300]  # Get up to 300 waypoints
    
    return seq


def take_off(scf):
    """Take off using MotionCommander"""
    with MotionCommander(scf, default_height=TAKEOFF_HEIGHT) as mc:
        print(f'Taking off {scf.cf.link_uri}')
        time.sleep(2.0)  # Hover for 2 seconds after takeoff


def land(scf):
    """Land using high level commander"""
    commander = scf.cf.high_level_commander
    commander.land(0.0, 2.0)
    time.sleep(2)
    commander.stop()


def run_sequence(scf: SyncCrazyflie, sequence):
    """
    Execute trajectory using MotionCommander with relative movements
    """
    if not sequence:
        print(f'No sequence for {scf.cf.link_uri}')
        return
    
    cf = scf.cf
    
    # Start with MotionCommander context
    with MotionCommander(scf, default_height=TAKEOFF_HEIGHT) as mc:
        # Current position starts at takeoff position (0, 0, TAKEOFF_HEIGHT)
        current_pos = np.array([0.0, 0.0, TAKEOFF_HEIGHT])
        
        for i, waypoint in enumerate(sequence):
            target_time, x, y, z, yaw = waypoint
            target_pos = np.array([x, y, z])
            
            # Calculate relative movement
            delta = target_pos - current_pos
            distance = np.linalg.norm(delta)
            
            if distance < 0.01:  # Skip if too close
                continue
            
            # Calculate duration based on velocity
            duration = distance / DEFAULT_VELOCITY
            duration = max(0.1, min(duration, 2.0))  # Clamp between 0.1 and 2.0 seconds
            
            print(f'Moving to ({x:.2f}, {y:.2f}, {z:.2f}) - delta: ({delta[0]:.2f}, {delta[1]:.2f}, {delta[2]:.2f}) - {cf.link_uri}')
            
            # Use move_distance for relative movement
            mc.move_distance(delta[0], delta[1], delta[2], velocity=DEFAULT_VELOCITY)
            
            # Update current position
            current_pos = target_pos
            
            # Small sleep between waypoints
            time.sleep(0.1)


def emergency_land(scf):
    """Emergency landing procedure"""
    try:
        commander = scf.cf.high_level_commander
        commander.land(0.0, 2.0)
        time.sleep(2.5)
        commander.stop()
        print(f'Emergency landed {scf.cf.link_uri}')
    except Exception as e:
        print(f'Emergency land failed for {scf.cf.link_uri}: {e}')


if __name__ == '__main__':
    seq_args = init_data(PATH)
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    
    with Swarm(uris, factory=factory) as swarm:
        try:
            # swarm.parallel_safe(light_check)
            swarm.reset_estimators()
            time.sleep(2)
            
            swarm.parallel_safe(take_off)
            swarm.parallel_safe(run_sequence, args_dict=seq_args)
            swarm.parallel_safe(land)
            
        except (ValueError, KeyboardInterrupt, IndexError, Exception) as e:
            print(f"Error occurred: {e}")
            print("Attempting emergency landing...")
            swarm.parallel_safe(emergency_land)
