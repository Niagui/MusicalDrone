import numpy as np
import cvxpy as cp


def solve_safe_control(x_current, u_nominal, obstacles, d_safe=0.3):
    """
    x_current: [x, y, z, vx, vy, vz] (Real-time telemetry)
    u_nominal: [ax, ay, az] (Target acceleration)
    """
    u = cp.Variable(3)
    p = x_current[0:3]
    v = x_current[3:6]
    
    constraints = []
    
    # Pairwise Drone/Obstacle constraints
    for obs in obstacles:
        c = obs['c']
        r = obs['r']
        
        # h(x) = distance squared - safety radius squared
        h = np.linalg.norm(p - c)**2 - (r + d_safe)**2
        
        # Lie Derivatives (Simplified HOCBF)
        # h_dot = 2(p-c) @ v
        # h_ddot = 2(v@v) + 2(p-c) @ acceleration
        # Constraint: h_ddot + alpha1*h_dot + alpha2*h >= 0
        alpha1, alpha2 = 10.0, 5.0 
        
        rel_p = p - c
        b = -2 * (v @ v) - alpha1 * (2 * rel_p @ v) - alpha2 * h
        constraints.append(2 * rel_p @ u >= b)

    # Minimize difference between safe and nominal control
    prob = cp.Problem(cp.Minimize(cp.norm(u - u_nominal)), constraints)
    prob.solve()
    
    return u.value if u.value is not None else u_nominal
