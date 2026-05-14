#ifndef CONFIG_H
#define CONFIG_H
#pragma once

struct Drone_config
{
    int num_drones = 5;
    float init_dist = 0.45;
    // 0.0 = ignore box size for motion, 1.0 = full box-size scaling.
    float box_motion_scale_strength = 1.0;
    // Segment-level hard caps for boid dynamics.
    float vmax_cap = 2.5;
    float amax_cap = 8.0;
};

struct Bound_config
{
    float x_min = -1.1; //-1.1
    float x_max = 1.1;  //1.1
    float y_min = -1.1; //-1.1
    float y_max = 1.1; // 1.1
    float z_min = 0.5; //0.5
    float z_max = 1.2; //1.2
};

struct CBF_config
{
    float safety_radius = 0.3;
    float neighbor_range = 0.6;
    float alpha = 0.7;
    // Higher weight penalizes slack harder (stronger boundary/collision enforcement).
    float slack_weight = 40.0;
    // Lower max slack allows less constraint violation (more damping near boundaries).
    float slack_max = 1.0;
    bool verbose = false;
};

struct Boids_Param
{
    
};

struct Config
{
    Drone_config drone_config;
    Bound_config bound_config;
    CBF_config cbf_config;
};

#endif
