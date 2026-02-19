#ifndef CONFIG_H
#define CONFIG_H
#pragma once

struct Drone_config
{
    int num_drones = 2;
    float init_dist = 1;
};

struct Bound_config
{
    float x_min, x_max;
    float y_min, y_max;
    float z_min, z_max;
};

struct CBF_config
{
    float safety_radius = 0.4;
    float neighbor_range = 10.0;
    float alpha = 1.0;
    bool verbose = false;
};

struct Config
{
    Drone_config drone_config;
    Bound_config bound_config;
    CBF_config cbf_config;
};

#endif