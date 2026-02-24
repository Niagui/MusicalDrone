#ifndef CONFIG_H
#define CONFIG_H
#pragma once

struct Drone_config
{
    int num_drones = 4;
    float init_dist = 0.45;
};

struct Bound_config
{
    float x_min = -1.2;
    float x_max = 1.2;
    float y_min = -1.2;
    float y_max = 1.2;
    float z_min = 0.5;
    float z_max = 1.5;
};

struct CBF_config
{
    float safety_radius = 0.3;
    float neighbor_range = 0.6;
    float alpha = 0.7;
    bool verbose = true;
};

struct Config
{
    Drone_config drone_config;
    Bound_config bound_config;
    CBF_config cbf_config;
};

#endif