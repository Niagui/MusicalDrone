#pragma once
#include <vector>
#include <string>
#include "cbf_solver.h"
#include "config.h"

bool ReloadAndApplyEmotions(float t);

// basic vector struct for positions/velocities
// struct Vec3 {
//     float x, y, z;
// };

// Parameters for tuning behavior (emotion can override)
struct BoidParams {
    //Radius Seperation - controls how close drones get to each other before seperating; +: repelling -: denser
    float r_sep;

    //Neighbor Radius - controls alignment/at what range drones influence each others movement; +: influenced by more neighbors 
    float r_nei;

    //Seperation weight - controls strength of avoidance to keep drones from running into each other; +: more push within r_sep 
    float k_sep;

    //Alignment weight - controls how much drones attempt to match velocity to neighbooring drones; +: stronger match within r_nei
    float k_ali;

    //Cohesion weight - controls the strength of pull towards the neighbooring drones average position; +: grouping
    float k_coh;

    //Goal/formation weight - controls pull towards choreography plan/path; +: incline toward attractor
    float k_goal;

    //Maximum speed(m/s)
    float vmax;

    //Maximum acceleration(m/s^2)
    float amax;

    //average target altitude
    float altitude;

    //randomness to break perfect symmetry
    float jitter;
};

struct Boundaries {
    float xmin, xmax;
    float ymin, ymax;
    float zmin, zmax;
};

const std::vector<float>& GetLastWeights();
bool LoadEmotionFile(const std::string& path);
bool LoadResetTimes(const std::string& filename);
std::vector<float> GetEmotionWeights(float t);
void ApplyEmotion(const std::vector<float>& w, BoidParams& P);
BoidParams ApplyEmotionHard(const std::vector<float>& w, float scale);
bool ReloadAndApplyEmotions(float t);

// External functions
void InitBoids(int count);
void ResizeBoids(int count);
void UpdateBoids(float dt, const std::vector<Vec3>& targets);
void SetBoidParams(const BoidParams& p);
void SetSimTime(float t);
void ResetVelocities();
void EnsureEmotionsLoaded();
void EnsureSegmentsLoaded();

Boundaries GetBoxBounds();

float GetAudioLength();
void StartAudio(const std::string& filename);


std::vector<Vec3>& GetBoidPositions();  //to draw in basicVisuals.cpp
std::vector<Vec3>& GetBoidVelocities();
std::vector<Vec3>& GetBoidAcclerations();

const BoidParams& GetBoidParams();
const std::vector<std::string>& GetEmotionLabels();


//cbf
void EnableCBF(bool enable);
void SetCBFConfig(const CBFConfig& config);