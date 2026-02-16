#ifndef CBF_SOLVER_H
#define CBF_SOLVER_H
#pragma once

#include <vector>
#include <osqp.h>
#include "config.h"

extern Config cfg;

struct Vec3 {
    float x, y, z;
};

struct CBFConfig 
{
    float safety_radius = cfg.cbf_config.safety_radius;  // meters
    float alpha = cfg.cbf_config.alpha;            // CBF aggressiveness parameter
    float neighbor_range = cfg.cbf_config.neighbor_range;   // only consider neighbors within this range
    
    // Solver settings
    int max_iter = 2000;
    float eps_abs = 1e-3f;
    float eps_rel = 1e-3f;
    bool verbose = cfg.cbf_config.verbose;
    
    // Reinitialization threshold
    int constraint_change_threshold = 2;
};


class CBFSolver
{
    public:
        CBFSolver();
        ~CBFSolver();

        Vec3 Solve(int drone_id, 
                Vec3 v_nom, 
                Vec3 v_curr,
               const std::vector<Vec3>& pos,
               const std::vector<Vec3>& vel,
               float v_max,
               float a_max,
               float dt,
               const CBFConfig& config);

        void Reset();

    private:
        void Cleanup();
        int BuildConstraints(int agent_idx, Vec3 v_curr,
                            const std::vector<Vec3>& pos,
                            const std::vector<Vec3>& vel,
                            float v_max, float a_max, float dt,
                            const CBFConfig& cfg);
        void InitSolver(int n_constraints);
        
        // OSQP structures
        OSQPSolver* solver_ = nullptr;
        OSQPSettings* settings_ = nullptr;
        
        // Constraint matrices (reused across solves)
        std::vector<OSQPFloat> A_data_;
        std::vector<OSQPInt> A_indices_;
        std::vector<OSQPFloat> l_;
        std::vector<OSQPFloat> u_;
        std::vector<OSQPFloat> q_;
        
        CBFConfig config_;
        int last_n_constraints_ = 0;
        bool initialized_ = false;
};

class CBFManager 
{
    public:
        void Init(int num_agents);
        void Resize(int num_agents);
        
        Vec3 Solve(int agent_idx, Vec3 v_nom,
                const std::vector<Vec3>& positions,
                const std::vector<Vec3>& velocities,
                float v_max, float a_max, float dt);
        
        void SetConfig(const CBFConfig& cfg) { config_ = cfg; }
        void ResetAll();
        
    private:
        std::vector<CBFSolver> solvers_;
        CBFConfig config_;
};


#endif
