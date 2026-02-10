#include "cbf_solver.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

// Math helpers
static inline Vec3 sub(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline Vec3 mul(Vec3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
static inline float dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline float len(Vec3 v) { return std::sqrt(dot(v, v)); }
static inline float clamp(float x, float lo, float hi) { 
    return std::max(lo, std::min(hi, x)); 
}
static inline Vec3 clampLen(Vec3 v, float maxLen) {
    float L = len(v);
    return (L > maxLen) ? mul(v, maxLen/L) : v;
}

// ============================================================================
// CBFSolver
// ============================================================================

CBFSolver::CBFSolver() {
    q_.resize(3, 0.0);
}

CBFSolver::~CBFSolver() {
    Cleanup();
}

void CBFSolver::Cleanup() {
    if (solver_) {
        osqp_cleanup(solver_);
        solver_ = nullptr;
    }
    if (settings_) {
        free(settings_);
        settings_ = nullptr;
    }
    initialized_ = false;
}

void CBFSolver::Reset() {
    Cleanup();
    last_n_constraints_ = 0;
}

int CBFSolver::BuildConstraints(
    int agent_idx,
    Vec3 v_curr,
    const std::vector<Vec3>& pos,
    const std::vector<Vec3>& vel,
    float v_max,
    float a_max,
    float dt,
    const CBFConfig& cfg)
{
    A_data_.clear();
    A_indices_.clear();
    l_.clear();
    u_.clear();
    
    Vec3 p = pos[agent_idx];
    int row = 0;
    
    // Drone-drone CBF constraints
    for (size_t j = 0; j < pos.size(); j++) {
        if ((int)j == agent_idx) continue;
        
        Vec3 d = sub(p, pos[j]);
        float dist2 = dot(d, d);
        
        if (dist2 >= cfg.neighbor_range * cfg.neighbor_range) continue;
        
        float h = dist2 - cfg.safety_radius * cfg.safety_radius;
        Vec3 grad = mul(d, 2.0f);
        
        // Add constraint row
        A_data_.push_back(grad.x);
        A_data_.push_back(grad.y);
        A_data_.push_back(grad.z);
        A_indices_.push_back(row * 3 + 0);
        A_indices_.push_back(row * 3 + 1);
        A_indices_.push_back(row * 3 + 2);
        
        float rhs = -cfg.alpha * h + 2.0f * dot(d, vel[j]);
        l_.push_back(rhs);
        u_.push_back(OSQP_INFTY);
        row++;
    }
    
    // Velocity limits
    for (int col = 0; col < 3; col++) {
        A_data_.push_back(1.0f);
        A_indices_.push_back(row * 3 + col);
        l_.push_back(-v_max);
        u_.push_back(v_max);
        row++;
    }
    
    // Acceleration limits
    float a_lim = a_max * dt;
    float v_arr[3] = {v_curr.x, v_curr.y, v_curr.z};
    for (int col = 0; col < 3; col++) {
        A_data_.push_back(1.0f);
        A_indices_.push_back(row * 3 + col);
        l_.push_back(v_arr[col] - a_lim);
        u_.push_back(v_arr[col] + a_lim);
        row++;
    }
    
    return row;
}

void CBFSolver::InitSolver(int n_constraints) {
    Cleanup();
    
    if (n_constraints == 0 || A_data_.empty()) {
        std::cerr << "[CBF] No constraints to solve\n";
        return;
    }
    
    // P matrix (constant diagonal for quadratic cost)
    OSQPFloat P_x[3] = {2.0, 2.0, 2.0};
    OSQPInt P_i[3] = {0, 1, 2};
    OSQPInt P_p[4] = {0, 1, 2, 3};
    
    OSQPCscMatrix P_matrix;
    P_matrix.m = 3;
    P_matrix.n = 3;
    P_matrix.nzmax = 3;
    P_matrix.nz = -1;
    P_matrix.p = P_p;
    P_matrix.i = P_i;
    P_matrix.x = P_x;
    
    // Convert to column-sparse format
    std::vector<OSQPInt> col_counts(3, 0);
    for (size_t k = 0; k < A_indices_.size(); k++) {
        int col = A_indices_[k] % 3;
        col_counts[col]++;
    }
    
    std::vector<OSQPInt> A_p(4);
    A_p[0] = 0;
    for (int i = 0; i < 3; i++) {
        A_p[i+1] = A_p[i] + col_counts[i];
    }
    
    std::vector<OSQPInt> A_p_csc_;
    std::vector<OSQPInt> A_i_csc_;
    std::vector<OSQPFloat> A_x_csc_;

    
    A_p_csc_.assign(4, 0);
    A_x_csc_.assign(A_data_.size(), 0);
    A_i_csc_.assign(A_indices_.size(), 0);

    // build A_p_csc_ from col_counts...
    A_p_csc_[0] = 0;
    for (int c = 0; c < 3; c++) A_p_csc_[c+1] = A_p_csc_[c] + col_counts[c];

    // fill using your loop (just swap names)
    std::vector<OSQPInt> positions(3, 0);
    for (size_t k = 0; k < A_indices_.size(); k++) {
        int idx = A_indices_[k];
        int row = idx / 3;
        int col = idx % 3;
        int pos = A_p_csc_[col] + positions[col]++;
        A_x_csc_[pos] = A_data_[k];
        A_i_csc_[pos] = row;
    }
    
    OSQPCscMatrix A_matrix;
    A_matrix.m = n_constraints;
    A_matrix.n = 3;
    A_matrix.nzmax = A_data_.size();
    A_matrix.nz = -1;
    A_matrix.p = A_p_csc_.data();
    A_matrix.i = A_i_csc_.data();
    A_matrix.x = A_x_csc_.data();
    
    // Settings
    settings_ = (OSQPSettings*)malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings_);
    settings_->verbose = config_.verbose ? 1 : 0;
    settings_->warm_starting = 1;
    // settings_->polishing = 1;
    settings_->max_iter = 4000;
    settings_->eps_abs = 1e-3;
    settings_->eps_rel = 1e-3;
    
    // Setup
    OSQPInt exitflag = osqp_setup(&solver_, &P_matrix, q_.data(), &A_matrix,
                                  l_.data(), u_.data(),
                                  n_constraints, 3, settings_);
    
    if (exitflag != 0) {
        std::cerr << "[CBF] Setup failed: " << exitflag << "\n";
        initialized_ = false;
        return;
    }
    
    initialized_ = true;
    last_n_constraints_ = n_constraints;
}

Vec3 CBFSolver::Solve(
    int agent_idx,
    Vec3 v_nom,
    Vec3 v_curr,
    const std::vector<Vec3>& positions,
    const std::vector<Vec3>& velocities,
    float v_max,
    float a_max,
    float dt,
    const CBFConfig& config)
{
    config_ = config;  // Store config for InitSolver
    
    // Build constraints
    int n_con = BuildConstraints(agent_idx, v_curr, positions, velocities,
                                 v_max, a_max, dt, config);
    
    int n_cbf = n_con - 6;  // Subtract vel + accel constraints
    
    // Early exit
    if (n_cbf == 0) {
        Vec3 v = clampLen(v_nom, v_max);
        float a_lim = a_max * dt;
        v.x = clamp(v.x, v_curr.x - a_lim, v_curr.x + a_lim);
        v.y = clamp(v.y, v_curr.y - a_lim, v_curr.y + a_lim);
        v.z = clamp(v.z, v_curr.z - a_lim, v_curr.z + a_lim);
        return v;
    }
    
    // Update objective
    q_[0] = -2.0 * v_nom.x;
    q_[1] = -2.0 * v_nom.y;
    q_[2] = -2.0 * v_nom.z;
    
    // Reinit if needed
    if (!initialized_ || std::abs(n_con - last_n_constraints_) > 2) {
        InitSolver(n_con);
        if (!initialized_) {
            return clampLen(v_nom, v_max);
        }
    } else {
        osqp_update_data_vec(solver_, q_.data(), l_.data(), u_.data());
        InitSolver(n_con); //might be optimizable 
    }
    
    // Solve
    osqp_solve(solver_);
    
    if (!solver_ || !solver_->info) {
        return clampLen(v_nom, v_max);
    }

    const int st = solver_->info->status_val;

    // Accept only solved / solved inaccurate
    if (st != OSQP_SOLVED && st != OSQP_SOLVED_INACCURATE) {
        // IMPORTANT: don't touch solver_->solution->x here
        // Optional debug:
        if (config_.verbose) {
            //std::cerr << "[CBF] OSQP status: " << solver_->info->status << "\n";
        }
        return clampLen(v_nom, v_max);
    }

    if (!solver_->solution || !solver_->solution->x) {
        return clampLen(v_nom, v_max);
    }

    Vec3 v_safe = {
        (float)solver_->solution->x[0],
        (float)solver_->solution->x[1],
        (float)solver_->solution->x[2]
    };
    
    float mag = len(v_safe);
    if (mag > 10.0f * v_max || std::isnan(mag) || std::isinf(mag)) {
        std::cerr << "[CBF] Invalid solution magnitude: " << mag << "\n";
        return clampLen(v_nom, v_max);
    }
    
    return v_safe;
}

// ============================================================================
// CBFManager
// ============================================================================

void CBFManager::Init(int num_agents) {
    solvers_.clear();
    solvers_.resize(num_agents);
}

void CBFManager::Resize(int num_agents) {
    solvers_.resize(num_agents);
}

Vec3 CBFManager::Solve(
    int agent_idx,
    Vec3 v_nom,
    const std::vector<Vec3>& positions,
    const std::vector<Vec3>& velocities,
    float v_max,
    float a_max,
    float dt)
{
    if (agent_idx >= (int)solvers_.size()) return v_nom;
    
    Vec3 v_curr = velocities[agent_idx];
    return solvers_[agent_idx].Solve(agent_idx, v_nom, v_curr,
                                     positions, velocities,
                                     v_max, a_max, dt, config_);
}

void CBFManager::ResetAll() {
    for (auto& s : solvers_) {
        s.Reset();
    }
}