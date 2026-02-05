#include "cbf_solver.h"
#include <cmath>
#include <algorithm>
#include <iostream>

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
    A_data_.reserve(100);
    A_indices_.reserve(100);
    l_.reserve(30);
    u_.reserve(30);
    q_.resize(3);
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
    
    Vec3 pi = pos[agent_idx];
    int row = 0;
    
    // CBF constraints: drone-drone collision avoidance
    // h = ||pi - pj||^2 - r^2
    // hdot = 2*(pi - pj)^T * (vi - vj)
    // Constraint: hdot + alpha*h >= 0
    //   => 2*(pi - pj)^T * vi >= -alpha*h + 2*(pi - pj)^T * vj
    
    for (size_t j = 0; j < pos.size(); j++) {
        if ((int)j == agent_idx) continue;
        
        Vec3 d = sub(pi, pos[j]);
        float dist2 = dot(d, d);
        
        if (dist2 >= cfg.neighbor_range * cfg.neighbor_range) continue;
        
        float h = dist2 - cfg.safety_radius * cfg.safety_radius;
        Vec3 grad = mul(d, 2.0f);
        
        // Row: [2*dx, 2*dy, 2*dz] * v >= rhs
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
    
    // Velocity magnitude limit: -v_max <= v[i] <= v_max
    for (int i = 0; i < 3; i++) {
        A_data_.push_back(1.0f);
        A_indices_.push_back(row * 3 + i);
        l_.push_back(-v_max);
        u_.push_back(v_max);
        row++;
    }
    
    // Acceleration limit: |v - v_curr| <= a_max * dt
    float a_lim = a_max * dt;
    float v_arr[3] = {v_curr.x, v_curr.y, v_curr.z};
    for (int i = 0; i < 3; i++) {
        A_data_.push_back(1.0f);
        A_indices_.push_back(row * 3 + i);
        l_.push_back(v_arr[i] - a_lim);
        u_.push_back(v_arr[i] + a_lim);
        row++;
    }
    
    return row; // total constraints
}

void CBFSolver::SetupSolver(int n_constraints) {
    Cleanup();
    
    // P = 2*I (for objective ||v - v_nom||^2)
    OSQPFloat P_data[3] = {2.0, 2.0, 2.0};
    OSQPInt P_indices[3] = {0, 1, 2};
    OSQPInt P_indptr[4] = {0, 1, 2, 3};
    
    OSQPCscMatrix* P = (OSQPCscMatrix*)malloc(sizeof(OSQPCscMatrix));
    P->m = 3;
    P->n = 3;
    P->p = P_indptr;
    P->i = P_indices;
    P->x = P_data;
    P->nzmax = 3;
    P->nz = -1;
    
    // A matrix (constraints)
    std::vector<OSQPInt> A_indptr(4, 0);
    for (size_t k = 0; k < A_indices_.size(); k++) {
        int col = A_indices_[k] % 3;
        A_indptr[col + 1]++;
    }
    for (int i = 1; i < 4; i++) {
        A_indptr[i] += A_indptr[i-1];
    }
    
    OSQPCscMatrix* A = (OSQPCscMatrix*)malloc(sizeof(OSQPCscMatrix));
    A->m = n_constraints;
    A->n = 3;
    A->p = A_indptr.data();
    A->i = A_indices_.data();
    A->x = A_data_.data();
    A->nzmax = A_data_.size();
    A->nz = -1;
    
    // Settings
    settings_ = (OSQPSettings*)malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings_);
    settings_->verbose = 0;
    settings_->warm_starting = 1;
    settings_->polish = 1;
    settings_->max_iter = 2000;
    settings_->eps_abs = 1e-3;
    settings_->eps_rel = 1e-3;
    
    // Setup
    OSQPInt status = osqp_setup(&solver_, P, q_.data(), A,
                                l_.data(), u_.data(),
                                n_constraints, 3, settings_);
    
    free(P);
    free(A);
    
    if (status != 0) {
        std::cerr << "[CBF] Setup failed\n";
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
    // Build constraints
    int n_con = BuildConstraints(agent_idx, v_curr, positions, velocities,
                                 v_max, a_max, dt, config);
    
    // Early exit if no neighbors
    int n_cbf = n_con - 6; // subtract vel/accel constraints
    if (n_cbf == 0) {
        Vec3 v = clampLen(v_nom, v_max);
        float a_lim = a_max * dt;
        v.x = clamp(v.x, v_curr.x - a_lim, v_curr.x + a_lim);
        v.y = clamp(v.y, v_curr.y - a_lim, v_curr.y + a_lim);
        v.z = clamp(v.z, v_curr.z - a_lim, v_curr.z + a_lim);
        return v;
    }
    
    // Update objective: min ||v - v_nom||^2
    q_[0] = -2.0f * v_nom.x;
    q_[1] = -2.0f * v_nom.y;
    q_[2] = -2.0f * v_nom.z;
    
    // Reinit if constraint count changed significantly
    if (!initialized_ || std::abs(n_con - last_n_constraints_) > 2) {
        SetupSolver(n_con);
        if (!initialized_) return v_nom;
    } else {
        // Just update data (fast path)
        osqp_update_data_vec(solver_, q_.data(), l_.data(), u_.data());
    }
    
    // Solve
    osqp_solve(solver_);
    
    // Extract solution
    if (solver_->solution && solver_->solution->x) {
        return {
            (float)solver_->solution->x[0],
            (float)solver_->solution->x[1],
            (float)solver_->solution->x[2]
        };
    }
    
    return v_nom; // fallback
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