#include "cbf_solver.h"
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>


// Math
static constexpr int kVelVars = 3;
static constexpr int kDecisionVars = 4;  // [vx, vy, vz, slack]
static constexpr int kSlackCol = 3;
static constexpr float kOverlapEps = 1e-4f;

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
static inline Vec3 deterministic_xy_away(int agent_idx, int other_idx) {
    unsigned int h =
        (unsigned int)(agent_idx * 73856093u) ^
        (unsigned int)(other_idx * 19349663u);
    float angle = (float)(h % 1024u) * (6.28318530718f / 1024.0f);
    return {std::cos(angle), std::sin(angle), 0.0f};
}

static inline float push_in_1d(float p, float v,
                              float minb, float maxb,
                              float k_push,
                              float margin)
{
    // Start pushing slightly before the wall if margin > 0
    if (p < minb + margin) {
        // Need positive velocity to re-enter
        float v_in = k_push * (minb + margin - p);
        return std::max(v, v_in);
    }
    if (p > maxb - margin) {
        // Need negative velocity to re-enter
        float v_in = -k_push * (p - (maxb - margin));
        return std::min(v, v_in);
    }
    return v;
}

static inline Vec3 push_inward(Vec3 p, Vec3 v,
                              const CBFConfig& cfg,
                              float v_max,
                              float a_max,
                              float dt,
                              Vec3 v_curr)
{
    // Tunables 
    const float margin = 0.1f;   // how early to start pushing 
    const float k_push = 1.25f;   // how hard to push when outside

    // Push inward per-axis
    v.x = push_in_1d(p.x, v.x, cfg.x_min, cfg.x_max, k_push, margin);
    v.y = push_in_1d(p.y, v.y, cfg.y_min, cfg.y_max, k_push, margin);
    v.z = push_in_1d(p.z, v.z, cfg.z_min, cfg.z_max, k_push, margin);

    // Respect accel limits
    float a_lim = a_max * dt;
    v.x = clamp(v.x, v_curr.x - a_lim, v_curr.x + a_lim);
    v.y = clamp(v.y, v_curr.y - a_lim, v_curr.y + a_lim);
    v.z = clamp(v.z, v_curr.z - a_lim, v_curr.z + a_lim);

    // Respect speed cap
    v = clampLen(v, v_max);
    return v;
}

// ============================================================================
// CBFSolver
// ============================================================================

CBFSolver::CBFSolver() {
    q_.resize(kDecisionVars, 0.0);
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
    last_A_nnz_ = 0;
    last_A_p_csc_.clear();
    last_A_i_csc_.clear();
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
    float a_lim = a_max * dt;
    
    for (size_t j = 0; j < pos.size(); j++) {
        if ((int)j == agent_idx) continue;
        
        Vec3 d = sub(p, pos[j]);
        float dist2 = dot(d, d);
        if (dist2 >= cfg.neighbor_range * cfg.neighbor_range) continue;
        
        float h = dist2 - cfg.safety_radius * cfg.safety_radius;
        
        // Skip if safe and already diverging — constraint not needed,
        // and including it risks infeasibility for no benefit
        Vec3 v_rel = sub(vel[agent_idx], vel[j]);
        float h_dot = 2.0f * dot(d, v_rel);
        if (h > 0.0f && h_dot >= 0.0f) continue;

        Vec3 grad = mul(d, 2.0f);  // gradient of h w.r.t. v_i
        
        A_data_.push_back(grad.x);
        A_data_.push_back(grad.y);
        A_data_.push_back(grad.z);
        A_data_.push_back(1.0f);  // Relaxation slack.
        A_indices_.push_back(row * kDecisionVars + 0);
        A_indices_.push_back(row * kDecisionVars + 1);
        A_indices_.push_back(row * kDecisionVars + 2);
        A_indices_.push_back(row * kDecisionVars + kSlackCol);
        
        // j moving away relaxes constraint — don't let it tighten it
        float j_contrib = std::min(2.0f * dot(d, vel[j]), 0.0f);
        float rhs = -cfg.alpha * h + j_contrib;

        // KEY FIX: clamp rhs so it never exceeds what acceleration limits allow.
        // The maximum dot(grad, v) achievable is |grad| * v_max (by Cauchy-Schwarz).
        // If rhs exceeds this, the constraint is physically impossible to satisfy.
        float grad_norm = 2.0f * std::sqrt(dist2);  // |grad| = 2*dist
        float max_achievable = grad_norm * v_max;
        rhs = std::min(rhs, max_achievable * 0.95f);  // 5% margin

        // Also clamp against accel limit: max dot(grad/|grad|, v) reachable in one step
        // is roughly current outward speed + a_lim
        float v_outward_curr = dot(grad, Vec3{v_curr.x, v_curr.y, v_curr.z}) / (grad_norm + 1e-6f);
        float max_accel_achievable = (v_outward_curr + a_lim) * grad_norm;
        rhs = std::min(rhs, max_accel_achievable * 0.95f);

        l_.push_back(rhs);
        u_.push_back(OSQP_INFTY);
        row++;


        //keep them separated horizontally
        const float r_xy = cfg.safety_radius * 1.1f;
        const float dx = d.x;
        const float dy = d.y;
        const float dist2_xy = dx*dx + dy*dy;

        // Use an XY-specific neighbor gate so stacked drones get caught
        const float neigh2_xy = cfg.neighbor_range * cfg.neighbor_range;
        if (dist2_xy < neigh2_xy) {
            if (dist2_xy <= kOverlapEps * kOverlapEps) {
                continue;
            }

            float h_xy = dist2_xy - r_xy * r_xy;

            // hdot_xy = 2*[dx,dy]·(v_i - v_j)_xy
            float dvx = vel[agent_idx].x - vel[j].x;
            float dvy = vel[agent_idx].y - vel[j].y;
            float hdot_xy = 2.0f * (dx*dvx + dy*dvy);

            // same skip logic: safe and separating -> ignore
            if (!(h_xy > 0.0f && hdot_xy >= 0.0f)) {

                // grad wrt v_i is [2dx, 2dy, 0] (don't store z since it's 0)
                A_data_.push_back(2.0f * dx);
                A_data_.push_back(2.0f * dy);
                A_data_.push_back(1.0f);  // Relaxation slack.
                A_indices_.push_back(row * kDecisionVars + 0);
                A_indices_.push_back(row * kDecisionVars + 1);
                A_indices_.push_back(row * kDecisionVars + kSlackCol);

                // j_contrib in XY only
                float j_contrib_xy = std::min(2.0f * (dx*vel[j].x + dy*vel[j].y), 0.0f);
                float rhs_xy = -cfg.alpha * h_xy + j_contrib_xy;

                // Clamp feasibility like your 3D version
                float gradn_xy = 2.0f * std::sqrt(dist2_xy);
                float max_xy = gradn_xy * v_max;
                rhs_xy = std::min(rhs_xy, max_xy * 0.95f);

                float v_out_xy_curr =
                    ( (2.0f*dx)*v_curr.x + (2.0f*dy)*v_curr.y ) / (gradn_xy + 1e-6f);
                float max_accel_xy = (v_out_xy_curr + a_lim) * gradn_xy;
                rhs_xy = std::min(rhs_xy, max_accel_xy * 0.95f);

                l_.push_back(rhs_xy);
                u_.push_back(OSQP_INFTY);
                row++;
            }
        }
    }
    
    // Velocity limits
    for (int col = 0; col < kVelVars; col++) {
        A_data_.push_back(1.0f);
        A_indices_.push_back(row * kDecisionVars + col);
        l_.push_back(-v_max);
        u_.push_back(v_max);
        row++;
    }
    
    // Acceleration limits
    float v_arr[kVelVars] = {v_curr.x, v_curr.y, v_curr.z};
    for (int col = 0; col < kVelVars; col++) {
        A_data_.push_back(1.0f);
        A_indices_.push_back(row * kDecisionVars + col);
        l_.push_back(v_arr[col] - a_lim);
        u_.push_back(v_arr[col] + a_lim);
        row++;
    }

    const float px[kVelVars] = {p.x, p.y, p.z};
    const float min_b[kVelVars] = {cfg.x_min, cfg.y_min, cfg.z_min};
    const float max_b[kVelVars] = {cfg.x_max, cfg.y_max, cfg.z_max};

    for (int col = 0; col < kVelVars; col++) {
        float alpha_wall = (col == 2) ? cfg.alpha * 3.0f : cfg.alpha;

        // Clamp wall constraints the same way — they can also conflict with accel limits
        float rhs_lo = -alpha_wall * (px[col] - min_b[col]);
        float rhs_hi = -alpha_wall * (max_b[col] - px[col]);

        // Never demand more than accel limits can deliver
        rhs_lo = std::min(rhs_lo,  v_arr[col] + a_lim);
        rhs_hi = std::min(rhs_hi, -v_arr[col] + a_lim);  // upper wall uses -v

        A_data_.push_back(1.0f);
        A_data_.push_back(1.0f);  // Relaxation slack.
        A_indices_.push_back(row * kDecisionVars + col);
        A_indices_.push_back(row * kDecisionVars + kSlackCol);
        l_.push_back(rhs_lo);
        u_.push_back(OSQP_INFTY);
        row++;

        A_data_.push_back(-1.0f);
        A_data_.push_back(1.0f);  // Relaxation slack.
        A_indices_.push_back(row * kDecisionVars + col);
        A_indices_.push_back(row * kDecisionVars + kSlackCol);
        l_.push_back(rhs_hi);
        u_.push_back(OSQP_INFTY);
        row++;
    }

    // Keep slack bounded so violations remain limited.
    A_data_.push_back(1.0f);
    A_indices_.push_back(row * kDecisionVars + kSlackCol);
    l_.push_back(0.0f);
    u_.push_back(cfg.slack_max);
    row++;
    
    return row;
}

void CBFSolver::InitSolver(int n_constraints) {
    Cleanup();
    
    if (n_constraints == 0 || A_data_.empty() || A_x_csc_.empty()) {
        std::cerr << "[CBF] No constraints to solve\n";
        return;
    }
    
    // P matrix (diagonal quadratic cost for velocity + slack penalty).
    OSQPFloat P_x[4] = {2.0, 2.0, 2.0, 2.0 * config_.slack_weight};
    OSQPInt P_i[4] = {0, 1, 2, 3};
    OSQPInt P_p[5] = {0, 1, 2, 3, 4};
    
    /*
    P= [2, 0, 0
        0, 2, 0
        0, 0, 2]
    */

    OSQPCscMatrix P_matrix;
    P_matrix.m = kDecisionVars;
    P_matrix.n = kDecisionVars;
    P_matrix.nzmax = kDecisionVars;
    P_matrix.nz = -1;
    P_matrix.p = P_p;
    P_matrix.i = P_i;
    P_matrix.x = P_x;
    
    OSQPCscMatrix A_matrix;
    A_matrix.m = n_constraints;
    A_matrix.n = kDecisionVars;
    A_matrix.nzmax = (OSQPInt)A_x_csc_.size();
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
                                  n_constraints, kDecisionVars, settings_);
    
    if (exitflag != 0) {
        std::cerr << "[CBF] Setup failed: " << exitflag << "\n";
        initialized_ = false;
        return;
    }
    
    initialized_ = true;
    last_n_constraints_ = n_constraints;
    last_A_nnz_ = A_x_csc_.size();
    last_A_p_csc_ = A_p_csc_;
    last_A_i_csc_ = A_i_csc_;
}

void CBFSolver::BuildCSC(int n_constraints) {
    std::vector<OSQPInt> col_counts(kDecisionVars, 0);
    for (size_t k = 0; k < A_indices_.size(); k++) {
        col_counts[A_indices_[k] % kDecisionVars]++;
    }

    A_p_csc_.resize(kDecisionVars + 1);
    A_p_csc_[0] = 0;
    for (int c = 0; c < kDecisionVars; c++) {
        A_p_csc_[c+1] = A_p_csc_[c] + col_counts[c];
    }

    A_x_csc_.assign(A_data_.size(), 0.0);
    A_i_csc_.assign(A_data_.size(), 0);

    std::vector<OSQPInt> positions(kDecisionVars, 0);
    for (size_t k = 0; k < A_indices_.size(); k++) {
        int row = A_indices_[k] / kDecisionVars;
        int col = A_indices_[k] % kDecisionVars;
        int pos = A_p_csc_[col] + positions[col]++;
        A_x_csc_[pos] = A_data_[k];
        A_i_csc_[pos] = row;
    }
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
    int n_con = BuildConstraints(agent_idx, v_curr, positions, velocities, v_max, a_max, dt, config);

    BuildCSC(n_con);
    
    int n_cbf = n_con - kVelVars - kVelVars - 2 * kVelVars - 1;  // vel + accel + wall + slack-box
    
    // Early exit
    if (n_cbf <= 0) {
        Vec3 v = clampLen(v_nom, v_max);
        v = push_inward(positions[agent_idx], v, config, v_max, a_max, dt, v_curr);
        return v;
    }
    
    // Update objective
    q_[0] = -2.0 * v_nom.x;
    q_[1] = -2.0 * v_nom.y;
    q_[2] = -2.0 * v_nom.z;
    q_[3] = 0.0;
    
    // Reinit if needed
    bool structure_changed = (A_x_csc_.size() != last_A_nnz_) ||
                             (A_p_csc_ != last_A_p_csc_) ||
                             (A_i_csc_ != last_A_i_csc_);

    if (!initialized_ || n_con != last_n_constraints_ || structure_changed) {
        InitSolver(n_con);
        if (!initialized_) {
            return clampLen(v_nom, v_max);
        }
    } else {
        osqp_update_data_vec(solver_, q_.data(), l_.data(), u_.data());
        osqp_update_data_mat(solver_,
                     nullptr, nullptr, 0,
                     A_x_csc_.data(), nullptr, (OSQPInt)A_x_csc_.size());
        //InitSolver(n_con); //might be optimizable 
    }
    
    // Solve
    osqp_solve(solver_);
    
    if (!solver_ || !solver_->info) {
        Vec3 v = clampLen(v_nom, v_max);
        return push_inward(positions[agent_idx], v, config, v_max, a_max, dt, v_curr);
    }

    const int st = solver_->info->status_val;

    // // Accept only solved / solved inaccurate
    // if (st != OSQP_SOLVED && st != OSQP_SOLVED_INACCURATE) {
    //     // IMPORTANT: don't touch solver_->solution->x here
    //     // Optional debug:
    //     Vec3 v = clampLen(v_nom, v_max);
    //     return push_inward(positions[agent_idx], v, config, v_max, a_max, dt, v_curr);
    // }

    if (st != OSQP_SOLVED && st != OSQP_SOLVED_INACCURATE) {
        // Don't just return v_nom — actively push away from nearest neighbor.
        Vec3 v_emergency = {0, 0, 0};
        float closest = std::numeric_limits<float>::max();
        int overlap_neighbor = -1;
        for (size_t j = 0; j < positions.size(); j++) {
            if ((int)j == agent_idx) continue;
            Vec3 d = sub(positions[agent_idx], positions[j]);
            float dist = len(d);
            float dist2_xy = d.x * d.x + d.y * d.y;
            if (dist < closest && dist > kOverlapEps) {
                closest = dist;
                if (dist2_xy <= kOverlapEps * kOverlapEps) {
                    Vec3 away_xy = deterministic_xy_away(agent_idx, (int)j);
                    v_emergency = mul(away_xy, config.recovery_speed);
                } else {
                    v_emergency = mul(d, config.recovery_speed / dist);
                }
            }
            if (dist <= kOverlapEps && overlap_neighbor < 0) {
                overlap_neighbor = (int)j;
            }
        }
        if (closest == std::numeric_limits<float>::max() && overlap_neighbor >= 0) {
            Vec3 away = deterministic_xy_away(agent_idx, overlap_neighbor);
            v_emergency = mul(away, config.recovery_speed);
        }
        return clampLen(v_emergency, v_max);
    }

    if (!solver_->solution || !solver_->solution->x) {
        Vec3 v = clampLen(v_nom, v_max);
        return push_inward(positions[agent_idx], v, config, v_max, a_max, dt, v_curr);
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
    
    for (size_t j = 0; j < positions.size(); j++) {
        if ((int)j == agent_idx) continue;
        Vec3 d = sub(positions[agent_idx], positions[j]);
        float dist = len(d);
        if (dist < config.safety_radius * 0.9f) {
            float dist2_xy = d.x * d.x + d.y * d.y;
            Vec3 away = (dist2_xy > kOverlapEps * kOverlapEps && dist > kOverlapEps)
                ? mul(d, 1.0f / dist)
                : deterministic_xy_away(agent_idx, (int)j);
            float v_out = std::max(dot(v_safe, away), config.recovery_speed);
            v_safe.x += away.x * v_out;
            v_safe.y += away.y * v_out;
            v_safe.z += away.z * v_out;
            v_safe = clampLen(v_safe, v_max);
        }
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
