// boids.cpp - boids implementation for Drone Swarm Visualizer(basicVisuals.cpp)
// Implements emotion paramters via clap_weights.json
// REQUIRES: boids.h, json.hpp

//////////////////////////////////////////////////////////
// --- Boids (Notes) ---
// 1. Numbers/params are in meters and seconds, but are not real units, just make sure vmax and amax make sense for dt size
// 2. To prevent slowing down when increased number of drones, it only checks some nieghbors instead of every pair
// 3. Drones are pushed towards set alt to keep from too much drifting
// 4. Limitations on acceleration is to keep visualizer from blowing up
/////////////////////////////////////////////////////////

#include "boids.h"
#include "json.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>


static std::vector<Vec3> gPos, gVel, gAcc;
static std::vector<float> gLastWeights;
static std::vector<std::string> EMO_LABELS;
static bool gLabelsLoaded = false;

static inline float clampf(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
}

static float gTime = 0.0f;

void SetSimTime(float t){
    gTime = t;
}

static const BoidParams Neutral = {        //original neutral parameters
    /* r_sep */ 0.6f,  /* r_nei */ 2.5f,
    /* k_sep */ 1.0f,  /* k_ali */ 1.0f,  /* k_coh */ 0.7f,  /* k_goal */ 1.1f,
    /* vmax  */ 6.0f,  /* amax  */ 12.0f,
    /* altitude */ 1.7f, /* jitter */ 0.25f
};
static BoidParams P = Neutral;

//resetting
std::vector<float> gResetTimes;
static int   gNextResetIndex = 0;
static bool  gResetTimesInit = false;
static float gLastBoidsTime  = 0.0f;


struct StyleParams {
    float spin_rate;        // yaw rotation around goal
    float bob_amp;          // vertical bobbing amplitude
    float bob_freq;         // vertical bobbing frequency
    float pause_prob;       // chance to briefly freeze / hesitate
    float turn_sharpness;   // extra curvature in paths
};

struct ParamDelta {     //For easier batch update
    float r_sep    = 0.0f;
    float r_nei    = 0.0f;
    float k_sep    = 0.0f;
    float k_ali    = 0.0f;
    float k_coh    = 0.0f;
    float k_goal   = 0.0f;
    float vmax     = 0.0f;
    float amax     = 0.0f;
    float altitude = 0.0f;
    float jitter   = 0.0f;
};


//bound
static const float X_MIN = -5.0f;
static const float X_MAX =  5.0f;
static const float Z_MIN = -5.0f;
static const float Z_MAX =  5.0f;
static const float Y_MIN = 0.5f;
static const float Y_MAX = 5.0f;

Boundaries GetBoxBounds() {
    return Boundaries{ X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX };
}

static const ParamDelta ANCHORS[7] = {
    {
        /* r_sep   */ -0.15f,   // likes being close
        /* r_nei   */ +0.20f,   // sees neighbors a bit further out
        /* k_sep   */ -0.30f,   // less "get away from me"
        /* k_ali   */ +0.40f,   // strongly aligns with the group
        /* k_coh   */ +0.30f,   // wants to flock
        /* k_goal  */ +0.30f,   // reacts quickly to goals/targets
        /* vmax    */ +2.0f,    // faster than neutral (hero: 7.7 mph vs others)
        /* amax    */ +4.0f,    // snappier acceleration
        /* altitude*/ +0.70f,   // flies higher
        /* jitter  */ +0.15f    // playful / excited
    },

    /* 1: sad (Exhausted Drone) */
    {
        /* r_sep   */  0.00f,   // not especially pushy
        /* r_nei   */ -0.40f,   // pays attention to a smaller neighborhood
        /* k_sep   */ -0.20f,   // doesn't strongly avoid others
        /* k_ali   */ -0.40f,   // poor alignment, kind of drifts
        /* k_coh   */ -0.10f,   // mild cohesion, it's just "there"
        /* k_goal  */ -0.45f,   // slow to respond to commands
        /* vmax    */ -3.0f,    // much slower (dragging)
        /* amax    */ -7.0f,    // weak acceleration
        /* altitude*/ -0.70f,   // stays low
        /* jitter  */ +0.10f    // slight wobble / tired shakiness
    },

    /* 2: sleepy (Exhausted Drone variant) */
    {
        /* r_sep   */  0.00f,
        /* r_nei   */ -0.40f,
        /* k_sep   */ -0.25f,   // even less separation, kind of "slumps" into group
        /* k_ali   */ -0.45f,
        /* k_coh   */ -0.05f,
        /* k_goal  */ -0.55f,   // even slower to act than "sad"
        /* vmax    */ -3.5f,    // really slow
        /* amax    */ -8.0f,    // sluggish
        /* altitude*/ -0.80f,   // very low
        /* jitter  */ +0.05f    // more "heavy" than wobbly
    },

    /* 3: brave (Adventurer Hero) */
    {
        /* r_sep   */ -0.10f,   // still comfortable close
        /* r_nei   */ +0.20f,
        /* k_sep   */ -0.20f,
        /* k_ali   */ +0.35f,
        /* k_coh   */ +0.25f,
        /* k_goal  */ +0.35f,   // very direct in executing commands
        /* vmax    */ +2.2f,    // slightly faster than "happy"
        /* amax    */ +4.5f,
        /* altitude*/ +0.70f,
        /* jitter  */ +0.10f    // brave more than "giggly"
    },

    /* 4: grumpy (Anti-Social Drone) */
    {
        /* r_sep   */ +0.35f,   // keeps its distance
        /* r_nei   */ -0.50f,   // narrow social radius
        /* k_sep   */ +0.40f,   // strong "get away from me"
        /* k_ali   */ -0.30f,   // doesn't like aligning
        /* k_coh   */ -0.35f,   // low cohesion, hangs back
        /* k_goal  */ -0.30f,   // reluctant to follow commands
        /* vmax    */ -1.0f,    // a bit slower
        /* amax    */ -3.0f,    // drags into motion
        /* altitude*/  0.00f,   // middle altitude
        /* jitter  */ +0.10f    // a little twitchy / begrudging
    },

    /* 5: scared (Sneaky Spy Drone) */
    {
        /* r_sep   */ +0.50f,   // stays far from others
        /* r_nei   */  0.00f,   // normal neighbor radius
        /* k_sep   */ +0.60f,   // strong avoidance
        /* k_ali   */ +0.10f,   // slightly aligns when fleeing
        /* k_coh   */ -0.20f,   // doesn't like the group
        /* k_goal  */ -0.35f,   // hesitant to go directly to goal
        /* vmax    */ +0.5f,    // quick but not the fastest
        /* amax    */ +2.0f,    // sharp jerks / nervous moves
        /* altitude*/ -0.60f,   // low altitude
        /* jitter  */ +0.30f    // very jittery -> "looking around"
    },

    /* 6: shy (Anti-Social-ish but less harsh) */
    {
        /* r_sep   */ +0.25f,   // keeps a bit of distance
        /* r_nei   */ -0.30f,
        /* k_sep   */ +0.25f,
        /* k_ali   */ -0.20f,
        /* k_coh   */ -0.20f,
        /* k_goal  */ -0.20f,   // needs coaxing
        /* vmax    */ -0.8f,    // slightly slower
        /* amax    */ -2.5f,
        /* altitude*/ -0.10f,   // slightly lower than neutral
        /* jitter  */ +0.15f    // anxious
    }
};

// --- JSON / Emotion Labeleling --- //

using json = nlohmann::json;
static json EMO;                //holds loaded json array
static bool gEmoLoaded = false; //only load once

static inline void clampToBox(Vec3& p) {
    p.x = clampf(p.x, X_MIN, X_MAX);
    p.y = clampf(p.y, Y_MIN, Y_MAX);
    p.z = clampf(p.z, Z_MIN, Z_MAX);
}

static inline void applyBoxConstraint(Vec3& p, Vec3& v) {
    // X walls
    if (p.x < X_MIN) {
        p.x = X_MIN;
        if (v.x < 0.0f) v.x = -v.x;   // bounce
    } else if (p.x > X_MAX) {
        p.x = X_MAX;
        if (v.x > 0.0f) v.x = -v.x;
    }

    // Y walls
    if (p.y < Y_MIN) {
        p.y = Y_MIN;
        if (v.y < 0.0f) v.y = -v.y;
    } else if (p.y > Y_MAX) {
        p.y = Y_MAX;
        if (v.y > 0.0f) v.y = -v.y;
    }

    // Z walls
    if (p.z < Z_MIN) {
        p.z = Z_MIN;
        if (v.z < 0.0f) v.z = -v.z;
    } else if (p.z > Z_MAX) {
        p.z = Z_MAX;
        if (v.z > 0.0f) v.z = -v.z;
    }
}


bool LoadEmotionLabels(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;

    json J;
    f >> J;

    if (!J.is_array()) return false;

    EMO_LABELS.clear();
    for (auto& item : J) {
        if (item.is_string()) {
            EMO_LABELS.push_back(item.get<std::string>());
        }
    }

    std::cerr << "Loaded labels (" << EMO_LABELS.size() << "): ";
    for (auto& s : EMO_LABELS) std::cerr << s << " ";
    std::cerr << "\n";

    return true;
}

const std::vector<std::string>& GetEmotionLabels(){
    return EMO_LABELS;
}

std::vector<float> GetEmotionWeights(float t){
    if(!EMO.is_array() || EMO.empty()) return {};
    for (auto& seg :EMO){
        float s = seg.value("start", 0.0f);
        float e = seg.value("end", 0.0f);
        if (t >= s && t < e) {
            if (seg.contains("weights")) return seg["weights"].get<std::vector<float>>();
            return {};
        }
    }
    //hold last emotion once its over
    auto& last = EMO.back();
    if(last.contains("weights")) return last["weights"].get<std::vector<float>>();
    return{};
}


static inline float clamp01(float x){ return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
static inline float map01(float x, float a, float b){ x = clamp01(x); return a + (b - a) * x; }


void ApplyEmotion(const std::vector<float>& w, BoidParams& P) {
    float e = 0.0f;
    if (!w.empty() && std::isfinite(w[0])) e = w[0];

    // Keep in reasonable range
    e = clampf(e, -1.0f, 1.0f);

    float new_vmax = 6.0f + 3.0f * e;                 // faster with higher e
    float new_ksep = 1.0f + 1.5f * std::max(0.0f, e); // stronger separation when e>0

    // clamps so swarm stays cohesive
    P.vmax = clampf(new_vmax, 0.5f, 12.0f);
    P.k_sep = clampf(new_ksep, 0.1f, 2.5f);
}

void ApplyEmotionHard(const std::vector<float>& w, BoidParams& P)
{
    if (w.empty()) return;

    // Accumulate weighted deltas
    ParamDelta D{};
    for (int i = 0; i < 7; ++i) {
        const ParamDelta& A = ANCHORS[i];
        D.r_sep    += w[i] * A.r_sep;
        D.r_nei    += w[i] * A.r_nei;
        D.k_sep    += w[i] * A.k_sep;
        D.k_ali    += w[i] * A.k_ali;
        D.k_coh    += w[i] * A.k_coh;
        D.k_goal   += w[i] * A.k_goal;
        D.vmax     += w[i] * A.vmax;
        D.amax     += w[i] * A.amax;
        D.altitude += w[i] * A.altitude;
        D.jitter   += w[i] * A.jitter;
    }

    // Apply exact preset deltas on current P (no lerp)
    P.r_sep    = clampf(P.r_sep    + D.r_sep,    0.10f,  5.0f);
    P.r_nei    = clampf(P.r_nei    + D.r_nei,    0.50f, 10.0f);
    P.k_sep    = clampf(P.k_sep    + D.k_sep,    0.10f,  2.5f);
    P.k_ali    = clampf(P.k_ali    + D.k_ali,    0.00f,  3.0f);
    P.k_coh    = clampf(P.k_coh    + D.k_coh,    0.00f,  3.0f);
    P.k_goal   = clampf(P.k_goal   + D.k_goal,   0.20f,  3.0f);
    P.vmax     = clampf(P.vmax     + D.vmax,     0.50f, 12.0f);
    P.amax     = clampf(P.amax     + D.amax,     1.00f, 20.0f);
    P.altitude = clampf(P.altitude + D.altitude, 0.50f, 10.0f);
    P.jitter   = clampf(P.jitter   + D.jitter,   0.00f,  1.0f);
}

void LoadResetTimes(const std::string& filename)
{
    namespace fs = std::filesystem;
    fs::path cwd  = fs::current_path();
    fs::path path = cwd.parent_path() / "json" / filename;
    std::string pathStr = path.string();

    gResetTimes.clear();
    gNextResetIndex = 0;

    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Failed to open reset-times JSON: " << filename << "\n";
        return;
    }

    json j;
    in >> j;

    // j should be an array of [start, end]
    for (const auto& seg : j) {
        if (!seg.is_array() || seg.size() < 2) continue;
        float endTime = seg[1].get<float>();
        gResetTimes.push_back(endTime);
    }

    std::cerr << "Loaded " << gResetTimes.size()
              << " reset times from " << filename << "\n";
}

bool LoadEmotionFile(const std::string& path){
    std::ifstream f(path);
    if (!f) return false;
    json J; f >> J;
    if (!J.is_array()) return false;
    EMO = std::move(J);
    return true;
}

static std::filesystem::path find_json_file(const std::string &filename) {
    namespace fs = std::filesystem;
    fs::path cwd = fs::current_path();

    // look from these folder:
    std::vector<fs::path> candidates = {
        cwd / "json" / filename,
        cwd.parent_path() / "json" / filename,
    };

    for (const auto &p : candidates) {
        std::cerr << "[DEBUG] trying: " << p.string() << "\n";

        if (fs::exists(p)) {
            return p;
        }
    }

    return {};  // empty path = not found
}

void EnsureEmotionsLoaded() {

    if (!gEmoLoaded){
        auto clap_path = find_json_file("clap_weights.json");
        bool ok = LoadEmotionFile(clap_path.string());

        if (clap_path.empty()) {
            std::cerr << "ERROR: Could not find clap_weights.json\n";
            gEmoLoaded = false;
            return;
        }

        if (!ok) {
            std::cerr << "ERROR: Could not load clap weights file\n"
                      << clap_path.string() << "\n";  
        } else {
            std::cerr << "Loaded JSON successfully. #Segments = " 
                    << EMO.size() << "\n";
        }
        gEmoLoaded = ok;
    }

    if (!gLabelsLoaded) {
        auto anchor_path = find_json_file("anchor_labels.json");
        bool ok = LoadEmotionLabels(anchor_path.string());

        if (anchor_path.empty()) {
            std::cerr << "ERROR: Could not find anchor_labels.json\n";
            gEmoLoaded = false;
            return;
        }

        if (!ok) {
            std::cerr << "ERROR: Could not load anchor_labels.json\n";
        } else {
            std::cerr << "Loaded JSON successfully." << "\n";
        }
        gLabelsLoaded = ok;
    }
}



bool ReloadAndApplyEmotions(float t) {
    // // try local path first, then /mnt/data as fallback
    // bool ok = LoadEmotionFile("clap_weights.json");
    // if (!ok) ok = LoadEmotionFile("/mnt/data/clap_weights.json");
    // if (!ok) return false;

    // auto w = GetEmotionWeights(t);
    // if (w.empty()) return false;

    // // mutate current params
    // BoidParams& Pmut = const_cast<BoidParams&>(GetBoidParams());
    // ApplyEmotionHard(w, P);
    return true;
}


float GetAudioLength(){
    EnsureEmotionsLoaded();
    if(!EMO.is_array() || EMO.empty()) {
        std::cerr << "[DEBUG] EMO is empty or not an array\n";
        return 0.0f;
    }

    const auto& last = EMO.back();
    return last.value("end", 0.0f);
}

// static BoidParams PFrom = P, PTo = P;
// static float gTransLeft = 0.0f, gTransTotal = 0.0f;
// static std::string gEmotion = "neutral";

// static inline float lerp(float a, float b, float t){ return a + (b - a) * t; }
// static inline BoidParams lerpParams(const BoidParams& A, const BoidParams& B, float t){
//     BoidParams R = A;
//     R.r_sep   = lerp(A.r_sep,   B.r_sep,   t);
//     R.r_nei   = lerp(A.r_nei,   B.r_nei,   t);
//     R.k_sep   = lerp(A.k_sep,   B.k_sep,   t);
//     R.k_ali   = lerp(A.k_ali,   B.k_ali,   t);
//     R.k_coh   = lerp(A.k_coh,   B.k_coh,   t);
//     R.k_goal  = lerp(A.k_goal,  B.k_goal,  t);
//     R.vmax    = lerp(A.vmax,    B.vmax,    t);
//     R.amax    = lerp(A.amax,    B.amax,    t);
//     R.altitude= lerp(A.altitude,B.altitude,t);
//     R.jitter  = lerp(A.jitter,  B.jitter,  t);
//     return R;
// }


// current params for HUD/debug
const BoidParams& GetBoidParams(){
    return P;
}



// math help
static inline Vec3 add(Vec3 a, Vec3 b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; }
static inline Vec3 sub(Vec3 a, Vec3 b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
static inline Vec3 mul(Vec3 a, float s){ return {a.x*s,a.y*s,a.z*s}; }
static inline float dot(Vec3 a, Vec3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
static inline float len(Vec3 a){ return std::sqrt(dot(a,a)); }
// normalization - returns (0, 0, 0) if vector is small
static inline Vec3 norm(Vec3 a){ float L=len(a); return L>1e-6f? mul(a,1.0f/L):Vec3{0,0,0}; }
// clamps vector mag to Lmax to maintain direction
static inline Vec3 clampLen(Vec3 v, float Lmax){ float L=len(v); return (L>Lmax)? mul(v,Lmax/L): v; }


// allocate seeds
void InitBoids(int count){
    gPos.resize(count);
    gVel.resize(count);
    gAcc.resize(count);
    for(int i=0;i<count;i++){
        gPos[i] = { (float)i*0.1f, P.altitude, 0.0f };
        gVel[i] = { 0.0f, 0.0f, 0.0f };
        gAcc[i] = { 0.0f, 0.0f, 0.0f };
    }
}

//resize buffers to a new count
//old contents are maintained by std::vector::resize
void ResizeBoids(int count){
    gPos.resize(count);
    gVel.resize(count);
    gAcc.resize(count);
}

//replace all params at once
void SetBoidParams(const BoidParams& p){ P = p; }


//zeroes all velocities when switching out from boids
void ResetVelocities(){
    for(auto &v : gVel) v = {0,0,0};
}

std::vector<Vec3>& GetBoidPositions(){ return gPos; }

const std::vector<float>& GetLastWeights(){
    return gLastWeights;
}

//Integration station: advance positions and velocities by dt(seconds)
void UpdateBoids(float dt, const std::vector<Vec3>& targets){
    EnsureEmotionsLoaded();
    float t = gTime;

    auto w = GetEmotionWeights(t);
    gLastWeights = w; 
    ApplyEmotionHard(w,P);


    int n = gPos.size();
    if ((int)targets.size() != n) return;
    //for large swarms, reduces pair checks to ~N*(N/stride)
    int stride = std::max(1, n/250);

    for(int i=0;i<n;i++){
        Vec3 pi = gPos[i];
        Vec3 vi = gVel[i];

        //collects for three classic boids params
        Vec3 fsep{0,0,0}, sumPos{0,0,0}, sumVel{0,0,0};
        int nAliCoh = 0;

        //seperation: pushes away if drones within r_sep
        for(int j=(i+1)%stride;j<n;j+=stride){
            Vec3 d = sub(gPos[j], pi);
            float d2 = dot(d,d);
            if(d2 < P.r_sep*P.r_sep && d2>1e-6f)
                fsep = add(fsep, mul(d, -1.0f/std::sqrt(d2)));
            //alignment/cohesion collects within r_nei
            if(d2 < P.r_nei*P.r_nei){
                sumPos = add(sumPos, gPos[j]);
                sumVel = add(sumVel, gVel[j]);
                nAliCoh++;
            }
        }

        //converts neighbor sums to aligned steering directions to reach target positions
        Vec3 fali{0,0,0}, fcoh{0,0,0};
        if(nAliCoh>0){
            Vec3 avgPos = mul(sumPos, 1.0f/nAliCoh);
            Vec3 avgVel = mul(sumVel, 1.0f/nAliCoh);
            //alignment: steer towards neighbooring drones average direction
            fali = sub(norm(avgVel), norm(vi));
            //cohesion: steer towards neighbooring drones center of mass
            fcoh = sub(avgPos, pi);
        }

        //goal seeking towards formation position
        Vec3 fgoal = sub(targets[i], pi);
        fgoal.y += (P.altitude - pi.y)*0.5f; //stablaizes altitude

        //jitters to break perfect symmetry
        Vec3 fjit = { (float)(rand()/double(RAND_MAX)-0.5)*P.jitter,
                      (float)(rand()/double(RAND_MAX)-0.5)*0.3f*P.jitter,
                      (float)(rand()/double(RAND_MAX)-0.5)*P.jitter };

        
        //Weighted sum of directions and jitters, all normalized
        Vec3 acc{0,0,0};
        acc = add(acc, mul(norm(fsep), P.k_sep));
        acc = add(acc, mul(norm(fali), P.k_ali));
        acc = add(acc, mul(norm(fcoh), P.k_coh));
        acc = add(acc, mul(norm(fgoal), P.k_goal));
        acc = add(acc, fjit);

        //caps on acceleration, speed and position
        acc = clampLen(acc, P.amax);
        vi  = clampLen(add(vi, mul(acc, dt)), P.vmax);
        pi  = add(pi, mul(vi, dt));
        
        //bound to a rectangular box
        clampToBox(pi);
        //applyBoxConstraint(pi, vi);

        gVel[i] = vi;
        gPos[i] = pi;
    }
}
