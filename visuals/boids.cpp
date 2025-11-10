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


static std::vector<Vec3> gPos, gVel, gAcc;
static inline float clampf(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
}

static BoidParams P = {
    /* r_sep */ 0.6f,  /* r_nei */ 2.5f,
    /* k_sep */ 1.0f,  /* k_ali */ 1.0f,  /* k_coh */ 0.7f,  /* k_goal */ 1.1f,
    /* vmax  */ 6.0f,  /* amax  */ 12.0f,
    /* altitude */ 1.7f, /* jitter */ 0.25f
};

// --- JSON / Emotion Labeleling --- //

using json = nlohmann::json;
static json EMO;                //holds loaded json array
static bool gEmoLoaded = false; //only load once
static float gTime = 0.0f;     

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

bool LoadEmotionFile(const std::string& path){
    std::ifstream f(path);
    if (!f) return false;
    json J; f >> J;
    if (!J.is_array()) return false;
    EMO = std::move(J);
    return true;
}


static void EnsureEmotionsLoaded() {
    if (gEmoLoaded) return;
    gEmoLoaded = LoadEmotionFile("clap_weights.json") ||
                 LoadEmotionFile("/mnt/data/clap_weights.json");
}



bool ReloadAndApplyEmotions(float t) {
    // try local path first, then /mnt/data as fallback
    bool ok = LoadEmotionFile("clap_weights.json");
    if (!ok) ok = LoadEmotionFile("/mnt/data/clap_weights.json");
    if (!ok) return false;

    auto w = GetEmotionWeights(t);
    if (w.empty()) return false;

    // mutate current params
    BoidParams& Pmut = const_cast<BoidParams&>(GetBoidParams());
    ApplyEmotion(w, Pmut);
    return true;
}

static BoidParams PFrom = P, PTo = P;
static float gTransLeft = 0.0f, gTransTotal = 0.0f;
static std::string gEmotion = "neutral";

static inline float lerp(float a, float b, float t){ return a + (b - a) * t; }
static inline BoidParams lerpParams(const BoidParams& A, const BoidParams& B, float t){
    BoidParams R = A;
    R.r_sep   = lerp(A.r_sep,   B.r_sep,   t);
    R.r_nei   = lerp(A.r_nei,   B.r_nei,   t);
    R.k_sep   = lerp(A.k_sep,   B.k_sep,   t);
    R.k_ali   = lerp(A.k_ali,   B.k_ali,   t);
    R.k_coh   = lerp(A.k_coh,   B.k_coh,   t);
    R.k_goal  = lerp(A.k_goal,  B.k_goal,  t);
    R.vmax    = lerp(A.vmax,    B.vmax,    t);
    R.amax    = lerp(A.amax,    B.amax,    t);
    R.altitude= lerp(A.altitude,B.altitude,t);
    R.jitter  = lerp(A.jitter,  B.jitter,  t);
    return R;
}


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

//Integration station: advance positions and velocities by dt(seconds)
void UpdateBoids(float dt, const std::vector<Vec3>& targets){
    EnsureEmotionsLoaded();
    gTime += dt;
    float t = gTime;

    auto w = GetEmotionWeights(t);
    ApplyEmotion(w, P);


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

        gVel[i] = vi;
        gPos[i] = pi;
    }
}
