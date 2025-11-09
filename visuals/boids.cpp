// boids.cpp - boids implementation for Drone Swarm Visualizer(basicVisuals.cpp)

//////////////////////////////////////////////////////////
// --- Boids (Notes) ---
// 1. Numbers/params are in meters and seconds, but are not real units, just make sure vmax and amax make sense for dt size
// 2. To prevent slowing down when increased number of drones, it only checks some nieghbors instead of every pair
// 3. Drones are pushed towards set alt to keep from too much drifting
// 4. Limitations on acceleration is to keep visualizer from blowing up
/////////////////////////////////////////////////////////

#include "boids.h"
#include <cmath>
#include <algorithm>
#include <cstdlib>

static std::vector<Vec3> gPos, gVel, gAcc;

static BoidParams P = {
    /* r_sep */ 0.6f,  /* r_nei */ 2.5f,
    /* k_sep */ 1.0f,  /* k_ali */ 1.0f,  /* k_coh */ 0.7f,  /* k_goal */ 1.1f,
    /* vmax  */ 6.0f,  /* amax  */ 12.0f,
    /* altitude */ 1.7f, /* jitter */ 0.25f
};

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
    int n = gPos.size();
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
