#include "boids.h"
#include <iostream>
#include <vector>
#include <cstdio>
#include <cmath>
#include "config.h" 

Config cfg;

static std::vector<Vec3> SampleCircle(int n, float radius, float phase) {
    std::vector<Vec3> pts(n);
    for (int i = 0; i < n; ++i) {
        float a = (i / (float)n) * 2.0f * 3.14159265358979323846f + phase;
        pts[i] = { radius * std::cos(a), 0.0f, radius * std::sin(a) };
    }
    return pts;
}

static std::vector<Vec3> SampleLine(int n)
{
    float init_dist = cfg.drone_config.init_dist;

    std::vector<Vec3> pts;
    pts.reserve(n);

    if (n <= 0) return pts;

    if (n == 1) {
        pts.push_back({0.0f, 0.0f, 0.0f});
        return pts;
    }

    float span = init_dist * (n - 1);
    float y0   = -0.5f * span;

    for (int i = 0; i < n; i++) {
        float y = y0 + i * init_dist;
        pts.push_back({0.0f, y, 0.0f});
    }
    return pts;
}

int main() {
    const int   droneCount   = cfg.drone_config.num_drones;
    const float totalTime    = GetAudioLength();
    const float simDt        = 1.0f / 60.0f;
    const float outputDt     = 0.2f;  // snapshot every 0.2 sec

    std::cerr << totalTime << std::endl;


    InitBoids(droneCount);

    float t        = 0.0f;
    float nextDump = 0.0f;

    // csv header
    // std::printf("t,drone,x,y,z,"
    //             "r_sep,r_nei,k_sep,k_ali,k_coh,k_goal,"
    //             "vmax,amax,altitude,jitter\n");

    //give 2 seconds to settle
    while (t <= totalTime + 5) {

        float phase = 0.25f * t; 
        std::vector<Vec3> slots = SampleLine(droneCount);
        for (auto& s : slots) s.z = 0.0f;  

        // updates
        SetSimTime(t);
        UpdateBoids(simDt, slots);

        // dump every 0.2 seconds
        if (t + 0.5f * simDt >= nextDump) {
            const BoidParams& P          = GetBoidParams();
            const std::vector<Vec3>& pos = GetBoidPositions();
            const std::vector<Vec3>& vel = GetBoidVelocities();
            const std::vector<Vec3>& acc = GetBoidAcclerations();

            for (int i = 0; i < droneCount; ++i) {
                const Vec3& p = pos[i];
                const Vec3& v = vel[i];
                const Vec3& a = acc[i];
                // std::printf(
                //     "%.3f,%d,%.4f,%.4f,%.4f,"
                //     "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                //     "%.3f,%.3f,%.3f,%.3f\n",
                //     nextDump, i, p.x, p.y, p.z,
                //     P.r_sep, P.r_nei, P.k_sep, P.k_ali, P.k_coh, P.k_goal,
                //     P.vmax, P.amax, P.altitude, P.jitter
                // );

                std::printf(
                    "%d,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    i, nextDump, p.x, p.y, p.z,
                    v.x, v.y, v.z, a.x, a.y, a.z  //include yaw as 0 and normalize time
                );
            }
            nextDump += outputDt;
        }
        
        t += simDt;
    }

    return 0;
}
