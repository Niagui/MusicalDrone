// simulate_boids.cpp  (no OpenGL)
// g++ -std=c++17 simulate_boids.cpp boids.cpp -o simulate_boids

#include "boids.h"
#include <iostream>
#include <vector>
#include <cstdio>
#include <cmath>

// --- copy or include these from basicVisuals.cpp ---

static std::vector<Vec3> SampleCircle(int n, float radius, float phase) {
    std::vector<Vec3> pts(n);
    for (int i = 0; i < n; ++i) {
        float a = (i / (float)n) * 2.0f * 3.14159265358979323846f + phase;
        pts[i] = { radius * std::cos(a), 0.0f, radius * std::sin(a) };
    }
    return pts;
}


int main() {
    const int   droneCount   = 6;
    const float totalTime    = GetAudioLength();
    const float simDt        = 1.0f / 60.0f;
    const float outputDt     = 0.2f;  // snapshot every 0.2 s

    std::cerr << totalTime << std::endl;


    InitBoids(droneCount);

    float t        = 0.0f;
    float nextDump = 0.0f;

    // CSV header (one line per drone per snapshot)
    std::printf("t,drone,x,y,z,"
                "r_sep,r_nei,k_sep,k_ali,k_coh,k_goal,"
                "vmax,amax,altitude,jitter\n");

    while (t <= totalTime + 1e-6f) {
        // Example: CIRCLE formation like your CIRCLE case
        float phase = 0.25f * t; // same as CIRCLE spin in basicVisuals
        std::vector<Vec3> slots = SampleCircle(droneCount, 3.0f, phase);
        for (auto& s : slots) s.y = 0.0f;   // altitude handled by boids params

        // advance emotions and boids
        SetSimTime(t);
        UpdateBoids(simDt, slots);

        // dump every 0.2 seconds
        if (t + 0.5f * simDt >= nextDump) {
            const BoidParams& P          = GetBoidParams();
            const std::vector<Vec3>& pos = GetBoidPositions();

            for (int i = 0; i < droneCount; ++i) {
                const Vec3& p = pos[i];
                std::printf(
                    "%.3f,%d,%.4f,%.4f,%.4f,"
                    "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                    "%.3f,%.3f,%.3f,%.3f\n",
                    nextDump, i, p.x, p.y, p.z,
                    P.r_sep, P.r_nei, P.k_sep, P.k_ali, P.k_coh, P.k_goal,
                    P.vmax, P.amax, P.altitude, P.jitter
                );
            }

            nextDump += outputDt;
        }

        t += simDt;
    }

    return 0;
}
