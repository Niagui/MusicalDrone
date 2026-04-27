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
#include <osqp.h>
#include "config.h" 

//json
using json = nlohmann::json;
static json clap_weights;                //holds loaded json array
static bool clap_weights_loaded = false; //only load once

//loading
static std::vector<std::string> emotion_labels;
static bool emo_labels_loaded = false;
static std::vector<float> last_weights;
static bool gUsePhraseAttractor = true; //  llm generated phrase attractor toggle
static bool gPhrasePlanLoaded = false;
static bool gPhrasePlanAttempted = false;
static json phrase_plan_json;

//basic physics
static std::vector<Vec3> position, velocity, acceleration;  //position, velocity, acceleration


//bound box
static const float X_MIN = cfg.bound_config.x_min;
static const float X_MAX =  cfg.bound_config.x_max;
static const float Z_MIN = cfg.bound_config.z_min;  
static const float Z_MAX =  cfg.bound_config.z_max;
static const float Y_MIN = cfg.bound_config.y_min;
static const float Y_MAX = cfg.bound_config.y_max;


//parameter bounds
static const float R_SEP_MIN = 1.0f;    //collision prevention 
static const float R_SEP_MAX = 2.0f; 
static const float R_NEI_MIN = 1.2f; 
static const float R_NEI_MAX = 9.0f; 
static const float K_SEP_MIN = 1.0f; 
static const float K_SEP_MAX = 9.f; 

//clock
static float sim_time = 0.0f;
static float gAppliedSegStart = -1.0f;
static float gAppliedSegEnd   = -1.0f;
static float gAudioEndTime = -1.0f;
static bool frozen = false;
//this is the time to mark the timeline inside the simulation
void SetSimTime(float t){
    sim_time = t;
}

//resetting
std::vector<float> gResetTimes;
static int   gNextResetIndex = 0;
static float gLastBoidsTime  = 0.0f;
bool gSegmentsLoaded = false;


// math helper functions
//choke in range lo to hi
static inline float clampf(float x, float lo, float hi) {return std::max(lo, std::min(hi, x));} 
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline Vec3 add(Vec3 a, Vec3 b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; }
static inline Vec3 sub(Vec3 a, Vec3 b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
static inline Vec3 mul(Vec3 a, float s){ return {a.x*s,a.y*s,a.z*s}; }
static inline float dot(Vec3 a, Vec3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
static inline float len(Vec3 a){ return std::sqrt(dot(a,a)); }
static inline Vec3 pow(Vec3 a, float p){return {std::pow(a.x, p), std::pow(a.y, p), std::pow(a.z, p)};}
// normalization - returns (0, 0, 0) if vector is small
static inline Vec3 norm(Vec3 a){ float L=len(a); return L>1e-6f? mul(a,1.0f/L):Vec3{0,0,0}; }
// clamps vector mag to Lmax to maintain direction
static inline Vec3 clampLen(Vec3 v, float Lmax){ float L=len(v); return (L>Lmax)? mul(v,Lmax/L): v; }



struct EmotionSegment {
    float start = 0.0f;
    float end = 0.0f;
    std::vector<float> weights;
    bool valid = false;
};

struct PhrasePlanSegment {
    float start = 0.0f;
    float end = 0.0f;
    int beat_count = 8;
    float height_level = 0.5f;
    float depth_level = 0.5f;
    float speed_level = 0.5f;
    std::string motion_mode = "hold";
    std::string vertical_trend = "hold";
    std::string transition_style = "smooth";
    json beat_plan = json::array();
    bool valid = false;
};

struct ParamDelta 
{   //For easier batch update
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


// struct StyleParams {
//     float spin_rate;        // yaw rotation around goal
//     float bob_amp;          // vertical bobbing amplitude
//     float bob_freq;         // vertical bobbing frequency
//     float pause_prob;       // chance to briefly freeze / hesitate
//     float turn_sharpness;   // extra curvature in paths
// };



//--------------------------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------------------------//

// Loading helpers

bool LoadEmotionLabels(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;

    json J;
    f >> J;

    if (!J.is_array()) return false;

    emotion_labels.clear();
    for (auto& item : J) {
        if (item.is_string()) {
            emotion_labels.push_back(item.get<std::string>());
        }
    }

    std::cerr << "Loaded labels (" << emotion_labels.size() << "): ";
    for (auto& s : emotion_labels) std::cerr << s << " ";
    std::cerr << "\n";

    return true;
}

const std::vector<std::string>& GetEmotionLabels(){
    return emotion_labels;
}

static EmotionSegment GetEmotionSegment(float t)
{
    EmotionSegment seg;
    if(!clap_weights.is_array() || clap_weights.empty()) return seg;

       for (auto& it : clap_weights){
        float s = it.value("start", 0.0f);
        float e = it.value("end", 0.0f);
        if (t >= s && t < e) {
            seg.start = s;
            seg.end   = e;
            if (it.contains("weights")) seg.weights = it["weights"].get<std::vector<float>>();
            seg.valid = true;
            return seg;
        }
    }

    // hold last segment after audio ends
    auto& last = clap_weights.back();
    seg.start = last.value("start", 0.0f);
    seg.end   = last.value("end", seg.start + 1.0f);
    if(last.contains("weights")) seg.weights = last["weights"].get<std::vector<float>>();
    seg.valid = true;
    return seg;
}

std::vector<float> GetEmotionWeights(float t)
{
    EmotionSegment seg = GetEmotionSegment(t);
    return seg.valid ? seg.weights : std::vector<float>{};
}

bool LoadEmotionFile(const std::string& path){
    std::ifstream f(path);
    if (!f) return false;
    json J; f >> J;
    if (!J.is_array()) return false;
    clap_weights = std::move(J);
    return true;
}

static std::filesystem::path find_json_file(const std::string &filename) {
    namespace fs = std::filesystem;
    fs::path cwd = fs::current_path();

    std::vector<fs::path> search_roots;
    if (const char* env_json_dir = std::getenv("DRONE_JSON_DIR"); env_json_dir && *env_json_dir) {
        fs::path env_path(env_json_dir);
        search_roots.push_back(env_path);
        search_roots.push_back(env_path / "json");
    }

    search_roots.push_back(cwd / "json");
    search_roots.push_back(cwd.parent_path() / "json");

    for (const auto &root : search_roots) {
        fs::path p = root / filename;
        std::cerr << "[DEBUG] trying: " << p.string() << "\n";

        if (fs::exists(p)) {
            return p;
        }
    }

    return {};  // empty path = not found
}

bool LoadPhrasePlanFile(const std::string& path){
    std::ifstream f(path);
    if (!f) return false;
    json J; f >> J;
    if (!J.is_object() || !J.contains("phrases") || !J["phrases"].is_array()) return false;
    phrase_plan_json = std::move(J["phrases"]);
    return phrase_plan_json.is_array() && !phrase_plan_json.empty();
}

bool LoadResetTimes(const std::string& filename = "sections.json")
{
    auto segmentPath = find_json_file(filename).string();
    gResetTimes.clear();
    gNextResetIndex = 0;

    std::ifstream in(segmentPath);
    if (!in) {
        std::cerr << "Failed to open reset-times JSON: " << segmentPath << "\n";
        return false;
    }

    json j;
    in >> j;

    for (const auto& seg : j) {
        if (seg.is_array() && seg.size() >= 2) {
            float endTime = seg[1].get<float>();
            gResetTimes.push_back(endTime);
        }
    }

    std::cerr << "Loaded " << gResetTimes.size()
              << " reset times from " << filename << "\n";
    return true;
}

void EnsureEmotionsLoaded() 
{
    if (!clap_weights_loaded){
        auto clap_path = find_json_file("clap_weights.json");
        bool ok = LoadEmotionFile(clap_path.string());

        if (clap_path.empty()) {
            std::cerr << "ERROR: Could not find clap_weights.json\n";
            clap_weights_loaded = false;
            return;
        }

        if (!ok) {
            std::cerr << "ERROR: Could not load clap weights file\n"
                      << clap_path.string() << "\n";  
        } else {
            std::cerr << "Loaded JSON successfully. #Segments = " 
                    << clap_weights.size() << "\n";
        }
        clap_weights_loaded = ok;
    }

    if (!emo_labels_loaded) {
        auto anchor_path = find_json_file("anchor_labels.json");
        bool ok = LoadEmotionLabels(anchor_path.string());

        if (anchor_path.empty()) {
            std::cerr << "ERROR: Could not find anchor_labels.json\n";
            clap_weights_loaded = false;
            return;
        }

        if (!ok) {
            std::cerr << "ERROR: Could not load anchor_labels.json\n";
        } else {
            std::cerr << "Loaded JSON successfully." << "\n";
        }
        emo_labels_loaded = ok;
    }
}

void EnsureSegmentsLoaded()
{
    if (!gSegmentsLoaded)
    {
        bool ok = LoadResetTimes();

        if(!ok){
            std::cerr << "Could not load segments\n";
        }else{
            std::cerr << "Loaded segments successfully." << "\n";
            gSegmentsLoaded = true;
        }
    }
    return;
}

void EnsurePhrasePlanLoaded()
{
    if (!gUsePhraseAttractor || gPhrasePlanLoaded || gPhrasePlanAttempted) return;
    gPhrasePlanAttempted = true;

    auto phrase_path = find_json_file("phrase_plan.json");
    if (phrase_path.empty()) {
        std::cerr << "WARNING: Could not find phrase_plan.json, using fixed targets.\n";
        return;
    }

    bool ok = LoadPhrasePlanFile(phrase_path.string());
    if (!ok) {
        std::cerr << "WARNING: Could not load phrase_plan.json, using fixed targets.\n";
        return;
    }

    std::cerr << "Loaded phrase plan successfully. #Phrases = "
              << phrase_plan_json.size() << "\n";
    gPhrasePlanLoaded = true;
}

static PhrasePlanSegment GetPhrasePlanSegment(float t)
{
    PhrasePlanSegment seg;
    if (!phrase_plan_json.is_array() || phrase_plan_json.empty()) return seg;

    auto fill_segment = [&](const json& item) {
        seg.start = item.value("start", 0.0f);
        seg.end = item.value("end", seg.start + 1.0f);
        seg.beat_count = std::max(1, item.value("beat_count", 8));
        seg.height_level = clampf(item.value("height_level", 0.5f), 0.0f, 1.0f);
        seg.depth_level = clampf(item.value("depth_level", 0.5f), 0.0f, 1.0f);
        seg.speed_level = clampf(item.value("speed_level", 0.5f), 0.0f, 1.0f);
        seg.motion_mode = item.value("motion_mode", std::string("hold"));
        seg.vertical_trend = item.value("vertical_trend", std::string("hold"));
        seg.transition_style = item.value("transition_style", std::string("smooth"));
        if (item.contains("beat_plan") && item["beat_plan"].is_array()) {
            seg.beat_plan = item["beat_plan"];
        } else {
            seg.beat_plan = json::array();
        }
        seg.valid = true;
    };

    for (const auto& item : phrase_plan_json) {
        float start = item.value("start", 0.0f);
        float end = item.value("end", start + 1.0f);
        if (t >= start && t < end) {
            fill_segment(item);
            return seg;
        }
    }

    if (t < phrase_plan_json.front().value("start", 0.0f)) {
        fill_segment(phrase_plan_json.front());
        return seg;
    }

    fill_segment(phrase_plan_json.back());
    return seg;
}

static inline float smooth01(float x)
{
    x = clampf(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

static float ApplyPhraseEase(float u, const std::string& style)
{
    u = clampf(u, 0.0f, 1.0f);

    if (style == "drift") {
        return 0.5f - 0.5f * std::cos(u * 3.14159265358979323846f);
    }
    if (style == "surge") {
        float s = smooth01(u);
        return clampf(0.2f * u + 0.8f * s * s, 0.0f, 1.0f);
    }
    if (style == "snap") {
        if (u < 0.25f) return 0.0f;
        return smooth01((u - 0.25f) / 0.75f);
    }
    return smooth01(u);
}

static Vec3 PhraseActionDelta(const std::string& action, float xy_amp, float z_amp)
{
    if (action == "advance") return {0.0f,  xy_amp, 0.0f};
    if (action == "retreat") return {0.0f, -xy_amp, 0.0f};
    if (action == "sweep_left") return {-xy_amp, 0.0f, 0.0f};
    if (action == "sweep_right") return { xy_amp, 0.0f, 0.0f};
    if (action == "rise") return {0.0f, 0.0f, z_amp};
    if (action == "fall") return {0.0f, 0.0f, -z_amp};
    return {0.0f, 0.0f, 0.0f};
}

static Vec3 EvaluatePhraseAttractor(float t)
{
    PhrasePlanSegment phrase = GetPhrasePlanSegment(t);
    if (!phrase.valid) return {0.0f, 0.0f, 0.0f};

    float duration = std::max(1e-3f, phrase.end - phrase.start);
    float u = clampf((t - phrase.start) / duration, 0.0f, 1.0f);
    float ease = ApplyPhraseEase(u, phrase.transition_style);

    float xy_amp = lerpf(0.12f, 0.55f, phrase.speed_level);
    float z_amp = lerpf(0.05f, 0.20f, phrase.speed_level);

    Vec3 base = {0.0f, lerpf(Y_MIN + 0.15f, Y_MAX - 0.15f, phrase.depth_level), 0.0f};
    Vec3 motion = mul(PhraseActionDelta(phrase.motion_mode, xy_amp, z_amp), ease);

    if (phrase.vertical_trend == "rise") motion.z += z_amp * ease;
    else if (phrase.vertical_trend == "fall") motion.z -= z_amp * ease;

    float settle_gain = 1.0f;
    for (const auto& event : phrase.beat_plan) {
        if (!event.is_object()) continue;
        int beat = std::max(1, event.value("beat", 1));
        std::string action = event.value("action", std::string("hold"));
        float event_u = clampf((float)(beat - 1) / (float)phrase.beat_count, 0.0f, 1.0f);
        float beat_progress = clampf((u - event_u) * (float)phrase.beat_count, 0.0f, 1.0f);
        float beat_ease = ApplyPhraseEase(beat_progress, phrase.transition_style);
        if (beat_ease <= 0.0f) continue;

        if (action == "settle") {
            settle_gain = std::min(settle_gain, 1.0f - 0.65f * beat_ease);
            continue;
        }

        motion = add(motion, mul(PhraseActionDelta(action, xy_amp * 0.65f, z_amp), beat_ease));
    }

    motion.x *= settle_gain;
    motion.y *= settle_gain;
    return add(base, motion);
}



float GetAudioLength(){
    EnsureEmotionsLoaded();
    if(!clap_weights.is_array() || clap_weights.empty()) {
        std::cerr << "[DEBUG] clap_weights is empty or not an array\n";
        return 0.0f;
    }

    const auto& last = clap_weights.back();
    return last.value("end", 0.0f);
}


//--------------------------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------------------------//


//neutral parameters. Start with this scheme every time we change the emotion
static const BoidParams Neutral = 
{        
    /* r_sep */ 0.5f,  /* r_nei */ 2.0f,
    /* k_sep */ 1.0f,  /* k_ali */ 1.0f,  /* k_coh */ 1.0f,  /* k_goal */ 1.1f,
    /* vmax  */ 3.0f,  /* amax  */ 10.0f,
    /* altitude */ 1.7f, /* jitter */ 0.25f
};
static BoidParams P = Neutral;


static const ParamDelta ANCHORS[7] = 
{   
    //keep sum of each column close to 0 for best expresiveness
    //r_sep   r_nei   k_sep   k_ali   k_coh   k_goal   vmax    amax   altitude  jitter
    { +0.60f, -2.00f, +2.00f, -1.00f, -1.50f, +1.40f, +3.0f,  +8.0f,  +1.50f,  +0.40f }, // happy (fast, cohesive, lively)
    { -0.70f, +3.00f, +3.00f, +1.00f, -0.50f, -1.00f, -2.5f,  -8.0f,  -0.80f,  -0.15f }, // sad (slow, heavy, low drive)
    { +0.60f, -1.00f, -2.50f, -0.90f, -0.60f, -1.95f, -3.5f,  -8.0f,  -1.00f,  -0.20f }, // sleepy (very slow, minimal jitter)
    { -0.50f, +2.00f, +2.00f, +1.40f, +0.80f, +1.50f, +4.0f,  +8.0f,  +1.20f,  -0.25f }, // brave (fast, decisive, strong goal)
    { +0.55f, -2.20f, -4.00f, -0.20f, -1.35f, -0.45f, -3.0f,  +4.0f,  +0.20f,  +0.15f }, // grumpy (aggressive spacing, low cohesion)
    { +0.80f, -2.80f, +5.00f, +0.30f, -0.55f, +0.80f, +3.5f,  +8.0f,  -0.60f,  +0.50f }, // scared (panic: fast + jitter + separation)
    { -0.55f, +1.20f, -2.80f, +0.80f, +1.00f, -0.60f, -2.5f,  -4.0f,  -0.20f,  -0.05f }, // shy (small/slow, stays together, low goal)
};



static inline Vec3 BoxCenter()
{
    return Vec3{ 0.5f*(X_MIN+X_MAX), 0.5f*(Y_MIN+Y_MAX), 0.5f*(Z_MIN+Z_MAX) };
}
static inline float BoxHalfX() { return 0.5f*(X_MAX-X_MIN); }
static inline float BoxHalfY() { return 0.5f*(Y_MAX-Y_MIN); } 
static constexpr float OLD_HALF_X = 5.0f;
static constexpr float OLD_HALF_Y = 5.0f;
static inline float ScaleXY() { return std::min(BoxHalfX()/OLD_HALF_X, BoxHalfY()/OLD_HALF_Y); }
static inline float DynamicsScaleFromBox(float sxy) {
    float strength = clampf(cfg.drone_config.box_motion_scale_strength, 0.0f, 1.0f);
    float dyn = (1.0f - strength) + strength * sxy;
    return clampf(dyn, 0.50f, 1.0f);
}
static inline float FitTargetsToBoxScale(const std::vector<Vec3>& targets, Vec3 center, float margin_xy) {
    if (targets.empty()) return 1.0f;

    float max_dx = 1e-4f;
    float max_dy = 1e-4f;
    for (const Vec3& t : targets) {
        Vec3 d = sub(t, center);
        max_dx = std::max(max_dx, std::abs(d.x));
        max_dy = std::max(max_dy, std::abs(d.y));
    }

    float avail_half_x = std::max(0.05f, BoxHalfX() - margin_xy);
    float avail_half_y = std::max(0.05f, BoxHalfY() - margin_xy);
    float sx = avail_half_x / max_dx;
    float sy = avail_half_y / max_dy;
    float fit = std::min(1.0f, std::min(sx, sy));
    return clampf(fit, 0.20f, 1.0f);
}




Boundaries GetBoxBounds() 
{
    return Boundaries{ X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX };
}

// Box scaling (volume-based)
static constexpr float DEFAULT_BOX_VOLUME = 10.0f * 10.0f * 1.0f;


static inline void clampToBox(Vec3& p) 
{
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

//   Compute a single target boid parameters with precomputed weights
BoidParams ApplyEmotionHard(const std::vector<float>& w, float scale = 1)
{
    BoidParams target = Neutral;

    if (w.empty()) 
        return target;  //early exist if no weights

    float wsum = 0.0f;

    for (float x : w)
    {
        wsum += std::abs(x);
    } 
        
    if (wsum < 1e-6f) 
        return target;

    float inv = 1.0f / wsum;

    for (int i = 0; i < 7; ++i) {
        float wi = w[i] * inv;      //ensure obvious change
        const ParamDelta& A = ANCHORS[i];

        target.r_nei    += wi * A.r_nei;
        target.k_sep    += wi * A.k_sep;
        target.r_sep    += wi * A.r_sep;
        target.k_ali    += wi * A.k_ali;
        target.k_coh    += wi * A.k_coh;
        target.k_goal   += wi * A.k_goal;
        target.vmax     += wi * A.vmax;
        target.amax     += wi * A.amax;
        target.altitude += wi * A.altitude;
        target.jitter   += wi * A.jitter;
    }

    // Decoupled scaling policy:
    // - social radii shrink mildly with box
    // - dynamics keep a higher floor so motion stays lively
    float social_scale = clampf(0.90f + 0.10f * scale, 0.90f, 1.0f);
    float dyn_scale = scale;
    float alt_scale = clampf(0.85f + 0.15f * scale, 0.85f, 1.0f);

    target.r_sep    *= social_scale;
    target.r_nei    *= social_scale;
    target.vmax     *= dyn_scale;
    target.amax     *= dyn_scale;
    target.altitude *= alt_scale;

    // target.k_ali *= scale;
    // target.k_coh*= scale;
    // target.k_sep*= scale;
    // target.k_goal*= scale;

    target.r_sep    = clampf(target.r_sep,    R_SEP_MIN * social_scale,  R_SEP_MAX * social_scale);
    target.k_sep    = clampf(target.k_sep,    K_SEP_MIN,  K_SEP_MAX);
    target.k_ali    = clampf(target.k_ali,    0.00f,  3.0f);
    target.k_coh    = clampf(target.k_coh,    0.00f,  3.0f);
    target.k_goal   = clampf(target.k_goal,   0.20f,  3.0f);
    target.vmax     = clampf(target.vmax,     0.50f * dyn_scale, 10.0f * dyn_scale);
    target.amax     = clampf(target.amax,     1.00f * dyn_scale, 20.0f * dyn_scale);
    target.altitude = clampf(target.altitude, 0.50f * alt_scale, 5.0f * alt_scale);
    target.jitter   = clampf(target.jitter,   0.00f,  1.0f);
    target.vmax     = std::min(target.vmax, std::max(cfg.drone_config.vmax_cap, 0.2f));
    target.amax     = std::min(target.amax, std::max(cfg.drone_config.amax_cap, 0.5f));

    // Keep boid separation behavior consistent with CBF safety.
    target.r_sep = std::max(target.r_sep, cfg.cbf_config.safety_radius * 1.10f);
    target.r_nei = std::max(target.r_nei, target.r_sep + 0.05f);

    return target;
}



// current params for HUD/debug
const BoidParams& GetBoidParams(){
    return P;
}

static BoidParams LerpParams(const BoidParams& a, const BoidParams& b, float t)
{
    BoidParams o = a;
    auto lp = [&](float& x, float xa, float xb){ x = xa + (xb - xa) * t; };

    lp(o.r_sep,    a.r_sep,    b.r_sep);
    lp(o.r_nei,    a.r_nei,    b.r_nei);
    lp(o.k_sep,    a.k_sep,    b.k_sep);
    lp(o.k_ali,    a.k_ali,    b.k_ali);
    lp(o.k_coh,    a.k_coh,    b.k_coh);
    lp(o.k_goal,   a.k_goal,   b.k_goal);
    lp(o.vmax,     a.vmax,     b.vmax);
    lp(o.amax,     a.amax,     b.amax);
    lp(o.altitude, a.altitude, b.altitude);
    lp(o.jitter,   a.jitter,   b.jitter);
    return o;
}



/***********
 * cbf
 **************/

static inline float cbf_constraint(Vec3 pos, Vec3 vel, Vec3 center, float r=0.5, float alpha=1.0)
{
    Vec3 d = sub(pos, center);
    float h = dot(d, d) - r * r;
    float hdot = 2.0f * dot(d, vel); 
    return hdot + alpha*h;
}

static CBFManager cbf_manager;
static bool cbf_enabled = true;

void EnableCBF(bool enable) {
    cbf_enabled = enable;
}

void SetCBFConfig(const CBFConfig& config) {
    cbf_manager.SetConfig(config);
}



/****************
 * boids
 **************/

// allocate seeds
void InitBoids(int count){
    position.resize(count);
    velocity.resize(count);
    acceleration.resize(count);

    cbf_manager.Init(count);

    const float droneR = 0.092f;
    const float cbfSafeDist = cfg.cbf_config.safety_radius * 1.05f;
    const float minDist = std::max(2.5f * droneR, cbfSafeDist);
    const float z0 = clampf(P.altitude, Z_MIN, Z_MAX);  // Z is vertical

    float totalLen = (count - 1) * minDist;
    float startX = -0.5f * totalLen;

    for(int i=0;i<count;i++){
        position[i] = { startX + i * minDist, 0.0f, z0 }; // line on horizontal plane, fixed altitude in Z
        velocity[i] = { 0.0f, 0.0f, 0.0f };
        acceleration[i] = { 0.0f, 0.0f, 0.0f };
    }
}

//resize buffers to a new count
//old contents are maintained by std::vector::resize
void ResizeBoids(int count){
    position.resize(count);
    velocity.resize(count);
    acceleration.resize(count);
    cbf_manager.Resize(count);
}

//zeroes all velocities when switching out from boids
void ResetVelocities(){
    for(auto &v : velocity) v = {0,0,0};
}

std::vector<Vec3>& GetBoidPositions(){ return position; }
std::vector<Vec3>& GetBoidVelocities(){ return velocity; }
std::vector<Vec3>& GetBoidAcclerations(){ return acceleration; }

const std::vector<float>& GetLastWeights(){
    return last_weights;
}

//WARNING: this thing updates per frame not per segments
void UpdateBoids(float dt, const std::vector<Vec3>& targets){
    EnsureEmotionsLoaded();
    EnsureSegmentsLoaded();
    EnsurePhrasePlanLoaded();
    float t = sim_time;
    const float sXY = ScaleXY();
    const float dynScale = DynamicsScaleFromBox(sXY);

    // set audio end time once
    if (gAudioEndTime < 0.0f) {
        gAudioEndTime = clap_weights.back().value("end", 0.0f);
    }
    bool songEnded = (gAudioEndTime > 0.0f && t >= gAudioEndTime);

    if (songEnded) {
        if (!frozen) {
            // one-time freeze action
            for (size_t i = 0; i < velocity.size(); ++i) {
                velocity[i] = {0,0,0};
                acceleration[i] = {0,0,0};
                // optional: lock them to current altitude target
                position[i].z = clampf(position[i].z, Z_MIN, Z_MAX);
            }

            // optional: set params to something stable for HUD
            P = Neutral;
            P.jitter = 0.0f;

            frozen = true;
        }
        return; // stop updating positions entirely
    }


    if (gNextResetIndex < (int)gResetTimes.size()) {
        float resetT = gResetTimes[gNextResetIndex];

        // trigger when current time crosses the reset timestamp
        if (gLastBoidsTime < resetT && t >= resetT) {
            // std::cerr << "Resetting BoidParams at t=" << resetT << "\n";
            
            // // Reset parameters to neutral
            // std::cerr << "Before reset vmax=" << P.vmax << "\n";
            P = Neutral;
            // std::cerr << "After reset vmax=" << P.vmax << "\n";

            //Reset velocities so old emotion doesn't persist
            ResetVelocities();
            gNextResetIndex++;
        }
    }
    gLastBoidsTime = t;

    //update emotion once a every some time
    auto seg = GetEmotionSegment(t);
    if (seg.valid && (seg.start != gAppliedSegStart || seg.end != gAppliedSegEnd)) {
        gAppliedSegStart = seg.start;
        gAppliedSegEnd   = seg.end;

        auto weights = GetEmotionWeights(t);
        last_weights = weights;
        
        
        P = ApplyEmotionHard(weights, dynScale);
    }

    int n = position.size();
    if ((int)targets.size() != n) return;

    std::vector<Vec3> goalTargets = targets;
    Vec3 phraseAttractor = {0.0f, 0.0f, P.altitude};
    if (gUsePhraseAttractor && gPhrasePlanLoaded) {
        phraseAttractor = EvaluatePhraseAttractor(t);
        for (Vec3& target : goalTargets) {
            target.x += phraseAttractor.x;
            target.y += phraseAttractor.y;
        }
    }

    const Vec3 c = BoxCenter();
    const float margin_xy = std::max(0.10f, cfg.cbf_config.safety_radius * 0.5f);
    const float targetXYScale = FitTargetsToBoxScale(goalTargets, c, margin_xy);
    auto ScaleTargetXY = [&](Vec3 t){
        Vec3 d = sub(t, c);
        d.x *= targetXYScale;
        d.y *= targetXYScale;
        d.z = 0.0f; // keep goal horizontal; altitude handled separately in Z
        return add(c, d);
    };
    
    float desired_spacing = std::max(cfg.cbf_config.safety_radius * 1.10f, P.r_sep * 0.60f);
    float avail_half_x = std::max(0.10f, BoxHalfX() - margin_xy);
    float avail_half_y = std::max(0.10f, BoxHalfY() - margin_xy);
    float avail_area = (2.0f * avail_half_x) * (2.0f * avail_half_y);
    float needed_area = (float)n * 0.866f * desired_spacing * desired_spacing;
    float crowd_ratio = needed_area / std::max(avail_area, 1e-3f);
    float goal_crowd_scale = clampf(1.0f / std::sqrt(std::max(1.0f, crowd_ratio)), 0.35f, 1.0f);
    float sep_crowd_boost = clampf(1.0f + 0.35f * std::max(0.0f, crowd_ratio - 1.0f), 1.0f, 1.8f);
    const float motion_gain = 1.0f;
    const float vmax_eff = P.vmax * motion_gain;
    const float amax_eff = P.amax * motion_gain;
    const float alt_lo = Z_MIN + 0.08f;
    const float alt_hi = Z_MAX - 0.08f;

    //for large swarms, reduces pair checks to ~N*(N/stride)
    int stride = std::max(1, n/250);

    for(int i=0;i<n;i++){
        Vec3 pi = position[i];
        Vec3 vi = velocity[i];

        //collects for three classic boids params
        Vec3 fsep{0,0,0}, sumPos{0,0,0}, sumVel{0,0,0};
        int nAliCoh = 0;

        //seperation: pushes away if drones within r_sep
        for(int j=(i+1)%stride;j<n;j+=stride){
            Vec3 d = sub(position[j], pi);
            float d2 = dot(d,d);
            if(d2 < P.r_sep*P.r_sep && d2>1e-6f)
                fsep = add(fsep, mul(d, -1.0f/std::sqrt(d2)));
            //alignment/cohesion collects within r_nei
            if(d2 < P.r_nei*P.r_nei){
                sumPos = add(sumPos, position[j]);
                sumVel = add(sumVel, velocity[j]);
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
        Vec3 ti = ScaleTargetXY(goalTargets[i]);

        Vec3 fgoal = sub(ti, pi);
        float bob = std::sin(sim_time * (0.8f + 1.2f * P.jitter) + i * 0.37f) * (0.4f * P.jitter);

        float desiredAlt = P.altitude * (1.0f + 0.35f * P.jitter);
        if (gUsePhraseAttractor && gPhrasePlanLoaded) {
            desiredAlt = lerpf(desiredAlt, phraseAttractor.z, 0.7f);
        }
        float altTarget = clampf(desiredAlt + bob, alt_lo, alt_hi);
        fgoal.z += (altTarget - pi.z);   // Z is vertical

        //jitters to break perfect symmetry
        Vec3 fjit = {   (float)(rand()/double(RAND_MAX)-0.5)*P.jitter,
                        (float)(rand()/double(RAND_MAX)-0.5)*P.jitter,
                        (float)(rand()/double(RAND_MAX)-0.5)*0.3f*P.jitter }; // smaller vertical jitter on Z

        
        //Weighted sum of directions and jitters, all normalized
        Vec3 acc{0,0,0};
        acc = add(acc, mul(norm(fsep), P.k_sep * sep_crowd_boost));
        acc = add(acc, mul(norm(fali), P.k_ali));
        acc = add(acc, mul(norm(fcoh), P.k_coh));
        acc = add(acc, mul(norm(fgoal), P.k_goal * goal_crowd_scale));
        acc = add(acc, fjit);

        //caps on acceleration, speed and 
        acc = clampLen(acc, amax_eff);
        acceleration[i] = acc;

        Vec3 v_nom  = clampLen(add(vi, mul(acc, dt)), vmax_eff);
        
        if (cbf_enabled) {
            vi = cbf_manager.Solve(i, v_nom, position, velocity, vmax_eff, amax_eff, dt);
        } else {
            vi = v_nom;
        }

        pi  = add(pi, mul(vi, dt));
        
        //bound to a rectangular box
        if(!cbf_enabled){
            clampToBox(pi);
        }

        velocity[i] = vi;
        position[i] = pi;
    }
}
