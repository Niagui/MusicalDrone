// basicVisuals.cpp - Drone Swarm Visualizer

//////////////////////////////////////////////////////////
// --- Drone Swarm Visualizer ---
// - Option 1: Basic Visuals Motion
// - Option 2: Boids - motion controlled by boids.cpp
/////////////////////////////////////////////////////////
// -- Build Notes --
// - Uses <GLUT/glut.h>
// - Requires boids.h/boids.cpp
/////////////////////////////////////////////////////////
// -- Keyboard Controls --
// Change drone count: +/-
// Built-in Formations: 1-cirlce, 2-line, 3-wave, 4-heart
// Pause: space bar
// Spin toggle: s
// Speed increase: .
// Speed decrease: ,
// Camera: mouse drag
// Boids toggle: b
//////////////////////////////////////////////////////////

#include "boids.h"


#if defined(__APPLE__)
  #include <GLUT/glut.h>   //macOS
#else
  #include <GL/glut.h>     //Linux/Windows
#endif

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include <thread>
#include <chrono>
//audio playback
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "config.h" 
#include "json.hpp"

using json = nlohmann::json;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Config cfg;

// --- GLOBAL VARIABLES ---
const float DRONE_RADIUS = 0.092; 

int   gWinW = 1200, gWinH = 800; //window size
int   gDroneCount = cfg.drone_config.num_drones;        // +/- to change
float gTime = 0.0f;             // seconds (advances when playing)
static float gSimTime = 0.0f;               // simulation time in seconds
static const float kFixedDt = 1.0f / 60.0f; // 60 FPS sim step
bool  gPlaying = true;          // Space toggles
bool  gSpin = true;             // S toggles
bool  gUseBoids = true;        // b toggles, true by default
static std::string gHudMsg;
static char gHudBuf[128];
static float gHudUntil = 0.0; 
static const float kLineToCircleTime = 1.2f;

// Sphere cam params
float gCamTheta = 0.7f;   // azimuth
float gCamPhi   = 0.9f;   // elevation
float gCamR     = 12.0f;  // radius
int   gLastX=0, gLastY=0; bool gDragging=false;

// Formation 
enum Form { CIRCLE=1, LINE=2, WAVE, HEART };
Form  gForm = LINE;

float gAltitude = 1.6f;   // Y height of all formations
float gSpeed    = 1.0f;   // Global animation speed

// Per-drone state: current positions and targets for formations
std::vector<Vec3> gPos; //current drone position
std::vector<Vec3> gSlots; //target position

static inline float clampf(float x, float lo, float hi){ return x<lo?lo:(x>hi?hi:x); }
static inline float Lerp(float a, float b, float t) { return a + (b-a)*t; }
static inline float smooth01(float x) {
    x = clampf(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}
static inline Vec3 LerpVec3(const Vec3& a, const Vec3& b, float t) {
    return {
        Lerp(a.x, b.x, t),
        Lerp(a.y, b.y, t),
        Lerp(a.z, b.z, t)
    };
}

//music
ma_engine gEngine;
ma_sound  gMusic;
bool gAudioInit = false;
static bool gAudioLoaded = false;      // Track if sound file is loaded
static bool gAudioSyncMode = false;    // When true, sync visuals to audio
static float gAudioDuration = 0.0f;    // Total audio length in seconds
static const char* kDefaultAudioFile = "testSong.mp3";
static std::string gAudioInputPath = kDefaultAudioFile;

struct WaypointSample {
    float t = 0.0f;
    Vec3 p{0.0f, 0.0f, 0.0f};
};

static bool gUseWaypointPlayback = true;
static bool gTrajectoryLoaded = false;
static std::string gTrajectoryInputPath = "trajectories.csv";
static std::filesystem::path gResolvedTrajectoryPath;
static std::vector<std::vector<WaypointSample>> gWaypointTracks;
static float gWaypointDuration = 0.0f;

struct EmotionLogSegment {
    float start = 0.0f;
    float end = 0.0f;
    std::vector<float> weights;
};

static std::vector<std::string> gEmotionLogLabels;
static std::vector<EmotionLogSegment> gEmotionLogSegments;
static bool gEmotionLogLoaded = false;
static int gLastEmotionLogIndex = -1;


//collision prints
static int   gFrameIndex = 0;
static float gLastCollisionPrintTime = -1e9f;
static std::set<std::pair<int, int>> gActiveCollisionPairs;


static std::filesystem::path ResolveAudioPath(const std::string& filename)
{
    namespace fs = std::filesystem;

    fs::path input(filename);
    std::vector<fs::path> candidates;

    if (input.is_absolute()) {
        candidates.push_back(input);
    } else {
        fs::path cwd = fs::current_path();
        candidates.push_back(cwd / input);
        candidates.push_back(cwd / "audio" / input);
        candidates.push_back(cwd.parent_path() / "audio" / input);
    }

    std::error_code ec;
    for (const fs::path& candidate : candidates) {
        if (fs::exists(candidate, ec)) {
            return candidate.lexically_normal();
        }
        ec.clear();
    }

    if (input.is_absolute()) {
        return input.lexically_normal();
    }

    return (fs::current_path() / input).lexically_normal();
}

static std::filesystem::path ResolveTrajectoryPath(const std::string& filename)
{
    namespace fs = std::filesystem;

    fs::path input(filename);
    std::vector<fs::path> candidates;

    if (input.is_absolute()) {
        candidates.push_back(input);
    } else {
        fs::path cwd = fs::current_path();
        candidates.push_back(cwd / input);
        candidates.push_back(cwd.parent_path() / input);
    }

    std::error_code ec;
    for (const fs::path& candidate : candidates) {
        if (fs::exists(candidate, ec)) {
            return candidate.lexically_normal();
        }
        ec.clear();
    }

    if (input.is_absolute()) {
        return input.lexically_normal();
    }

    return (fs::current_path() / input).lexically_normal();
}

static std::filesystem::path ResolveJsonPath(const std::string& filename)
{
    namespace fs = std::filesystem;

    fs::path cwd = fs::current_path();
    std::vector<fs::path> candidates;

    if (!gResolvedTrajectoryPath.empty()) {
        candidates.push_back(gResolvedTrajectoryPath.parent_path() / "json" / filename);
    }

    if (const char* env_json_dir = std::getenv("DRONE_JSON_DIR"); env_json_dir && *env_json_dir) {
        fs::path env_path(env_json_dir);
        candidates.push_back(env_path / filename);
        if (env_path.filename() != "json") {
            candidates.push_back(env_path / "json" / filename);
        }
    }

    candidates.push_back(cwd / "json" / filename);
    candidates.push_back(cwd.parent_path() / "json" / filename);
    candidates.push_back(cwd / filename);

    std::error_code ec;
    for (const fs::path& candidate : candidates) {
        if (fs::exists(candidate, ec)) {
            return candidate.lexically_normal();
        }
        ec.clear();
    }

    return {};
}

static void PrintUsage(const char* exeName)
{
    std::cout << "Usage: " << exeName << " [song-path] [--trajectory path/to/trajectory.csv]\n";
    std::cout << "   or: " << exeName << " --song <song-path>\n";
    std::cout << "   or: " << exeName << " --live-boids\n";
    std::cout << "Example: " << exeName << " ../audio/testSong.mp3\n";
    std::cout << "Example: " << exeName << " --song sad.mp3 --trajectory ../data/sad.mp3/trajectory.csv\n";
}

static void ParseCommandLine(int* argc, char** argv)
{
    std::vector<char*> filteredArgs;
    filteredArgs.reserve(*argc + 1);
    filteredArgs.push_back(argv[0]);

    bool audioPathSet = false;
    const std::string songPrefix = "--song=";
    const std::string trajectoryPrefix = "--trajectory=";
    const std::string waypointsPrefix = "--waypoints=";

    for (int i = 1; i < *argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        }

        if (arg == "--song" || arg == "-s") {
            if (i + 1 >= *argc) {
                std::cerr << "Missing value for " << arg << "\n";
                PrintUsage(argv[0]);
                std::exit(1);
            }

            gAudioInputPath = argv[++i];
            audioPathSet = true;
            continue;
        }

        if (arg.rfind(songPrefix, 0) == 0) {
            gAudioInputPath = arg.substr(songPrefix.size());
            audioPathSet = true;
            continue;
        }

        if (arg == "--trajectory" || arg == "--waypoints" || arg == "-t") {
            if (i + 1 >= *argc) {
                std::cerr << "Missing value for " << arg << "\n";
                PrintUsage(argv[0]);
                std::exit(1);
            }

            gTrajectoryInputPath = argv[++i];
            gUseWaypointPlayback = true;
            continue;
        }

        if (arg.rfind(trajectoryPrefix, 0) == 0) {
            gTrajectoryInputPath = arg.substr(trajectoryPrefix.size());
            gUseWaypointPlayback = true;
            continue;
        }

        if (arg.rfind(waypointsPrefix, 0) == 0) {
            gTrajectoryInputPath = arg.substr(waypointsPrefix.size());
            gUseWaypointPlayback = true;
            continue;
        }

        if (arg == "--live-boids") {
            gUseWaypointPlayback = false;
            continue;
        }

        if (!audioPathSet && !arg.empty() && arg[0] != '-') {
            gAudioInputPath = arg;
            audioPathSet = true;
            continue;
        }

        filteredArgs.push_back(argv[i]);
    }

    filteredArgs.push_back(nullptr);
    for (int i = 0; i < static_cast<int>(filteredArgs.size()); ++i) {
        argv[i] = filteredArgs[i];
    }
    *argc = static_cast<int>(filteredArgs.size()) - 1;
}

static std::vector<std::string> SplitCsvLine(const std::string& line)
{
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        cells.push_back(cell);
    }
    return cells;
}

static bool ParseWaypointRow(const std::string& line, int& droneId, WaypointSample& sample)
{
    std::vector<std::string> cells = SplitCsvLine(line);
    if (cells.size() < 5) return false;

    try {
        droneId = std::stoi(cells[0]);
        sample.t = std::stof(cells[1]);
        sample.p = {std::stof(cells[2]), std::stof(cells[3]), std::stof(cells[4])};
    } catch (...) {
        return false;
    }

    return droneId >= 0;
}

static bool LoadWaypointTrajectory(const std::string& filename)
{
    namespace fs = std::filesystem;

    gResolvedTrajectoryPath = ResolveTrajectoryPath(filename);
    std::ifstream f(gResolvedTrajectoryPath);
    if (!f) {
        std::cerr << "WARNING: Could not open trajectory file "
                  << gResolvedTrajectoryPath.string()
                  << ". Falling back to live boids.\n";
        gTrajectoryLoaded = false;
        return false;
    }

    std::vector<std::vector<WaypointSample>> tracks;
    std::string line;
    int rows = 0;
    int maxDroneId = -1;
    while (std::getline(f, line)) {
        if (line.empty()) continue;

        int droneId = -1;
        WaypointSample sample;
        if (!ParseWaypointRow(line, droneId, sample)) {
            continue;
        }

        if (droneId >= (int)tracks.size()) {
            tracks.resize(droneId + 1);
        }
        tracks[droneId].push_back(sample);
        maxDroneId = std::max(maxDroneId, droneId);
        rows++;
    }

    if (rows == 0 || maxDroneId < 0) {
        std::cerr << "WARNING: No waypoint rows found in "
                  << gResolvedTrajectoryPath.string()
                  << ". Falling back to live boids.\n";
        gTrajectoryLoaded = false;
        return false;
    }

    gWaypointDuration = 0.0f;
    for (std::vector<WaypointSample>& track : tracks) {
        std::sort(track.begin(), track.end(), [](const WaypointSample& a, const WaypointSample& b) {
            return a.t < b.t;
        });
        if (!track.empty()) {
            gWaypointDuration = std::max(gWaypointDuration, track.back().t);
        }
    }

    gWaypointTracks = std::move(tracks);
    gDroneCount = (int)gWaypointTracks.size();
    gTrajectoryLoaded = true;

    std::cout << "Loaded waypoint trajectory: " << gResolvedTrajectoryPath.string() << "\n";
    std::cout << "  drones: " << gDroneCount << "  rows: " << rows
              << "  duration: " << gWaypointDuration << "s\n";
    return true;
}

static Vec3 SampleWaypointTrack(const std::vector<WaypointSample>& track, float t)
{
    if (track.empty()) return {0.0f, 0.0f, gAltitude};
    if (t <= track.front().t) return track.front().p;
    if (t >= track.back().t) return track.back().p;

    auto upper = std::lower_bound(
        track.begin(),
        track.end(),
        t,
        [](const WaypointSample& sample, float value) {
            return sample.t < value;
        }
    );

    if (upper == track.begin()) return upper->p;
    const WaypointSample& b = *upper;
    const WaypointSample& a = *(upper - 1);
    float span = std::max(1e-5f, b.t - a.t);
    float u = std::max(0.0f, std::min(1.0f, (t - a.t) / span));
    return {
        a.p.x + (b.p.x - a.p.x) * u,
        a.p.y + (b.p.y - a.p.y) * u,
        a.p.z + (b.p.z - a.p.z) * u
    };
}

static void ApplyWaypointPlayback(float t)
{
    if (!gTrajectoryLoaded) return;

    gPos.resize(gWaypointTracks.size());
    for (size_t i = 0; i < gWaypointTracks.size(); ++i) {
        gPos[i] = SampleWaypointTrack(gWaypointTracks[i], t);
    }
}

static void LoadEmotionLogData()
{
    gEmotionLogLabels = {"happy", "sad", "sleepy", "brave", "grumpy", "scared", "shy"};
    gEmotionLogSegments.clear();
    gEmotionLogLoaded = false;
    gLastEmotionLogIndex = -1;

    std::filesystem::path labelsPath = ResolveJsonPath("anchor_labels.json");
    if (!labelsPath.empty()) {
        try {
            std::ifstream labelsFile(labelsPath);
            json labelsJson;
            labelsFile >> labelsJson;
            if (labelsJson.is_array()) {
                std::vector<std::string> labels;
                for (const auto& item : labelsJson) {
                    if (item.is_string()) {
                        labels.push_back(item.get<std::string>());
                    }
                }
                if (!labels.empty()) {
                    gEmotionLogLabels = std::move(labels);
                }
            }
        } catch (const std::exception& err) {
            std::cerr << "[EMOTION] Could not load labels from "
                      << labelsPath.string() << ": " << err.what() << "\n";
        }
    }

    std::filesystem::path weightsPath = ResolveJsonPath("clap_weights.json");
    if (weightsPath.empty()) {
        std::cerr << "[EMOTION] Could not find clap_weights.json; emotion logging disabled.\n";
        return;
    }

    try {
        std::ifstream weightsFile(weightsPath);
        json weightsJson;
        weightsFile >> weightsJson;
        if (!weightsJson.is_array()) {
            std::cerr << "[EMOTION] clap_weights.json is not an array; emotion logging disabled.\n";
            return;
        }

        for (const auto& item : weightsJson) {
            if (!item.is_object() || !item.contains("weights") || !item["weights"].is_array()) {
                continue;
            }

            EmotionLogSegment segment;
            segment.start = item.value("start", 0.0f);
            segment.end = item.value("end", segment.start);
            for (const auto& value : item["weights"]) {
                if (value.is_number()) {
                    segment.weights.push_back(value.get<float>());
                }
            }

            if (!segment.weights.empty()) {
                gEmotionLogSegments.push_back(segment);
            }
        }
    } catch (const std::exception& err) {
        std::cerr << "[EMOTION] Could not load weights from "
                  << weightsPath.string() << ": " << err.what() << "\n";
        return;
    }

    std::sort(gEmotionLogSegments.begin(), gEmotionLogSegments.end(),
              [](const EmotionLogSegment& a, const EmotionLogSegment& b) {
                  return a.start < b.start;
              });

    gEmotionLogLoaded = !gEmotionLogSegments.empty();
    if (gEmotionLogLoaded) {
        std::cout << "[EMOTION] Loaded " << gEmotionLogSegments.size()
                  << " segments from " << weightsPath.string() << "\n";
    } else {
        std::cerr << "[EMOTION] No usable segments in "
                  << weightsPath.string() << "; emotion logging disabled.\n";
    }
}

static void LogEmotionForTime(float t)
{
    if (!gEmotionLogLoaded) return;

    int index = -1;
    for (int i = 0; i < (int)gEmotionLogSegments.size(); ++i) {
        const EmotionLogSegment& segment = gEmotionLogSegments[i];
        if (t >= segment.start && t < segment.end) {
            index = i;
            break;
        }
    }

    if (index < 0 && t >= gEmotionLogSegments.back().end) {
        index = (int)gEmotionLogSegments.size() - 1;
    }

    if (index < 0 || index == gLastEmotionLogIndex) return;
    gLastEmotionLogIndex = index;

    const EmotionLogSegment& segment = gEmotionLogSegments[index];
    std::cout << std::fixed << std::setprecision(3)
              << "[EMOTION] t=" << t
              << " segment=" << segment.start << "-" << segment.end
              << " weights:";

    for (size_t i = 0; i < segment.weights.size(); ++i) {
        const std::string label = (i < gEmotionLogLabels.size())
            ? gEmotionLogLabels[i]
            : ("w" + std::to_string(i));
        std::cout << " " << label << "=" << segment.weights[i];
    }
    std::cout << "\n";
}

//music helper functions
void StartAudio(const std::string &filename)
{
    namespace fs = std::filesystem;

    // Initialize engine if needed
    if (!gAudioInit) {
        ma_result r = ma_engine_init(NULL, &gEngine);
        if (r != MA_SUCCESS) {
            std::cerr << "Failed to init audio engine, code = " << r << "\n";
            return;
        }
        gAudioInit = true;
        std::cout << "Audio engine initialized\n";
    }
    
    // Clean up previous sound if loaded
    if (gAudioLoaded) {
        ma_sound_uninit(&gMusic);
        gAudioLoaded = false;
    }

    gAudioDuration = 0.0f;
    gAudioSyncMode = false;

    fs::path path = ResolveAudioPath(filename);
    std::string pathStr = path.string();

    if (!fs::exists(path)) {
        std::cerr << "Audio file not found: " << pathStr << "\n";
        return;
    }

    std::cout << "Loading audio: " << pathStr << "\n";

    // Load audio file
    ma_result r = ma_sound_init_from_file(&gEngine,
                                          pathStr.c_str(),
                                          0, nullptr, nullptr,
                                          &gMusic);
    if (r != MA_SUCCESS) {
        std::cerr << "Failed to load audio file, code = " << r << "\n";
        return;
    }

    // Get audio duration
    ma_uint64 lengthInFrames;
    ma_sound_get_length_in_pcm_frames(&gMusic, &lengthInFrames);
    ma_uint32 sampleRate = ma_engine_get_sample_rate(&gEngine);
    gAudioDuration = (float)lengthInFrames / (float)sampleRate;
    
    gAudioLoaded = true;
    
    std::cout << "Audio loaded successfully (duration: " 
              << gAudioDuration << "s)\n";
    
}

void PauseAudio() {
    if (gAudioLoaded && ma_sound_is_playing(&gMusic)) {
        ma_sound_stop(&gMusic);
        std::cout << "Audio paused\n";
    }
}

void ResumeAudio() {
    if (gAudioLoaded && !ma_sound_is_playing(&gMusic)) {
        ma_sound_start(&gMusic);
        std::cout << "Audio resumed\n";
    }
}

void StopAudioAndReset() {
    if (gAudioLoaded) {
        ma_sound_stop(&gMusic);
        ma_sound_seek_to_pcm_frame(&gMusic, 0);  // Reset to beginning
        gAudioSyncMode = false;
        std::cout << "Audio stopped and reset\n";
    }
}

void SeekAudioTo(float timeInSeconds) {
    if (!gAudioLoaded) return;
    
    ma_uint32 sampleRate = ma_engine_get_sample_rate(&gEngine);
    ma_uint64 frameIndex = (ma_uint64)(timeInSeconds * sampleRate);
    ma_sound_seek_to_pcm_frame(&gMusic, frameIndex);
    
    std::cout << "Audio seeked to " << timeInSeconds << "s\n";
}

float GetAudioPlaybackTime() {
    if (!gAudioLoaded) return 0.0f;
    
    float cursor;
    ma_sound_get_cursor_in_seconds(&gMusic, &cursor);
    return cursor;
}

bool IsAudioActuallyPlaying() {
    return gAudioLoaded && ma_sound_is_playing(&gMusic);
}

float GetAudioDuration() {
    return gAudioDuration;
}


// Formation examples (XZ plane)
// returns N points on a circle of radius
static std::vector<Vec3> SampleCircle(int n, float radius, float phase) {
    std::vector<Vec3> pts(n);
    for(int i=0;i<n;i++){
        float a = (i/(float)n)*2.0f*M_PI + phase;
        pts[i] = { radius*std::cos(a), radius*std::sin(a), 0.0f};
    }
    return pts;
}

// returns N points on a center horizontal line
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

// returns N points of wave across X, amp set on Y
static std::vector<Vec3> SampleWave(int n, float width, float amp, float phase){
    std::vector<Vec3> pts(n);
    for(int i=0;i<n;i++){
        float u = (i/(float)(n-1));
        float x = (u - 0.5f) * width;
        float y = std::sin(u*2.0f*M_PI + phase) * amp;
        pts[i] = { x, y, 0.0f };
    }
    return pts;
}

// returns N points on heart curve on XY
static std::vector<Vec3> SampleHeart(int n, float scale, float phase){
    std::vector<Vec3> pts(n);
    for(int i=0;i<n;i++){
        float t = (i/(float)n)*2.0f*M_PI + phase;
        float x = 16*std::pow(std::sin(t),3);
        float y = 13*std::cos(t) - 5*std::cos(2*t) - 2*std::cos(3*t) - std::cos(4*t);
        pts[i] = { (x/16.0f)*scale, (y/13.0f)*scale, 0.0f };
    }
    return pts;
}

static std::vector<Vec3> BlendSlots(const std::vector<Vec3>& from,
                                    const std::vector<Vec3>& to,
                                    float t)
{
    std::vector<Vec3> blended;
    blended.reserve(from.size());
    for (size_t i = 0; i < from.size() && i < to.size(); ++i) {
        blended.push_back(LerpVec3(from[i], to[i], t));
    }
    return blended;
}


static void DrawBoundsBox()
{
    Boundaries b = GetBoxBounds();  // from boids.cpp

    float x0 = b.xmin, x1 = b.xmax;
    float y0 = b.ymin, y1 = b.ymax;
    float z0 = b.zmin, z1 = b.zmax;

    glDisable(GL_LIGHTING);
    glColor3f(0.9f, 0.3f, 0.3f); // reddish box

    // --- bottom rectangle (z = z0) ---
    glBegin(GL_LINE_LOOP);
        glVertex3f(x0, y0, z0);
        glVertex3f(x1, y0, z0);
        glVertex3f(x1, y1, z0);
        glVertex3f(x0, y1, z0);
    glEnd();

    // --- top rectangle (z = z1) ---
    glBegin(GL_LINE_LOOP);
        glVertex3f(x0, y0, z1);
        glVertex3f(x1, y0, z1);
        glVertex3f(x1, y1, z1);
        glVertex3f(x0, y1, z1);
    glEnd();

    // --- vertical edges (vary z, keep x/y fixed) ---
    glBegin(GL_LINES);
        glVertex3f(x0, y0, z0); glVertex3f(x0, y0, z1);
        glVertex3f(x1, y0, z0); glVertex3f(x1, y0, z1);
        glVertex3f(x1, y1, z0); glVertex3f(x1, y1, z1);
        glVertex3f(x0, y1, z0); glVertex3f(x0, y1, z1);
    glEnd();

    glEnable(GL_LIGHTING);
}

// Compute the target slots for current formation at time gTime
static void ResampleSlots(){
    float blend = smooth01(gSimTime / kLineToCircleTime);
    float circleRadius = (cfg.drone_config.init_dist * gDroneCount) / (2.0f * M_PI);
    std::vector<Vec3> lineSlots = SampleLine(gDroneCount);
    std::vector<Vec3> circleSlots = SampleCircle(gDroneCount, circleRadius, 0.0f);
    gSlots = BlendSlots(lineSlots, circleSlots, blend);
}

// Resize/initialize arrays and seed positions
// Forwards new count to boids
static void ResizeArrays(){
    gSlots.resize(gDroneCount);
    gPos.resize  (gDroneCount);
    ResizeBoids(gDroneCount);
    // Start from a line; the target slots will quickly move to a circle.
    auto init = SampleLine(gDroneCount);
    for(int i=0;i<gDroneCount;i++){
        gPos[i] = { init[i].x, init[i].y, gAltitude };
    }
}

// Steers drones towards target spots, seperation to avoid collision
static void UpdateFollowSlots(float dt) {
    const float cohesion = 0.85f;   // steer strength to target position
    const float maxStep  = 6.0f;    // max movement per second
    const float sepDist2 = 0.36f;   // (0.6 m)^2 separation
    int stride = std::max(1, gDroneCount/250);

    for (int i=0; i<gDroneCount; i++) {
        Vec3 target = { gSlots[i].x, gSlots[i].y, gAltitude};
        Vec3 p = gPos[i];

        //move toward target (clamped)
        Vec3 to = { target.x - p.x, target.y - p.y, target.z - p.z };
        float L = std::sqrt(to.x*to.x + to.y*to.y + to.z*to.z);
        if (L > 1e-5f) {
            float s = std::min(maxStep*dt, L) / L;
            to = { to.x*s, to.y*s, to.z*s };
        }

        //separation so drones don't collide
        Vec3 sep{0,0,0};
        for (int j=(i+1)%stride; j<gDroneCount; j+=stride) {
            Vec3 q = gPos[j];
            float dx=p.x-q.x, dy=p.y-q.y, dz=p.z-q.z;
            float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 > 1e-6f && d2 < sepDist2) {
                float inv = 1.0f/std::sqrt(d2);
                sep.x += dx*inv; sep.y += dy*inv; sep.z += dz*inv;
            }
        }
        //combine target spot and seperation
        p.x += cohesion*to.x + 0.6f*sep.x*dt;
        p.y += cohesion*to.y + 0.6f*sep.y*dt;
        p.z += cohesion*to.z + 0.6f*sep.z*dt;

        gPos[i] = p;
    }
}

// collision check: if collide then print smth
static void CheckCollisions(const std::vector<Vec3>& pos, float nowSimTime)
{
    const float collisionDistance = 2.0f * DRONE_RADIUS;
    const float collisionDistance2 = collisionDistance * collisionDistance;

    const int n = (int)pos.size();
    if (n < 2) return;

    std::set<std::pair<int, int>> currentCollisions;

    for (int i = 0; i < n; ++i) {
        const Vec3& a = pos[i];
        for (int j = i + 1; j < n; ++j) {
            const Vec3& b = pos[j];
            float dx = a.x - b.x;
            float dy = a.y - b.y;
            float dz = a.z - b.z;
            float d2 = dx*dx + dy*dy + dz*dz;

            if (d2 <= collisionDistance2) {
                std::pair<int, int> pair{i, j};
                currentCollisions.insert(pair);
                if (gActiveCollisionPairs.find(pair) == gActiveCollisionPairs.end()) {
                    float dist = std::sqrt(d2);
                    std::cout << "[COLLISION] t=" << nowSimTime
                              << " drone " << i << " & drone " << j
                              << " dist=" << dist
                              << " threshold=" << collisionDistance << "\n";
                }
            }
        }
    }

    gActiveCollisionPairs = std::move(currentCollisions);
}


//-------------------------------------------------------------------
//data
//-------------------------------------------------------------------

auto& labels = GetEmotionLabels();

// --- CAMERA ---
static void ApplyCamera(){
    float ex = gCamR * std::sin(gCamPhi) * std::cos(gCamTheta);
    float ey = gCamR * std::cos(gCamPhi);
    float ez = gCamR * std::sin(gCamPhi) * std::sin(gCamTheta);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(ex,ey,ez,  0,0,1.5,  0,0,1);
}

// Draw a drone (cone)
static void DrawDrone(){
    glutSolidCone(DRONE_RADIUS, 0.022, 6, 1);       //92mm * 92mm* 22mm
}

static void DrawBitmapString(float x, float y, void* font, const char* s) {
    glRasterPos2f(x, y);
    for (const char* p = s; *p; ++p) glutBitmapCharacter(font, *p);
}

static void DrawHUD() {
    if (gTime > gHudUntil) return;
    if (gHudMsg.empty()) return;

    // switch to screen-space
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    int w = glutGet(GLUT_WINDOW_WIDTH);
    int h = glutGet(GLUT_WINDOW_HEIGHT);
    gluOrtho2D(0, w, 0, h);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_LIGHTING);
    glColor3f(1.0f, 1.0f, 1.0f);

    glRasterPos2i(20, h - 40);

    for (char c : gHudMsg)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);

    glEnable(GL_LIGHTING);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    
}

// Simple bitmap text (overlay/HUD)
static void DoRasterString(float x, float y, const char *s){
    glRasterPos3f(x, y, 0);
    while(*s != '\0'){ glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *s); s++; }
}

// --- GLUT CALLBACKS ---
// sets up camera, position and lights, draws position grid, HUD and drones
static void Display(){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Basic light
    GLfloat lpos[] = { 5.0f, 6.0f, 4.0f, 1.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, lpos);

    ApplyCamera();

    // Ground grid
    glDisable(GL_LIGHTING);
    glColor3f(0.20f,0.20f,0.23f);
    glBegin(GL_LINES);
    for(int i=-10;i<=10;i++){
        glVertex3f((float)i, -10.0f, 0.0f); glVertex3f((float)i,  10.0f, 0.0f);
        glVertex3f(-10.0f, (float)i, 0.0f); glVertex3f( 10.0f, (float)i, 0.0f); 
    }
    glEnd();
    glEnable(GL_LIGHTING);

    //draw boundaries
    DrawBoundsBox();

    // Choose positioning for either boids or example formations
    const std::vector<Vec3>& positions =
        (gUseWaypointPlayback && gTrajectoryLoaded) ? gPos :
        (gUseBoids ? GetBoidPositions() : gPos);

    // draw drones and color by index gradient
    for (int i=0; i<gDroneCount; ++i){
        const Vec3 &p = positions[i];
        glPushMatrix();
            glTranslatef(p.x, p.y, p.z);
            // glRotatef(-90.0f, 1,0,0); // drone cone points up
            float u = i/(float)gDroneCount;
            glColor3f(0.7f, 0.85f*u, 1.0f - 0.6f*u);
            DrawDrone();
        glPopMatrix();
    }

    // HUD text (top-left in world space near camera; simpler than switching to ortho)
    glDisable(GL_LIGHTING);
    glColor3f(1,1,1);
    glPushMatrix();
    glLoadIdentity();
    //move text slightly in front of camera
    glTranslatef(-3.5f, 3.5f, -10.0f);

    const char* state = gPlaying ? ">" : "||";  // << use string not multi-char '||'
    char buf[256];

    if (gUseWaypointPlayback && gTrajectoryLoaded) {
        std::snprintf(buf, sizeof(buf),
            "waypoint t: %.2f / %.2f s", gSimTime, gWaypointDuration);
        DoRasterString(0, 0.f, buf);

        std::snprintf(buf, sizeof(buf),
            "trajectory drones: %d", gDroneCount);
        DoRasterString(0, -0.5f, buf);
    } else if (gUseBoids){
        const BoidParams& P = GetBoidParams();
        const auto& w      = GetLastWeights();
        
        std::snprintf(buf, sizeof(buf),
            "t: %.2f s", gSimTime);
        DoRasterString(0, 0.f, buf);
        
        std::snprintf(buf, sizeof(buf),
            "r_sep: %.2f  r_nei: %.2f  |  k_sep: %.2f  k_ali: %.2f  k_coh: %.2f  k_goal: %.2f",
            P.r_sep, P.r_nei, P.k_sep, P.k_ali, P.k_coh, P.k_goal);
        DoRasterString(0, -0.5f, buf);

        std::snprintf(buf, sizeof(buf),
            "vmax: %.2f  amax: %.2f  |  altitude: %.2f  jitter: %.2f",
            P.vmax, P.amax, P.altitude, P.jitter);
        DoRasterString(0, -1.f, buf);

        if (!labels.empty() && !w.empty()) {
            float y = -1.5f;
            float dy = 0.4f;
            float x = 0.f;
            float dx = 0.4f;
            std::string line;

            for (int i = 0; i < (int)labels.size(); i++) {
                char buf2[64];
                std::snprintf(buf2, sizeof(buf2), "%s:%.2f  ",
                            labels[i].c_str(),
                            (i < (int)w.size() ? w[i] : 0.0f));

                // Append to current line
                line += buf2;

                // Wrap every ~4 labels
                if ((i+1) % 4 == 0 || i == (int)labels.size() - 1) {
                    DoRasterString(x, y, line.c_str());
                    x += dx;
                    y -= dy;
                    line.clear();
                }
            }
        }
    }
    glPopMatrix();
    glEnable(GL_LIGHTING);

    glutSwapBuffers();

    DrawHUD();
}

static void Idle(){
    static bool first = true;
    static float lastWallTime = 0.0f;

    float wallTime = 0.001f * glutGet(GLUT_ELAPSED_TIME);

    if (first) {
        lastWallTime = wallTime;
        first = false;
    }

    // Compute dt = elapsed real time
    float dt = wallTime - lastWallTime;
    lastWallTime = wallTime;

    // Safety clamps for huge jumps
    if (dt < 0.0f) dt = 0.0f;
    if (dt > 0.1f) dt = 0.1f;

    if (gPlaying) {
        gSimTime += dt * gSpeed;  // you can change gSpeed to scrub faster/slower
    }

    if (gUseWaypointPlayback && gTrajectoryLoaded) {
        ApplyWaypointPlayback(gSimTime);
    } else {
        SetSimTime(gSimTime);
        ResampleSlots();

        // Advance generated positions only when playing.
        if (gPlaying) {
            if (gUseBoids) {
                UpdateBoids(dt * gSpeed, gSlots);
            } else {
                UpdateFollowSlots(dt * gSpeed);
            }
        }
    }

    gFrameIndex++;
    const std::vector<Vec3>& positions =
        (gUseWaypointPlayback && gTrajectoryLoaded) ? gPos :
        (gUseBoids ? GetBoidPositions() : gPos);
    CheckCollisions(positions, gSimTime);
    LogEmotionForTime(gSimTime);


    glutPostRedisplay();
}

static void Reshape(int w, int h){
    gWinW=w; gWinH=h; if(h==0) h=1;
    glViewport(0,0,w,h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (double)w/(double)h, 0.1, 1000.0);
}

// handles all keyboard input
static void Keyboard(unsigned char key, int, int){
    switch(key){
        case 'b': case 'B':
            if (gUseWaypointPlayback && gTrajectoryLoaded) {
                gHudMsg = "Waypoint playback is active";
                gHudUntil = gSimTime + 1.5f;
                break;
            }
            gUseBoids = !gUseBoids;
            if (!gUseBoids) ResetVelocities();
            break;
            
        case 27: 
            // ESC: cleanup and exit
            if (gAudioLoaded) {
                ma_sound_uninit(&gMusic);
            }
            if (gAudioInit) {
                ma_engine_uninit(&gEngine);
            }
            std::exit(0); 
            break;
            
        // space bar: controls both audio and visual
        case ' ': 
            gPlaying = !gPlaying;
            
            // Sync audio playback with simulation state
            if (gAudioLoaded && gAudioSyncMode) {
                if (gPlaying) {
                    ResumeAudio();
                    std::cout << "Playing (audio + visuals)\n";
                } else {
                    PauseAudio();
                    std::cout << "Paused (audio + visuals)\n";
                }
            } else {
                std::cout << (gPlaying ? "Playing (visuals only)\n" 
                                       : "Paused (visuals only)\n");
            }
            break;
            
        // case '+': case '=': 
        //     gDroneCount = std::min(3000, gDroneCount + 1); 
        //     ResizeArrays(); 
        //     break;
            
        // case '-': case '_': 
        //     gDroneCount = std::max(1, gDroneCount - 1); 
        //     ResizeArrays(); 
        //     break;
            
        // case '1': gForm = CIRCLE; break;
        // case '2': gForm = LINE;   break;
        // case '3': gForm = WAVE;   break;
        // case '4': gForm = HEART;  break;
        
        // case 's': case 'S': 
        //     gSpin = !gSpin; 
        //     break;
            
        // case ',': 
        //     gSpeed = clampf(gSpeed - 0.05f, 0.1f, 3.0f); 
        //     std::cout << "Speed: " << gSpeed << "x\n";
        //     break;
            
        // case '.': 
        //     gSpeed = clampf(gSpeed + 0.05f, 0.1f, 3.0f); 
        //     std::cout << "Speed: " << gSpeed << "x\n";
        //     break;
            
        //'E': loads emotional labels and starts music
        case 'e': case 'E': {
            float t = gSimTime;  // Use current simulation time
            
            // Load emotion data
            // bool emotionsOk = ReloadAndApplyEmotions(t);
            
            if (!gAudioLoaded) {
                StartAudio(gAudioInputPath);
            }

            if (gAudioLoaded) {
                // Seek audio to current simulation time
                SeekAudioTo(t);

                // Start audio playback
                ma_sound_start(&gMusic);

                // Enable audio sync mode
                gAudioSyncMode = true;
                gPlaying = true;

                gHudMsg = "Emotions + Audio loaded and playing";
                std::cout << "\n=== AUDIO + EMOTIONS STARTED ===\n";
                std::cout << "Song: " << ResolveAudioPath(gAudioInputPath) << "\n";
                std::cout << "Starting at t=" << t << "s\n";
                std::cout << "Audio duration: " << gAudioDuration << "s\n";
                std::cout << "Controls:\n";
                std::cout << "  Space = Pause/Resume\n";
                std::cout << "  R = Stop and reset\n";
                std::cout << "================================\n\n";
            } else {
                gHudMsg = "Emotions loaded, Audio failed";
                std::cerr << "ERROR: Failed to load audio\n";
            }
            
            gHudUntil = gSimTime + 2.5f;
            glutPostRedisplay();
            break;
        }
        
        // NEW: 'R': stop and reset
        case 'r': case 'R': {
            if (gAudioLoaded) {
                StopAudioAndReset();
                gSimTime = 0.0f;
                gPlaying = false;
                ResizeArrays();
                ResampleSlots();
                InitBoids(gDroneCount);
                if (gUseWaypointPlayback && gTrajectoryLoaded) {
                    ApplyWaypointPlayback(0.0f);
                }
                gLastEmotionLogIndex = -1;
                gActiveCollisionPairs.clear();
                LogEmotionForTime(gSimTime);
                
                gHudMsg = "Audio stopped, simulation reset to t=0";
                gHudUntil = 1.5f;
                
                std::cout << "=============\n\n";
                std::cout << "Audio stopped\n";
                std::cout << "Simulation time reset to 0\n";
                std::cout << "Press 'e' to reload and start\n";
                std::cout << "=============\n\n";
                
                glutPostRedisplay();
            } else {
                std::cout << "No audio loaded to reset\n";
            }
            break;
        }
    }
}

// handles orbit camera
static void Mouse(int button, int state, int x, int y){
    if(button == GLUT_LEFT_BUTTON){
        gDragging = (state == GLUT_DOWN);
        gLastX = x; gLastY = y;
    }
    // Wheel up/down (note: on some GLUT builds, 3/4 are scroll events)
    if(button == 3 && state == GLUT_DOWN){ gCamR = clampf(gCamR*0.9f, 2.0f, 80.0f); }
    if(button == 4 && state == GLUT_DOWN){ gCamR = clampf(gCamR*1.1f, 2.0f, 80.0f); }
}

static void Motion(int x, int y){
    if(!gDragging) return;
    int dx = x - gLastX; int dy = y - gLastY; gLastX = x; gLastY = y;
    gCamTheta -= dx * 0.005f;
    gCamPhi   -= dy * 0.005f;
    const float eps = 0.001f;
    gCamPhi = clampf(gCamPhi, eps, (float)M_PI - eps);
}

int main(int argc, char** argv){
    gDroneCount = cfg.drone_config.num_drones;
    ParseCommandLine(&argc, argv);
    if (gUseWaypointPlayback && !LoadWaypointTrajectory(gTrajectoryInputPath)) {
        gUseWaypointPlayback = false;
    }
    LoadEmotionLogData();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(gWinW, gWinH);
    glutCreateWindow("Drone Swarm Visualizer — Legacy OpenGL");
    
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);

    ResizeArrays();         //allocate gPos/gSlots
    ResampleSlots();        //starting target positions
    InitBoids(gDroneCount); //initlialize boids
    if (gUseWaypointPlayback && gTrajectoryLoaded) {
        ApplyWaypointPlayback(0.0f);
    }
    LogEmotionForTime(0.0f);
    

    glutDisplayFunc(Display);
    glutIdleFunc(Idle);
    glutReshapeFunc(Reshape);
    glutKeyboardFunc(Keyboard);
    glutMouseFunc(Mouse);
    glutMotionFunc(Motion);

    std::cout << "\n=======================================\n";
    std::cout << "   DRONE SWARM VISUALIZER - AUDIO SYNC\n";
    std::cout << "=======================================\n";
    std::cout << "Audio file: " << ResolveAudioPath(gAudioInputPath) << "\n";
    if (gUseWaypointPlayback && gTrajectoryLoaded) {
        std::cout << "Waypoint file: " << gResolvedTrajectoryPath << "\n";
        std::cout << "Playback mode: exported waypoint CSV\n";
    } else {
        std::cout << "Playback mode: live boids generation\n";
    }
    std::cout << "Controls:\n";
    std::cout << "  E = Load emotions + start audio\n";
    std::cout << "  Space = Pause/Resume\n";
    std::cout << "  R = Stop and reset\n";
    std::cout << "  B = Toggle boids (live mode only)\n";
    std::cout << "  +/- = Change drone count\n";
    std::cout << "  1-4 = Formation shapes\n";
    std::cout << "CLI: ./drones [song-path] --trajectory <trajectory.csv>\n";
    std::cout << "     ./drones --live-boids\n";
    std::cout << "=======================================\n\n";

    glutMainLoop();

    //clean up
    if (gAudioLoaded) {
        ma_sound_uninit(&gMusic);
    }
    if (gAudioInit) {
        ma_engine_uninit(&gEngine);
    }
    return 0;
}
