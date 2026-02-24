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
#include <iostream>
#include <filesystem>

#include <thread>
#include <chrono>
//audio playback
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "config.h" 

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

//music
ma_engine gEngine;
ma_sound  gMusic;
bool gAudioInit = false;
static bool gAudioLoaded = false;      // Track if sound file is loaded
static bool gAudioSyncMode = false;    // When true, sync visuals to audio
static float gAudioDuration = 0.0f;    // Total audio length in seconds


//collision prints
static int   gFrameIndex = 0;
static float gLastCollisionPrintTime = -1e9f;


//music helper functions
void StartAudio(const std::string &filename = "testSong.mp3")
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
    
    // Build file path
    fs::path cwd  = fs::current_path();
    fs::path path = cwd.parent_path() / "audio" / filename;
    std::string pathStr = path.string();

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
    float t = gTime;
    gSlots = SampleLine(gDroneCount);
}

// Resize/initialize arrays and seed positions
// Forwards new count to boids
static void ResizeArrays(){
    gSlots.resize(gDroneCount);
    gPos.resize  (gDroneCount);
    ResizeBoids(gDroneCount);
    // Place initial positions on a small disc around origin at altitude
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
    const float r = DRONE_RADIUS + 0.004;
    const float touch2 = (2.0f * r) * (2.0f * r);

    const int n = (int)pos.size();
    if (n < 2) return;

    //check every 5 frames
    if ((gFrameIndex % 5) != 0) return;

    const float printCooldown = 0.20f; //0.2 sec
    if (nowSimTime - gLastCollisionPrintTime < printCooldown) return;

    // For big swarms, don't do full O(n^2) every time.
    // Use a stride to reduce checks
    const int stride = std::max(1, n / 200);

    int printed = 0;
    const int maxPrintPerCall = 5;

    for (int i = 0; i < n && printed < maxPrintPerCall; ++i) {
        const Vec3& a = pos[i];

        // sample some j's instead of all and find euclidean distance between drones
        for (int j = i + 1; j < n && printed < maxPrintPerCall; j += stride) {
            const Vec3& b = pos[j];
            float dx = a.x - b.x;
            float dy = a.y - b.y;
            float dz = a.z - b.z;
            float d2 = dx*dx + dy*dy + dz*dz;

            //if inner distance < 2r they collided
            if (d2 <= touch2) {
                std::cout << "[COLLISION] t=" << nowSimTime
                          << " drone " << i << " & drone " << j
                          << " dist=" << std::sqrt(d2) << "\n";
                ++printed;
                gLastCollisionPrintTime = nowSimTime;
            }
        }
    }
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
    const std::vector<Vec3>& positions = gUseBoids ? GetBoidPositions() : gPos;

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

    if (gUseBoids){
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

    // Audio driven time sync
    // 1) Advance simulation time if playing (no wall clock)
    if (gPlaying) {
        gSimTime += dt * gSpeed;  // you can change gSpeed to scrub faster/slower
    }
    SetSimTime(gSimTime);
    // 2) Update targets for this time
    //ResampleSlots(); // still uses gTime internally

    // 3) Advance positions - ONLY WHEN PLAYING
    if (gPlaying) {
        UpdateBoids(dt * gSpeed, gSlots);
    }
    gFrameIndex++;
    const std::vector<Vec3>& positions = gUseBoids ? GetBoidPositions() : gPos;
    CheckCollisions(positions, gSimTime);


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
                    StartAudio("testSong.mp3");
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
    // StartAudio();
    // std::this_thread::sleep_for(std::chrono::duration<float>(0.2));
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
    

    glutDisplayFunc(Display);
    glutIdleFunc(Idle);
    glutReshapeFunc(Reshape);
    glutKeyboardFunc(Keyboard);
    glutMouseFunc(Mouse);
    glutMotionFunc(Motion);

    std::cout << "\n=======================================\n";
    std::cout << "   DRONE SWARM VISUALIZER - AUDIO SYNC\n";
    std::cout << "=======================================\n";
    std::cout << "Controls:\n";
    std::cout << "  E = Load emotions + start audio\n";
    std::cout << "  Space = Pause/Resume\n";
    std::cout << "  R = Stop and reset\n";
    std::cout << "  B = Toggle boids\n";
    std::cout << "  +/- = Change drone count\n";
    std::cout << "  1-4 = Formation shapes\n";
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
