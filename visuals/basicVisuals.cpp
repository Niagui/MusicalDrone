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


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// --- GLOBAL VARIABLES ---

int   gWinW = 1200, gWinH = 800; //window size
int   gDroneCount = 20;        // +/- to change
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
enum Form { CIRCLE=1, LINE, WAVE, HEART };
Form  gForm = CIRCLE;

float gAltitude = 1.6f;   // Y height of all formations
float gSpeed    = 1.0f;   // Global animation speed


// Per-drone state: current positions and targets for formations
std::vector<Vec3> gPos; //current drone position
std::vector<Vec3> gSlots; //target position

static inline float clampf(float x, float lo, float hi){ return x<lo?lo:(x>hi?hi:x); }
static inline float Lerp(float a, float b, float t) { return a + (b-a)*t; }


// Formation examples (XZ plane)
// returns N points on a circle of radius
static std::vector<Vec3> SampleCircle(int n, float radius, float phase) {
    std::vector<Vec3> pts(n);
    for(int i=0;i<n;i++){
        float a = (i/(float)n)*2.0f*M_PI + phase;
        pts[i] = { radius*std::cos(a), 0.0f, radius*std::sin(a) };
    }
    return pts;
}

// returns N points on a center horizontal line
static std::vector<Vec3> SampleLine(int n, float length){
    std::vector<Vec3> pts(n);
    for(int i=0;i<n;i++){
        float x = ((i/(float)(n-1)) - 0.5f) * length;
        pts[i] = { x, 0.0f, 0.0f };
    }
    return pts;
}

// returns N points of wave across X, amp set on Z
static std::vector<Vec3> SampleWave(int n, float width, float amp, float phase){
    std::vector<Vec3> pts(n);
    for(int i=0;i<n;i++){
        float u = (i/(float)(n-1));
        float x = (u - 0.5f) * width;
        float z = std::sin(u*2.0f*M_PI + phase) * amp;
        pts[i] = { x, 0.0f, z };
    }
    return pts;
}

// returns N points on heart curve on XZ
static std::vector<Vec3> SampleHeart(int n, float scale, float phase){
    std::vector<Vec3> pts(n);
    for(int i=0;i<n;i++){
        float t = (i/(float)n)*2.0f*M_PI + phase;
        float x = 16*std::pow(std::sin(t),3);
        float y = 13*std::cos(t) - 5*std::cos(2*t) - 2*std::cos(3*t) - std::cos(4*t);
        pts[i] = { (x/16.0f)*scale, 0.0f, (y/13.0f)*scale };
    }
    return pts;
}

// Compute the target slots for current formation at time gTime
static void ResampleSlots(){
    float t = gTime;
    switch(gForm){
        case CIRCLE: gSlots = SampleCircle(gDroneCount, 3.0f, gSpin? 0.25f*t : 0.0f); break;
        case LINE:   gSlots = SampleLine  (gDroneCount, 10.0f);                         break;
        case WAVE:   gSlots = SampleWave  (gDroneCount, 10.0f, 1.3f, 1.4f*t);           break;
        case HEART:  gSlots = SampleHeart (gDroneCount, 1.9f,  gSpin? 0.15f*t : 0.0f);  break;
    }
}

// Resize/initialize arrays and seed positions
// Forwards new count to boids
static void ResizeArrays(){
    gSlots.resize(gDroneCount);
    gPos.resize  (gDroneCount);
    ResizeBoids(gDroneCount);
    // Place initial positions on a small disc around origin at altitude
    auto init = SampleCircle(gDroneCount, 1.2f, 0.0f);
    for(int i=0;i<gDroneCount;i++){
        gPos[i] = { init[i].x, gAltitude, init[i].z };
    }
}

// Steers drones towards target spots, seperation to avoid collision
static void UpdateFollowSlots(float dt) {
    const float cohesion = 0.85f;   // steer strength to target position
    const float maxStep  = 6.0f;    // max movement per second
    const float sepDist2 = 0.36f;   // (0.6 m)^2 separation
    int stride = std::max(1, gDroneCount/250);

    for (int i=0; i<gDroneCount; i++) {
        Vec3 target = { gSlots[i].x, gAltitude, gSlots[i].z };
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

//data

auto& labels = GetEmotionLabels();

// --- CAMERA ---
static void ApplyCamera(){
    float ex = gCamR * std::sin(gCamPhi) * std::cos(gCamTheta);
    float ey = gCamR * std::cos(gCamPhi);
    float ez = gCamR * std::sin(gCamPhi) * std::sin(gCamTheta);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(ex,ey,ez,  0,1.5,0,  0,1,0);
}

// Draw a drone (cone)
static void DrawDrone(){
    glutSolidCone(0.06, 0.15, 10, 1);
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
        glVertex3f((float)i, 0.0f, -10.0f); glVertex3f((float)i, 0.0f, 10.0f);
        glVertex3f(-10.0f, 0.0f, (float)i); glVertex3f(10.0f, 0.0f, (float)i);
    }
    glEnd();
    glEnable(GL_LIGHTING);

    // Choose positioning for either boids or example formations
    const std::vector<Vec3>& positions = gUseBoids ? GetBoidPositions() : gPos;

    // draw drones and color by index gradient
    for (int i=0; i<gDroneCount; ++i){
        const Vec3 &p = positions[i];
        glPushMatrix();
            glTranslatef(p.x, p.y, p.z);
            glRotatef(-90.0f, 1,0,0); // drone cone points up
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
    // 1) Advance simulation time if playing (no wall clock)
    if (gPlaying) {
        gSimTime += kFixedDt * gSpeed;  // you can change gSpeed to scrub faster/slower
    }
    SetSimTime(gSimTime);
    // 2) Update targets for this time
    ResampleSlots(); // still uses gTime internally; we’ll align that below

    // 3) Advance positions
    if (gUseBoids) {
        // we’ll pass time into the boids system
        
        UpdateBoids(kFixedDt * gSpeed, gSlots);
    } else {
        UpdateFollowSlots(kFixedDt * gSpeed);
    }

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
        case 27: std::exit(0); break; // ESC
        case ' ': gPlaying = !gPlaying; break;
        case '+': case '=': gDroneCount = std::min(3000, gDroneCount + 1); ResizeArrays(); break;
        case '-': case '_': gDroneCount = std::max(1,   gDroneCount - 1); ResizeArrays(); break;
        case '1': gForm = CIRCLE; break;
        case '2': gForm = LINE;   break;
        case '3': gForm = WAVE;   break;
        case '4': gForm = HEART;  break;
        case 's': case 'S': gSpin = !gSpin; break;
        case ',': gSpeed = clampf(gSpeed - 0.05f, 0.1f, 3.0f); break;
        case '.': gSpeed = clampf(gSpeed + 0.05f, 0.1f, 3.0f); break;
        case 'e': case 'E': {
            float t = gTime;  // use current clock
            bool ok = ReloadAndApplyEmotions(t);
            const char* msg = ok
            ? "Emotions loaded from clap_weights.json and applied"
            : "Emotion load failed";
            gHudMsg = msg;
            gHudUntil = t + 1.5f;
            glutPostRedisplay();
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

    glutMainLoop();
    return 0;
}
