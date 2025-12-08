UNAME := $(shell uname)		#check your os

ifeq ($(UNAME), Darwin)
LDFLAGS = -framework OpenGL -framework GLUT	# macos
else
LDFLAGS = -lGL -lGLU -lglut	# Linux
endif

basicVisuals:
	$(CXX) -std=c++17 basicVisuals.cpp boids.cpp $(LDFLAGS) -o drones

trajectories:
	$(CXX) -std=c++17 simulateBoids.cpp boids.cpp $(LDFLAGS) -o traj
