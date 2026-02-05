UNAME := $(shell uname)		#check your os

ifeq ($(UNAME), Darwin)
LDFLAGS = -framework OpenGL -framework GLUT	# macos
else
LDFLAGS = -lGL -lGLU -lglut	# Linux
endif

OSQP_FLAGS = -Iosqp/include -Losqp/lib -losqp -lm

basicVisuals:
	$(CXX) -std=c++17 cbf_solver.cpp basicVisuals.cpp boids.cpp $(LDFLAGS) $(OSQP_FLAGS) -o drones

trajectories:
	$(CXX) -std=c++17 cbf_solver.cpp simulateBoids.cpp boids.cpp $(LDFLAGS) $(OSQP_FLAGS) -o traj
