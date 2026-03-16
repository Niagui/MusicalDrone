UNAME := $(shell uname)		#check your os

ifeq ($(UNAME), Darwin)
LDFLAGS = -framework OpenGL -framework GLUT	# macos
else
LDFLAGS = -lGL -lGLU -lglut	# Linux
endif

#WARNING DON'T TOUCH THESE FLAGS
OSQP_FLAGS = -Iosqp/osqp/include/public \
			 -Iosqp/osqp/build/include/public \
			 -Losqp/osqp/build/out -losqp -lm \
			 -Wl,-rpath,'$$ORIGIN/osqp/osqp/build/out'

basicVisuals:
	$(CXX) -std=c++17 cbf_solver.cpp basicVisuals.cpp boids.cpp $(LDFLAGS) $(OSQP_FLAGS) -o drones

trajectories:
	$(CXX) -std=c++17 cbf_solver.cpp simulateBoids.cpp boids.cpp $(LDFLAGS) $(OSQP_FLAGS) -o traj

osqp_test:
	$(CXX) -std=c++17 osqp_test.cpp $(OSQP_FLAGS) -o osqp_test