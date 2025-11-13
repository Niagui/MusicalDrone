UNAME := $(shell uname)		#check your os

ifeq ($(UNAME), Darwin)
LDFLAGS = -framework OpenGL -framework GLUT	# macos
else
LDFLAGS = -lGL -lGLU -lglut	# Linux
endif

basicVisuals:
	$(CXX) -std=c++14 basicVisuals.cpp boids.cpp $(LDFLAGS) -o drones