
EXECUTABLE := firework-pixel
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart -L./glew/lib/ -lGLEW

all: $(EXECUTABLE)

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc
INCLUDES=-I./glew/include
LIBS += GL glut cudart GLU

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/firework-pixel.o

.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS) *.ppm

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJS): firework-pixel.cu dirs build-glew
		$(NVCC) $< $(NVCCFLAGS) $(INCLUDES) -c -o $@

build-glew:
		git submodule update --init --recursive
		cd glew/auto && make
		cd glew && make