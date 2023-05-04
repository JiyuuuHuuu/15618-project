
EXECUTABLE := firework
BENCHMARK := benchmark
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart -L./glew/lib/ -lGLEW
B_LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart

all: $(EXECUTABLE) $(BENCHMARK)

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
B_LIBS=cudart

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

B_LDLIBS := $(addprefix -l, $(B_LIBS))

NVCC=nvcc

OBJS=$(OBJDIR)/firework.o $(OBJDIR)/kernel.o
OBJS_BENCH=$(OBJDIR)/benchmark.o $(OBJDIR)/kernel.o

.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(BENCHMARK) $(LOGS) *.ppm

$(BENCHMARK): dirs $(OBJS_BENCH)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS_BENCH) $(B_LDFLAGS) $(B_LDLIBS) $(LDFRAMEWORKS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/benchmark.o: benchmark.cpp kernel.h
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

$(OBJDIR)/firework.o: main.cpp kernel.h build-glew
		$(NVCC) $< $(NVCCFLAGS) $(INCLUDES) -c -o $@

$(OBJDIR)/kernel.o: kernel.cu helper.cu_inl pattern.cu_inl color.cu_inl kernel.h helper_math.h
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

build-glew:
		git submodule update --init --recursive
		cd glew/auto && make
		cd glew && make