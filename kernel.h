#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

#define W 800
#define H 400
#define MAX_PARTICLE_NUM 65536
#define MAX_SCHEDULE_NUM 256
#define TITLE_STRING "firework"
#define G 9.8f
#define RATE 0.75f
#define PI 3.1415926f
#define SHOW_FPS 1

struct uchar4;
struct int2;

const int PARTICLE_NUM_PER_FIREWORK = MAX_PARTICLE_NUM/MAX_SCHEDULE_NUM; // total number of threads must be multiple of this number

// explostion height is used as delta_t for child particles
struct particle {
  float2 p_0;
  float2 v_0;
  float2 a;
  float r;
  float t_0;
  float explosion_height;
  unsigned char color;
  unsigned char tail;
};

struct tail {
  float t_0;
  unsigned char color;
};

struct firework {
  particle pack[PARTICLE_NUM_PER_FIREWORK];
};

void kernelLauncher(uchar4 *d_out, int w, int h, particle *particles, tail *tails, int *idx_holder, float t);

void makePalette(void);
void setUpSchedule(particle *particles_host);

#endif