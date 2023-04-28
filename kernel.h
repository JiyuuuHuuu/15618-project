#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

#define W 1200
#define H 600
#define MAX_PARTICLE_NUM 32768
#define MAX_SCHEDULE_NUM 512
#define TITLE_STRING "firework"
#define G 9.8f

struct uchar4;
struct int2;

const int PARTICLE_NUM_PER_FIREWORK = MAX_PARTICLE_NUM/MAX_SCHEDULE_NUM; // total number of threads must be multiple of this number

// explostion height is used as delta_t child particles
struct particle {
  float2 p_0;
  float2 v_0;
  float2 a;
  float r;
  float t_0;
  float explosion_height;
  unsigned char color;
};

struct firework {
  particle pack[PARTICLE_NUM_PER_FIREWORK];
};

void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos, float t);

void kernelLauncher(uchar4 *d_out, int w, int h, particle *particles, int *idx_holder, float t);

void makePalette(void);
void setUpSchedule(particle *particles_host);

#endif