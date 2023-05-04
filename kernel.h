#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

#define TX 32
#define TY 32
#define BLK_SIZE TX * TY
#define SCAN_BLOCK_DIM   BLK_SIZE
#define W 800
#define H 400
#define MAX_PARTICLE_NUM 65536
#define FIREWORK_BUFFER_SIZE 32
#define MAX_SCHEDULE_NUM 256
#define TITLE_STRING "firework"
#define G 9.8f
#define RATE 0.8f
#define PI 3.1415926f
#define SHOW_FPS 1

struct uchar4;
struct int2;

#ifndef FIREWORK_BUFFER_SIZE
const int PARTICLE_NUM_PER_FIREWORK = MAX_PARTICLE_NUM/MAX_SCHEDULE_NUM; // total number of threads must be multiple of this number
#else
const int PARTICLE_NUM_PER_FIREWORK = 256; // total number of threads must be multiple of this number
#endif

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

#ifndef FIREWORK_BUFFER_SIZE
void kernelLauncher(uchar4 *d_out, int w, int h, particle *particles, tail *tails, int *idx_holder, float t);
#else
void kernelLauncher(uchar4 *d_out, int w, int h, particle *particles, tail *tails, int *idx_holder, float t, int rand_generate);
#endif

void makePalette(void);
void setUpSchedule(particle *particles_host);

#endif