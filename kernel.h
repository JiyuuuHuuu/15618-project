#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

#define W 1200
#define H 600
#define MAX_PARTICLE_NUM 8192
#define MAX_SCHEDULE_NUM 512
#define TITLE_STRING "firework"
#define G 9.8f

struct uchar4;
struct int2;

struct particle {
  float2 p_0;
  float2 v_0;
  float2 a;
  float r;
  float t_0;
  float explosion_height;
  unsigned char color;
};

void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos, float t);

void kernelLauncher(uchar4 *d_out, int w, int h, particle *particles, float t);

void makePalette(void);
void setUpSchedule(particle *particles_host);

#endif