#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include "constants.h"

#ifndef FIREWORK_BUFFER_SIZE
void kernelLauncher(uchar4 *d_out, int w, int h, particle *particles, tail *tails, int *idx_holder, float t);
#else
void kernelLauncher(uchar4 *d_out, int w, int h, particle *particles, tail *tails, int *idx_holder, float t, int rand_generate);
#endif

void makePalette(void);
void setUpSchedule(particle *particles_host);

#endif