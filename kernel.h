#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

#define W 1200
#define H 600
#define TITLE_STRING "firework"

struct uchar4;
struct int2;

void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos);

#endif