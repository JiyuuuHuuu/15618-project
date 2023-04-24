#include <stdio.h>
#include "kernel.h"
#include "helper.cu_inl"
#define TX 32
#define TY 32

__constant__ uchar4 cuPalette[256];
__constant__ particle cuSchedule[MAX_SCHEDULE_NUM];

__global__
void distanceKernel(uchar4 *d_out, int w, int h, int2 pos, float t) {
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  const int r = blockIdx.y*blockDim.y + threadIdx.y;
  if ((c >= w) || (r >= h)) return; // Check if within image bounds
  const int i = c + r*w; // 1D indexing
  const int dist = sqrtf((c - pos.x)*(c - pos.x) +
                         (r - pos.y)*(r - pos.y));
  const unsigned char intensity = clip(255 - dist - abs((int)(t*80)%510 - 255));
  d_out[i].x = intensity;
  d_out[i].y = intensity;
  d_out[i].z = 0;
  d_out[i].w = 255;
}

__global__
void fireworkKernel(uchar4 *d_out, int w, int h, particle *particles, float t) {
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  const int r = blockIdx.y*blockDim.y + threadIdx.y;
  if ((c >= w) || (r >= h)) return; // Check if within image bounds
  const int idx = c + r*w; // 1D indexing

  uchar4 pixel_color = make_uchar4(0, 0, 0, 255);
  for (int i = 0; i < MAX_SCHEDULE_NUM; i++) {
    if (cuSchedule[i].t_0 >= 0.0f && cuSchedule[i].t_0 <= t) {
      float2 p_0 = cuSchedule[i].p_0;
      float radius = cuSchedule[i].r;
      if (isWithinDistance(p_0, make_float2((float)c, (float)r), radius)) {
        pixel_color = cuPalette[cuSchedule[i].color];
      }
    }
  }
  d_out[idx] = pixel_color;
}

void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos, float t) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3((w + TX - 1)/TX, (h + TY - 1)/TY);
  distanceKernel<<<gridSize, blockSize>>>(d_out, w, h, pos, t);
  cudaDeviceSynchronize();
}

void kernelLauncher(uchar4 *d_out, int w, int h, particle *particles, float t) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3((w + TX - 1)/TX, (h + TY - 1)/TY);
  fireworkKernel<<<gridSize, blockSize>>>(d_out, w, h, particles, t);
  cudaDeviceSynchronize();
}

void makePalette(void) {
  uchar4 palette[256];

  palette[0] = make_uchar4(255, 0, 0, 255);
  palette[1] = make_uchar4(0, 255, 0, 255);
  palette[2] = make_uchar4(0, 0, 255, 255);
  palette[3] = make_uchar4(255, 255, 0, 255);
  palette[4] = make_uchar4(255, 0, 255, 255);
  palette[5] = make_uchar4(0, 255, 255, 255);

  cudaMemcpyToSymbol(cuPalette, palette, sizeof(uchar4) * 256);
}

void setUpSchedule(particle *particles_host) {
  cudaMemcpyToSymbol(cuSchedule, particles_host, MAX_SCHEDULE_NUM*sizeof(particle));
}