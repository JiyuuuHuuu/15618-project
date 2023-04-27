#include <stdio.h>
#include "kernel.h"
#include "helper.cu_inl"
#define TX 32
#define TY 32

/*
  particle array layout
  each cell is a struct firework;
  NOTE: each firework consists of multiple particle
  | |: empty, |*|: occupied
  | | |buffer_tail *|*|*|*|*|buffer_head| | |
*/

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

__device__
void launchSchedule(particle *particles, int *schedule_idx, int *buffer_head, float t) {
  // check and copy firework from schedule to work buffer for display
  firework *buffer = reinterpret_cast<firework *>(particles);
  const int blk_idx = blockIdx.y*gridDim.x + blockIdx.x;
  const int thd_idx = threadIdx.y*blockDim.x + threadIdx.x;
  const int idx = blk_idx*blockDim.x*blockDim.y + thd_idx; // get 1D index
  if (idx + *schedule_idx >= MAX_SCHEDULE_NUM) return;
  if (idx + *buffer_head >= MAX_PARTICLE_NUM) return;

  particle schedule_particle = cuSchedule[idx + *schedule_idx];
  if (schedule_particle.t_0 >= 0.0f && schedule_particle.t_0 <= t) {
    buffer[idx + *buffer_head].pack[0] = schedule_particle;

    // update indices
    if (idx + *schedule_idx + 1 == MAX_SCHEDULE_NUM) {
      *schedule_idx += idx + 1;
      *buffer_head += idx + 1;
    } else if (cuSchedule[idx + *schedule_idx + 1].t_0 > t) {
      *schedule_idx += idx + 1;
      *buffer_head += idx + 1;
    }
  }
}

__device__
void updateParticle(particle *particles, int *schedule_idx, int *buffer_head, int *buffer_tail, float t) {
  firework *buffer = reinterpret_cast<firework *>(particles);
  const int blk_idx = blockIdx.y*gridDim.x + blockIdx.x;
  const int thd_idx = threadIdx.y*blockDim.x + threadIdx.x;
  const int idx = blk_idx*blockDim.x*blockDim.y + thd_idx; // get 1D index
  const int firework_per_it = gridDim.x*gridDim.y*blockDim.x*blockDim.y/PARTICLE_NUM_PER_FIREWORK;
  const int particle_idx = idx % PARTICLE_NUM_PER_FIREWORK;

  int buffer_idx = idx/PARTICLE_NUM_PER_FIREWORK + *buffer_tail;
  for (int i = buffer_idx; i < *buffer_head; i += firework_per_it) {
    firework curr_firework = buffer[i];
    particle upshoot = curr_firework.pack[0];
    particle curr = curr_firework.pack[particle_idx];
    if (upshoot.t_0 < 0) continue;
    if (particle_idx == 0) {
      // upshooting particle
      if (curr.explosion_height > 0) {
        float2 p = currP(curr.p_0, curr.v_0, curr.a, t - curr.t_0);
        if (p.y >= curr.explosion_height) curr.explosion_height = -1.0f; // mark explosion phase
      } else {
        int isEnd = 1;
        for (int j = 1; j < PARTICLE_NUM_PER_FIREWORK; j++) {
          if (curr_firework.pack[j].t_0 >= 0) {
            isEnd = 0;
            break;
          }
        }
        if (isEnd) curr.t_0 = -1.0f; // mark this firework as evicted
      }
    } else {
      // child particle
      if (upshoot.explosion_height > 0) {
        float2 p_up = currP(upshoot.p_0, upshoot.v_0, upshoot.a, t - upshoot.t_0);
        if (p_up.y >= upshoot.explosion_height) {
          // TODO: generate child particle, determine velocity and acceleration
        }
      } else {
        // check particle end
        if (t - curr.t_0 >= curr.explosion_height)
          curr.t_0 = -1.0f;
      }
    }
    __syncthreads();
    buffer[i].pack[particle_idx] = curr;
  }
}

__global__
void fireworkKernel(uchar4 *d_out, int w, int h, particle *particles, float t, int *buffer_head, int *buffer_tail, int *schedule_idx,) {
  launchSchedule(particles, schedule_idx, buffer_head, t);
  __syncthreads();

  // display
  firework *buffer = reinterpret_cast<firework *>(particles);
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  const int r = blockIdx.y*blockDim.y + threadIdx.y;
  
  if ((c >= w) || (r >= h)) {
    const int idx = c + r*w; // 1D indexing
    uchar4 pixel_color = make_uchar4(0, 0, 0, 255);
    for (int i = *buffer_tail; i < *buffer_head; i++) {
      firework *curr_firework = buffer + i;
      particle upshoot = curr_firework->pack[0];
      if (upshoot.explosion_height > 0) {
        // only upshooting particle need display
        float2 p = currP(upshoot.p_0, upshoot.v_0, upshoot.a, t - upshoot.t_0);
        if (isWithinDistance(upshoot.p_0, make_float2((float)c, (float)r), upshoot.r)) {
          pixel_color = cuPalette[upshoot.color]; // TODO: support particle overlap
        }
      } else {
        // firework after explosion
        for (int j = 1; j < PARTICLE_NUM_PER_FIREWORK; j++) {
          particle curr = curr_firework->pack[j];
          if (p.t_0 < 0) continue;
          float2 p = currP(curr.p_0, curr.v_0, curr.a, t - curr.t_0);
          if (isWithinDistance(curr.p_0, make_float2((float)c, (float)r), curr.r)) {
            pixel_color = cuPalette[curr.color]; // TODO: support particle overlap
          }
        }
      }
    }
    d_out[idx] = pixel_color;
  }
  __syncthreads();

  // update particles
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