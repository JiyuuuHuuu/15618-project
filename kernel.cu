#include <stdio.h>
#include <curand_kernel.h>
#include "kernel.h"
#include "helper_math.h"
#include "helper.cu_inl"
#include "pattern.cu_inl"
#include "color.cu_inl"
#include "exclusiveScan.cu_inl"

/*
  particle array layout
  each cell is a struct firework;
  NOTE: each firework consists of multiple particle
  | |: empty, |*|: occupied
  | | |buffer_tail *|*|*|*|*|buffer_head| | |
*/

__constant__ uchar4 cuPalette[256];
__constant__ particle cuSchedule[MAX_SCHEDULE_NUM];

#ifndef FIREWORK_BUFFER_SIZE
__device__
void launchSchedule(particle *particles, int *schedule_idx, int *buffer_head, int *buffer_tail, float t) {
  // check and copy firework from schedule to work buffer for display
  firework *buffer = reinterpret_cast<firework *>(particles);
  const int blk_idx = blockIdx.y*gridDim.x + blockIdx.x;
  const int thd_idx = threadIdx.y*blockDim.x + threadIdx.x;
  const int idx = blk_idx*blockDim.x*blockDim.y + thd_idx; // get 1D index
  const int schedule_idx_local = *schedule_idx;
  const int buffer_head_local = *buffer_head;
  __syncthreads();
  if (idx + schedule_idx_local >= MAX_SCHEDULE_NUM ||
      idx + buffer_head_local >= MAX_PARTICLE_NUM) return;

  particle schedule_particle = cuSchedule[idx + schedule_idx_local];
  if (schedule_particle.t_0 >= 0.0f && schedule_particle.t_0 <= t) {
    // if (idx == 0) printf("in display: %d\n", *buffer_head - *buffer_tail);
    buffer[idx + buffer_head_local].pack[0] = schedule_particle;

    // update indices
    if (idx + schedule_idx_local + 1 == MAX_SCHEDULE_NUM ||
        cuSchedule[idx + schedule_idx_local + 1].t_0 > t) {
      *schedule_idx += idx + 1;
      *buffer_head += idx + 1;
    } else if (cuSchedule[idx + schedule_idx_local + 1].t_0 < 0) {
      // no schedule left to display
      *schedule_idx = MAX_SCHEDULE_NUM;
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
  const int firework_per_it = (gridDim.x*gridDim.y*blockDim.x*blockDim.y)/PARTICLE_NUM_PER_FIREWORK;
  const int particle_idx = idx % PARTICLE_NUM_PER_FIREWORK;

  unsigned int seed = idx*7 + (unsigned int)(t*100); // some random number that does not collide between threads
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
        if (p.y <= curr.explosion_height) curr.explosion_height = -1.0f; // mark explosion phase
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
        if (p_up.y <= upshoot.explosion_height) {
          // patternArray[upshoot.color](curr, p_up, t, particle_idx, seed);
          patterns(curr, p_up, t, particle_idx, seed, upshoot.color);
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
void fireworkKernel(uchar4 *d_out, int w, int h, particle *particles, tail *tails, float t, int *buffer_head, int *buffer_tail, int *schedule_idx) {
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  const int r = blockIdx.y*blockDim.y + threadIdx.y;
  const int idx = c + r*w; // 1D indexing
  launchSchedule(particles, schedule_idx, buffer_head, buffer_tail, t);
  __syncthreads();

  // display
  int tail_increment = 0;
  int freeup = 1;
  firework *buffer = reinterpret_cast<firework *>(particles);
  
  float2 pixel_pos = make_float2((float)c, (float)r);
  if (!((c >= w) || (r >= h))) {
    tail curr_tail = tails[idx];
    uchar4 pixel_color = make_uchar4(0, 0, 0, 255);
    for (int i = *buffer_tail; i < *buffer_head; i++) {
      firework *curr_firework = buffer + i;
      particle upshoot = curr_firework->pack[0];
      if (upshoot.t_0 < 0) {
        if (freeup) tail_increment++;
        continue;
      }
      freeup = 0;
      if (upshoot.explosion_height > 0) {
        // only upshooting particle need display
        upshoots(pixel_color, t, 0, upshoot, pixel_pos, curr_tail);
      } else {
        // firework after explosion
        for (int j = 1; j < PARTICLE_NUM_PER_FIREWORK; j++) {
          particle curr = curr_firework->pack[j];
          if (curr.t_0 < 0 || curr.t_0 > t) continue;
          colors(pixel_color, t, j, curr, pixel_pos, curr_tail);
        }
      }
    }
    tail_colors(pixel_color, t, 0, curr_tail);

    tails[idx] = curr_tail;
    d_out[idx] = pixel_color;
  }
  if (idx == 0) *buffer_tail += tail_increment;
  __syncthreads();

  updateParticle(particles, schedule_idx, buffer_head, buffer_tail, t);
}

void kernelLauncher(uchar4 *d_out, int w, int h, particle *particles, tail *tails, int *idx_holder, float t) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3((w + TX - 1)/TX, (h + TY - 1)/TY);
  fireworkKernel<<<gridSize, blockSize>>>(d_out, w, h, particles, tails, t, idx_holder, idx_holder+1, idx_holder+2);
  cudaError_t cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    printf("CUDA error after cudaDeviceSynchronize: %s\n", cudaGetErrorString(cudaStatus));
    exit(1);
  }
}
#else
__device__
void launchSchedule(particle *particles, int *schedule_idx, int *buffer_head, int *buffer_tail, float t) {
  // check and copy firework from schedule to work buffer for display
  firework *buffer = reinterpret_cast<firework *>(particles);
  const int blk_idx = blockIdx.y*gridDim.x + blockIdx.x;
  const int thd_idx = threadIdx.y*blockDim.x + threadIdx.x;
  const int idx = blk_idx*blockDim.x*blockDim.y + thd_idx; // get 1D index
  const int schedule_idx_local = *schedule_idx;
  const int buffer_head_local = *buffer_head;
  const int buffer_tail_local = *buffer_tail;
  __syncthreads();
  if (idx + schedule_idx_local >= MAX_SCHEDULE_NUM) return;

  particle schedule_particle = cuSchedule[idx + schedule_idx_local];
  if (schedule_particle.t_0 >= 0.0f && schedule_particle.t_0 <= t) {
    // if (idx == 0) printf("in display: %d\n", buffer_head_local - buffer_tail_local);
    int allocated = 0;
    if (idx + buffer_head_local - buffer_tail_local < FIREWORK_BUFFER_SIZE) {
      // buffer has extra space
      buffer[(idx + buffer_head_local)%FIREWORK_BUFFER_SIZE].pack[0] = schedule_particle;
      allocated = 1;
    } else {
      printf("!!!Warning: buffer full, skipping this firework\n");
    }

    // update indices
    if (idx + schedule_idx_local + 1 == MAX_SCHEDULE_NUM ||
        cuSchedule[idx + schedule_idx_local + 1].t_0 > t) {
      *schedule_idx += idx + 1;
      if (allocated)
        *buffer_head += idx + 1;
    } else if (cuSchedule[idx + schedule_idx_local + 1].t_0 < 0) {
      // no schedule left to display
      *schedule_idx = MAX_SCHEDULE_NUM;
      if (allocated)
        *buffer_head += idx + 1;
    }
  }
}

__device__
void launchScheduleRandom(particle *particles, int *buffer_head, int *buffer_tail, float t) {
  // check and copy firework from schedule to work buffer for display
  firework *buffer = reinterpret_cast<firework *>(particles);
  const int blk_idx = blockIdx.y*gridDim.x + blockIdx.x;
  const int thd_idx = threadIdx.y*blockDim.x + threadIdx.x;
  const int idx = blk_idx*blockDim.x*blockDim.y + thd_idx; // get 1D index
  const int buffer_head_local = *buffer_head;
  const int buffer_tail_local = *buffer_tail;
  __syncthreads();
  unsigned int seed = idx*7 + (unsigned int)(t*100); // some random number that does not collide between threads

  // if (idx == 0) printf("in display: %d\n", buffer_head_local - buffer_tail_local);
  if (idx + buffer_head_local - buffer_tail_local < FIREWORK_BUFFER_SIZE) {
    // buffer has extra space
    particle new_particle;

    new_particle.p_0 = make_float2(random_float((float)W * 0.1f, (float)W * 0.9f, seed),
                                   random_float((float)H * 0.85f, (float)H * 0.95f, seed));
    new_particle.v_0 = make_float2(random_float(-10.0f, 10.0f, seed),
                                   random_float(-100.0f, -80.0f, seed));
    new_particle.a = make_float2(0.0f, G);
    new_particle.r = random_float(8.0f, 10.0f, seed);
    new_particle.t_0 = t + random_float(0.5f, 2.0f, seed) + (float)(idx)/3.0f;
    float t_peak = -new_particle.v_0.y / G;
    float max_height = currP(new_particle.p_0, new_particle.v_0, new_particle.a, t_peak).y;
    new_particle.explosion_height = random_float(fmaxf(max_height * 1.2f, (float)H * 0.1f), fmaxf(max_height * 2.0f, (float)H * 0.7f), seed);
    new_particle.color = random_char(0, 6, seed);
    new_particle.tail = 1;
    
    buffer[(idx + buffer_head_local)%FIREWORK_BUFFER_SIZE].pack[0] = new_particle;
  }
  if (idx == 0) *buffer_head = buffer_tail_local + FIREWORK_BUFFER_SIZE;
}

__device__
void updateParticle(particle *particles, int *schedule_idx, int *buffer_head, int *buffer_tail, float t) {
  firework *buffer = reinterpret_cast<firework *>(particles);
  const int blk_idx = blockIdx.y*gridDim.x + blockIdx.x;
  const int thd_idx = threadIdx.y*blockDim.x + threadIdx.x;
  const int idx = blk_idx*blockDim.x*blockDim.y + thd_idx; // get 1D index
  const int particle_idx = idx % PARTICLE_NUM_PER_FIREWORK;

  unsigned int seed = idx*7 + (unsigned int)(t*100); // some random number that does not collide between threads
  int buffer_idx = idx/PARTICLE_NUM_PER_FIREWORK + *buffer_tail;
  if (buffer_idx >= *buffer_head) return;
  firework curr_firework = buffer[buffer_idx % FIREWORK_BUFFER_SIZE];
  particle upshoot = curr_firework.pack[0];
  particle curr = curr_firework.pack[particle_idx];
  if (upshoot.t_0 < 0) return;
  if (particle_idx == 0) {
    // upshooting particle
    if (curr.explosion_height > 0) {
      float2 p = currP(curr.p_0, curr.v_0, curr.a, t - curr.t_0);
      if (p.y <= curr.explosion_height) curr.explosion_height = -1.0f; // mark explosion phase
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
      if (p_up.y <= upshoot.explosion_height) {
        // patternArray[upshoot.color](curr, p_up, t, particle_idx, seed);
        patterns(curr, p_up, t, particle_idx, seed, upshoot.color);
      }
    } else {
      // check particle end
      if (t - curr.t_0 >= curr.explosion_height)
        curr.t_0 = -1.0f;
    }
  }
  __syncthreads();
  buffer[buffer_idx % FIREWORK_BUFFER_SIZE].pack[particle_idx] = curr;
}

#ifndef BATCH
__global__
void fireworkKernel(uchar4 *d_out, int w, int h, particle *particles, tail *tails, float t, int *buffer_head, int *buffer_tail, int *schedule_idx, int rand_generate) {
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  const int r = blockIdx.y*blockDim.y + threadIdx.y;
  const int idx = c + r*w; // 1D indexing
  if (rand_generate)
    launchScheduleRandom(particles, buffer_head, buffer_tail, t);
  else
    launchSchedule(particles, schedule_idx, buffer_head, buffer_tail, t);
  __syncthreads();

  // display
  int tail_increment = 0;
  int freeup = 1;
  firework *buffer = reinterpret_cast<firework *>(particles);
  
  float2 pixel_pos = make_float2((float)c, (float)r);
  if (!((c >= w) || (r >= h))) {
    tail curr_tail = tails[idx];
    uchar4 pixel_color = make_uchar4(0, 0, 0, 255);
    for (int i = *buffer_tail; i < *buffer_head; i++) {
      firework *curr_firework = buffer + (i % FIREWORK_BUFFER_SIZE);
      particle upshoot = curr_firework->pack[0];
      if (upshoot.t_0 < 0) {
        if (freeup) tail_increment++;
        continue;
      }
      freeup = 0;
      if (upshoot.explosion_height > 0) {
        // only upshooting particle need display
        if (upshoot.t_0 <= t)
          upshoots(pixel_color, t, 0, upshoot, pixel_pos, curr_tail);
      } else {
        // firework after explosion
        for (int j = 1; j < PARTICLE_NUM_PER_FIREWORK; j++) {
          particle curr = curr_firework->pack[j];
          if (curr.t_0 < 0 || curr.t_0 > t) continue;
          colors(pixel_color, t, j, curr, pixel_pos, curr_tail);
        }
      }
    }
    tail_colors(pixel_color, t, 0, curr_tail);

    tails[idx] = curr_tail;
    d_out[idx] = pixel_color;
  }
  if (idx == 0) *buffer_tail += tail_increment;
  __syncthreads();

  updateParticle(particles, schedule_idx, buffer_head, buffer_tail, t);
}
#else
__global__
void fireworkKernel(uchar4 *d_out, int w, int h, particle *particles, tail *tails, float t, int *buffer_head, int *buffer_tail, int *schedule_idx, int rand_generate) {
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  const int r = blockIdx.y*blockDim.y + threadIdx.y;
  const int idx = c + r*w; // 1D indexing

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  const float boxL = static_cast<float>(blockIdx.x * blockDim.x);
  const float boxR = static_cast<float>((blockIdx.x + 1) * blockDim.x);
  const float boxB = static_cast<float>(blockIdx.y * blockDim.y);
  const float boxT = static_cast<float>((blockIdx.y + 1) * blockDim.y);

  // if (idx == 800) printf("%f %f %f %f\n", boxL, boxR, boxB, boxT);

  __shared__ uint prefixSumInput[BLK_SIZE];
  __shared__ uint prefixSumOutput[BLK_SIZE];
  __shared__ uint prefixSumScratch[2 * BLK_SIZE];
  // __shared__ float2 sharedPBuffer[BLK_SIZE];
  // __shared__ float2 sharedVBuffer[BLK_SIZE];
  __shared__ unsigned char sharedColor[BLK_SIZE];
  __shared__ uint existPIdx[BLK_SIZE];

  if (rand_generate)
    launchScheduleRandom(particles, buffer_head, buffer_tail, t);
  else
    launchSchedule(particles, schedule_idx, buffer_head, buffer_tail, t);
  __syncthreads();

  // display
  // int tail_increment = 0;
  // int freeup = 1;
  float2 pixel_pos = make_float2((float)c, (float)r);
  firework *buffer = reinterpret_cast<firework *>(particles);
  int buffer_head_local = *buffer_head;
  int buffer_tail_local = *buffer_tail;
  int ttl_particle_cnt = (buffer_head_local - buffer_tail_local) * PARTICLE_NUM_PER_FIREWORK;

  tail curr_tail;
  if (!((c >= w) || (r >= h))) curr_tail = tails[idx];
  uchar4 pixel_color = make_uchar4(0, 0, 0, 255);
  for (int batch_offset = 0; batch_offset < ttl_particle_cnt; batch_offset += BLK_SIZE) {
    int particle_idx = (tid + batch_offset + buffer_tail_local * PARTICLE_NUM_PER_FIREWORK) %
                       (FIREWORK_BUFFER_SIZE * PARTICLE_NUM_PER_FIREWORK);
    if (tid + batch_offset < ttl_particle_cnt) {
      int firework_idx = particle_idx / PARTICLE_NUM_PER_FIREWORK;
      int pack_idx = particle_idx % PARTICLE_NUM_PER_FIREWORK;
      int upshoot_idx = tid - tid % PARTICLE_NUM_PER_FIREWORK;

      particle curr = buffer[firework_idx].pack[pack_idx];
      particle upshoot = buffer[firework_idx].pack[upshoot_idx];
      // sharedPBuffer[tid] = curr.p_0;
      // sharedVBuffer[tid] = curr.v_0;
      // sharedABuffer[tid] = curr.a;

      float2 p_t = currP(curr.p_0, curr.v_0, curr.a, t);

      int to_include = 0;
      if (circleInBox(p_t.x, p_t.y, curr.r, boxL, boxR, boxT, boxB) &&
          (curr.t_0 >= 0) && (curr.t_0 <= t))
        to_include = 1;
      if (tid == upshoot_idx) {
        curr.color = 200;
        if (curr.explosion_height <= 0) to_include = 0;
      } else {
        if (upshoot.explosion_height > 0) to_include = 0;
      }
      sharedColor[tid] = curr.color;
      prefixSumInput[tid] = to_include;
    }
    __syncthreads();

    sharedMemExclusiveScan(tid, prefixSumInput, prefixSumOutput, prefixSumScratch, BLK_SIZE);
    __syncthreads();

    int particle_cnt = prefixSumOutput[BLK_SIZE-1] + prefixSumInput[BLK_SIZE-1];
    if (batch_offset + BLK_SIZE >= ttl_particle_cnt)
      particle_cnt = prefixSumOutput[ttl_particle_cnt - batch_offset];

    // if (particle_cnt != 0) printf("%d\n", particle_cnt);

    if (prefixSumInput[tid] && tid + batch_offset < ttl_particle_cnt)
      existPIdx[prefixSumOutput[tid]] = tid;
    __syncthreads();

    if (!((c >= w) || (r >= h))) {
      for (int i = 0; i < particle_cnt; i++) {
        int particle_idx = (existPIdx[i] + batch_offset + buffer_tail_local * PARTICLE_NUM_PER_FIREWORK) %
                           (FIREWORK_BUFFER_SIZE * PARTICLE_NUM_PER_FIREWORK);
        int firework_idx = particle_idx / PARTICLE_NUM_PER_FIREWORK;
        int pack_idx = particle_idx % PARTICLE_NUM_PER_FIREWORK;
        particle curr = buffer[firework_idx].pack[pack_idx];
        curr.color = sharedColor[existPIdx[i]];
        colors(pixel_color, t, 0, curr, pixel_pos, curr_tail);
      }
    }
    __syncthreads();
  }

  if (!((c >= w) || (r >= h))) {
    tail_colors(pixel_color, t, 0, curr_tail);
    tails[idx] = curr_tail;
    d_out[idx] = pixel_color;
  }

  updateParticle(particles, schedule_idx, buffer_head, buffer_tail, t);
}
#endif

void kernelLauncher(uchar4 *d_out, int w, int h, particle *particles, tail *tails, int *idx_holder, float t, int rand_generate) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3((w + TX - 1)/TX, (h + TY - 1)/TY);
  fireworkKernel<<<gridSize, blockSize>>>(d_out, w, h, particles, tails, t, idx_holder, idx_holder+1, idx_holder+2, rand_generate);
  cudaError_t cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    printf("CUDA error after cudaDeviceSynchronize: %s\n", cudaGetErrorString(cudaStatus));
    exit(1);
  }
}
#endif

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