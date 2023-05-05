#include <cuda_runtime.h>
#include <stdio.h>
#include <stdio.h>
#include <time.h>
#include "helper_math.h"
#include "seq.h"

float2 currP_host(float2 p_0, float2 v_0, float2 a, float t) {
  return p_0 + v_0*t + 1.0f/2.0f*a*t*t;
}


void sleepNanoseconds(long nanoseconds) {
  struct timespec sleepTime;
  sleepTime.tv_sec = 0;
  sleepTime.tv_nsec = nanoseconds;
  
  nanosleep(&sleepTime, NULL);
}

void upshoots_host(uchar4 &pixel_color, const float t, int offset, const particle &curr, const float2 &pixel_pos, tail &curr_tail) {sleepNanoseconds(200);}

void patterns_host(particle &curr, const float2 p_0, const float t, int offset, unsigned int seed, int idx) {
  curr.t_0 = t + 0.3f;
  curr.explosion_height = 8.0f;
  sleepNanoseconds(2000);
}

void colors_host(uchar4 &pixel_color, const float t, int offset, const particle &curr, const float2 &pixel_pos, tail &curr_tail) {sleepNanoseconds(200);}

void tail_colors_host(uchar4 &pixel_color, const float t, int offset, const tail &curr_tail) {sleepNanoseconds(200);}

void launchSchedule_host(particle *particles, int *schedule_idx, int *buffer_head, int *buffer_tail, float t, const particle *schedule) {
  // check and copy firework from schedule to work buffer for display
  firework *buffer = reinterpret_cast<firework *>(particles);
  while(true) {
    if (*schedule_idx > MAX_SCHEDULE_NUM || *buffer_head >= MAX_PARTICLE_NUM) return;
    particle schedule_particle = schedule[*schedule_idx];
    if (schedule_particle.t_0 >= 0.0f && schedule_particle.t_0 <= t) {
      printf("launch schedule %d\n", *schedule_idx);
      buffer[*buffer_head].pack[0] = schedule_particle;
      (*schedule_idx)++;
      (*buffer_head)++;
      printf("%d %d\n", *buffer_tail, *buffer_head);
    } else if (schedule_particle.t_0 > t){
      return;
    }
  }
}

void updateParticle_host(particle *particles, int *schedule_idx, int *buffer_head, int *buffer_tail, float t) {
  firework *buffer = reinterpret_cast<firework *>(particles);

  for (int idx = *buffer_tail; idx < *buffer_head; idx++) {
    firework curr_firework = buffer[idx];
    particle upshoot = curr_firework.pack[0];
    if (upshoot.t_0 < 0) continue;
    for (int particle_idx = 0; particle_idx < PARTICLE_NUM_PER_FIREWORK; particle_idx++) {
      particle curr = curr_firework.pack[particle_idx];
      if (particle_idx == 0) {
        // upshooting particle
        if (curr.explosion_height > 0) {
          float2 p = currP_host(curr.p_0, curr.v_0, curr.a, t - curr.t_0);
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
          float2 p_up = currP_host(upshoot.p_0, upshoot.v_0, upshoot.a, t - upshoot.t_0);
          if (p_up.y <= upshoot.explosion_height) {
            patterns_host(curr, p_up, t, particle_idx, 0, upshoot.color);
          }
        } else {
          // check particle end
          if (t - curr.t_0 >= curr.explosion_height)
            curr.t_0 = -1.0f;
        }
      }
      buffer[idx].pack[particle_idx] = curr;
    }
  }
}

void fireworkHost(uchar4 *d_out, int w, int h, particle *particles, tail *tails, float t, int *buffer_head, int *buffer_tail, int *schedule_idx, const particle *schedule) {
  launchSchedule_host(particles, schedule_idx, buffer_head, buffer_tail, t, schedule);

  // display
  int freeup = 1;
  firework *buffer = reinterpret_cast<firework *>(particles);
  for (int r = 0; r < h; r++) {
    for (int c = 0; c < w; c++) {
      int tail_increment = 0;
      int idx = c + r*w;
      float2 pixel_pos = make_float2((float)c, (float)r);
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
          upshoots_host(pixel_color, t, 0, upshoot, pixel_pos, curr_tail);
        } else {
          // firework after explosion
          for (int j = 1; j < PARTICLE_NUM_PER_FIREWORK; j++) {
            particle curr = curr_firework->pack[j];
            if (curr.t_0 < 0 || curr.t_0 > t) continue;
            colors_host(pixel_color, t, j, curr, pixel_pos, curr_tail);
          }
        }
      }
      tail_colors_host(pixel_color, t, 0, curr_tail);
      tails[idx] = curr_tail;
      d_out[idx] = pixel_color;
      *buffer_tail += tail_increment;
      if (*buffer_tail >= *buffer_head) goto LOOP_END;
    }
  }
LOOP_END:
  updateParticle_host(particles, schedule_idx, buffer_head, buffer_tail, t);
}

void seqLauncher(uchar4 *d_out, int w, int h, particle *particles, tail *tails, int *idx_holder, float t, const particle *schedule){
  fireworkHost(d_out, w, h, particles, tails, t, idx_holder, idx_holder+1, idx_holder+2, schedule);
}