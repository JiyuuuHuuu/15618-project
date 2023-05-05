#include "kernel.h"
#include "seq.h"
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

#define DURATION 30

float t = 0.0f;
particle *particles_device;
tail *tails_device;
int *idx_holder_device;
uchar4 *d_out;
float framesPerSecond = 0.0f;
long long int lastTime = 0, currentTime, startTime;
int rand_generate = 0;

long long int timeSinceEpochMillisec() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

void CalculateFrameRate()
{
  if (!SHOW_FPS) return;
  currentTime = timeSinceEpochMillisec();
  ++framesPerSecond;
  if (currentTime - lastTime >= 1000) {
    lastTime = currentTime;
    printf("FPS: %d\n", (int)framesPerSecond);
    framesPerSecond = 0;
  }
}

class CmdParser {
  public:
    CmdParser(int argc, char **argv) {
      for (int i = 1; i < argc; ++i) {
        this->tokens.push_back(std::string(argv[i]));
      }
    }

    std::string GetOption(const std::string &option) const {
      std::vector<std::string>::const_iterator iter;
      iter = std::find(this->tokens.begin(), this->tokens.end(), option);
      if (iter != this->tokens.end() && ++iter != this->tokens.end()){
        return *iter;
      }
      return "";
    }
  private:
    std::vector<std::string> tokens;
};

bool parseFile(const std::string &file, particle *particles) {
  std::ifstream inFile;
  inFile.open(file);
  if (!inFile) {
    return false;
  }
  std::string line;
  int idx = 0;
  while (std::getline(inFile, line)) {
    std::stringstream sstream(line);
    std::string str;
    const char delim = ',';
    std::getline(sstream, str, delim);
    particles[idx].t_0 = (float)atof(str.c_str());
    std::getline(sstream, str, delim);
    float p0_x = (float)atof(str.c_str());
    std::getline(sstream, str, delim);
    float p0_y = (float)atof(str.c_str());
    particles[idx].p_0 = make_float2(p0_x, p0_y);
    std::getline(sstream, str, delim);
    float v0_x = (float)atof(str.c_str());
    std::getline(sstream, str, delim);
    float v0_y = (float)atof(str.c_str());
    particles[idx].v_0 = make_float2(v0_x, v0_y);
    particles[idx].a = make_float2(0.0f, G);
    std::getline(sstream, str, delim);
    particles[idx].r = (float)atof(str.c_str());
    std::getline(sstream, str, delim);
    particles[idx].explosion_height = (float)atof(str.c_str());
    std::getline(sstream, str, '\n');
    particles[idx].color = (int)atoi(str.c_str());
    idx++;
    if (idx > MAX_SCHEDULE_NUM) {
      // truncate the extra
      printf("Warning: too many input schedules, truncating tails\n");
      break;
    }
  }
  inFile.close();
  return true;
}

// Return true if valid.
bool check_schedule(particle *particles) {
  bool schedule_invalid = false;
  // Force t_0.
  for (int i = 1; i < MAX_SCHEDULE_NUM; ++i) {
    if (particles[i].t_0 == -1.0f) {
      break;
    }
    if (particles[i-1].t_0 > particles[i].t_0) {
      schedule_invalid = true;
    }
  }
  return !schedule_invalid;
}


int main(int argc, char** argv) {
  makePalette();

  // initiate CUDA mem
  particle *schedule_host;
  schedule_host = (particle *)malloc(MAX_SCHEDULE_NUM*sizeof(particle));

  for (int i = 0; i < MAX_SCHEDULE_NUM; i++) {
    schedule_host[i].t_0 = -1.0f;
    schedule_host[i].p_0 = make_float2(0.0f, 0.0f);
    schedule_host[i].v_0 = make_float2(0.0f, 0.0f);
    schedule_host[i].a = make_float2(0.0f, 0.0f);
    schedule_host[i].r = 0.0f;
    schedule_host[i].explosion_height = 0.0f;
    schedule_host[i].color = 0;
  }

  // Parse command line arguments.
  bool parse_success = false;
  CmdParser cmd_parser(argc, argv);
  std::string file = cmd_parser.GetOption("-f");
  // Parse file.
#ifndef FIREWORK_BUFFER_SIZE
  if (!file.empty()) {
    parseFile(file, schedule_host);
  } else {
    parseFile("./input/s000.csv", schedule_host);
  }
#else
  if (!file.empty()) {
    parseFile(file, schedule_host);
  } else {
    rand_generate = 1;
  }
#endif
  // Validate schedule.
  if (!check_schedule(schedule_host)) {
    printf("Error: invalid schedule\n");
    return 1;
  }

  for (int i = 0; i < MAX_SCHEDULE_NUM; i++) {
    schedule_host[i].tail = 1;
  }

  setUpSchedule(schedule_host);
#ifndef FIREWORK_BUFFER_SIZE
  cudaMalloc(&particles_device, sizeof(particle) * MAX_PARTICLE_NUM);
#else
  cudaMalloc(&particles_device, sizeof(firework) * FIREWORK_BUFFER_SIZE);
#endif
  cudaMalloc(&tails_device, sizeof(tail) * W * H);
  cudaMemset(tails_device, 0, sizeof(tail) * W * H);
  cudaMalloc(&idx_holder_device, sizeof(int) * 3);
  cudaMemset(idx_holder_device, 0, sizeof(int) * 3);
  cudaMalloc(&d_out, sizeof(uchar4) * W * H);

  startTime = timeSinceEpochMillisec();
  float counter = 0;
  while(timeSinceEpochMillisec() - startTime < DURATION * 1000) {
#ifndef FIREWORK_BUFFER_SIZE
    kernelLauncher(d_out, W, H, particles_device, tails_device, idx_holder_device, float(timeSinceEpochMillisec() - startTime)/1000.0f);
#else
    kernelLauncher(d_out, W, H, particles_device, tails_device, idx_holder_device, float(timeSinceEpochMillisec() - startTime)/1000.0f, rand_generate);
#endif
    counter++;
  }
  printf("fps cuda = %d\n", (int)(counter/(float)DURATION));

#ifndef FIREWORK_BUFFER_SIZE
  uchar4 *d_host = (uchar4*)malloc(sizeof(uchar4) * W * H);
  particle *particle_host = (particle*)malloc(sizeof(uchar4) * W * H);
  tail *tail_host = (tail*)malloc(sizeof(tail) * W * H);
  int idx_holder_host[3];
  idx_holder_host[0] = 0;
  idx_holder_host[1] = 0;
  idx_holder_host[2] = 0;

  startTime = timeSinceEpochMillisec();
  counter = 0;
  while(timeSinceEpochMillisec() - startTime < DURATION * 1000) {
    seqLauncher(d_host, W, H, particle_host, tail_host, idx_holder_host, float(timeSinceEpochMillisec() - startTime)/1000.0f, schedule_host);
    counter++;
    // printf("%d %d\n", idx_holder_host[1], idx_holder_host[0]);
  }
   printf("fps sequential = %d\n", (int)(counter/(float)DURATION));
#endif
  return 0;
}