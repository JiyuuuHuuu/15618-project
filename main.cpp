// Entry point of fireworks
// Usage:
//   ./firework -f ./input/s000.txt

#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

int2 loc = {W/2, H/2};
float t = 0.0f;   //timer

GLuint pbo;
GLuint tex;
struct cudaGraphicsResource *cuda_pbo_resource;
particle *particles_device;
int *idx_holder_device;

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
    if (idx >= MAX_SCHEDULE_NUM) {
      // truncate the extra
      printf("Warning: too many input schedules, truncating tails");
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

void handleKeyPress(unsigned char key, int x, int y) {
  switch (key) {
    case 'q':
    case 'Q':
      exit(0);
      break;
  }
}

void render() {
  uchar4 *d_out = 0;
  cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);
  kernelLauncher(d_out, W, H, particles_device, idx_holder_device, t);
  cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void drawTexture() {
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
  glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
  glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
  glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
  glEnd();
  glDisable(GL_TEXTURE_2D);
}

void time(int x) 
{
	if (glutGetWindow() )
	{
		glutPostRedisplay();
		glutTimerFunc(10, time, 0);
		t += 0.0166f;
	}
} 

void display() {
  render();
  drawTexture();
  glutSwapBuffers();
}

void initGLUT(int *argc, char **argv) {
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(W, H);
  glutCreateWindow(TITLE_STRING);
  glutKeyboardFunc(handleKeyPress);
}

void initPixelBuffer() {
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, 4*W*H*sizeof(GLubyte), 0, GL_STREAM_DRAW);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void exitfunc() {
  if (pbo) {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
  }
}

int main(int argc, char** argv) {
  makePalette();

  // initiate CUDA mem
  particle *particles_host;
  particles_host = (particle *)malloc(MAX_SCHEDULE_NUM*sizeof(particle));

  for (int i = 0; i < MAX_SCHEDULE_NUM; i++) {
    particles_host[i].t_0 = -1.0f;
    particles_host[i].p_0 = make_float2(0.0f, 0.0f);
    particles_host[i].v_0 = make_float2(0.0f, 0.0f);
    particles_host[i].a = make_float2(0.0f, 0.0f);
    particles_host[i].r = 0.0f;
    particles_host[i].explosion_height = 0.0f;
    particles_host[i].color = 0;
  }

  // Parse command line arguments.
  bool parse_success = false;
  CmdParser cmd_parser(argc, argv);
  std::string file = cmd_parser.GetOption("-f");
  // Parse file.
  if (!file.empty()) {
    parseFile(file, particles_host);
  } else {
    parseFile("./input/s000.csv", particles_host);
  }
  // Validate schedule.
  if (!check_schedule(particles_host)) {
    printf("Error: invalid schedule\n");
    return 1;
  }
  // printf("**t_0: %f\n", particles_host[0].t_0);

  // particles_host[0].p_0 = make_float2(300.0f, 580.0f);
  // particles_host[0].a = make_float2(0.0f, G);
  // particles_host[0].t_0 = 1.0f;
  // particles_host[0].v_0 = make_float2(0.0f, -100.0f);
  // particles_host[0].explosion_height = 300.0f;
  // particles_host[0].r = 10.0f;

  for (int i = 0; i < MAX_SCHEDULE_NUM; i++) {
    particles_host[i].explosion_height = 400.0f;
  }

  setUpSchedule(particles_host);
  cudaMalloc(&particles_device, sizeof(particle) * MAX_PARTICLE_NUM);
  cudaMalloc(&idx_holder_device, sizeof(int) * 3);
  cudaMemset(idx_holder_device, 0, sizeof(int) * 3);

  initGLUT(&argc, argv);
  gluOrtho2D(0, W, H, 0);
  glutDisplayFunc(display);
  time(0);
  glewInit();
  initPixelBuffer();
  printf("start rendering...\n");
  glutMainLoop();
  atexit(exitfunc);
  return 0;
}