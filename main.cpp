#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>

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

void render() {
  uchar4 *d_out = 0;
  cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);
  kernelLauncher(d_out, W, H, particles_device, t);
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
  }

  particles_host[0].p_0 = make_float2(600.0f, 300.0f);
  particles_host[0].t_0 = 2.0f;
  particles_host[0].r = 255.0f;
  particles_host[0].color = 0;

  setUpSchedule(particles_host);
  cudaMalloc(&particles_device, sizeof(particle) * MAX_PARTICLE_NUM);

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