#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "render.cuh"

__global__ void test_kernel(void) {
}

namespace firework {
	void parallel_render(void) {
		test_kernel <<<1, 1>>> ();
		printf("Hello, world!\n");
	}
}