#include <cuda_runtime.h>

struct particle {
    float2 start_position;
    float2 start_velocity;
    float color;
    float radius;
    float explosion_height;
    float launch_time;
};