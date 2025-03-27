#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

struct int2;
struct uchar4;

void stabilityKernelLauncher(uchar4* d_out, int width, int height, float param, int sys);
void kernelLauncher(uchar4* d_out, int w, int h, int2 pos);