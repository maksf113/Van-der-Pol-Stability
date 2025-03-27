#include "kernel.h"
#include <stdio.h>
#include <math.h>

constexpr int TX = 32;
constexpr int TY = 32;
constexpr float len = 5.0;
constexpr float dt = 0.005;
constexpr float finalTime = 10.0;
/*****************************************************
* solving equation:
* x'' = f(x, x', t)
* e.g.: x'' + x - const * (1 - x^2) * x' for van der Pol oscialtor
* substituting y = x' results in system of equations:
* x' = y
* y' = f(x, y, t)
* 
* Descretized variables:
t_n = dt * n
* x_k = x(t_k)
* y_k = y(t_k)
* Forward Euler:
* x_k+1 = x_k + y_k * dt
* y_k+1 = y_k + f(x_k, y_k, t_k) * dt
*****************************************************/
__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

// sclae int [-len, len]
__device__
float scale(int i, int w) { return 2.0f * len * (float(i) / w - 0.5f); }
// RHS for the equation
__device__
float f(float x, float y, float param, int sys)
{
    if (sys == 1) return x - 2 * param * y; // negative stiffness (reversed pendulum)
    if (sys == 2) return -x + param * (1.0f - x * x) * y; // van der Pol
    else return -x - 2 * param * y; // pendulum
}
// explicit Euler solver
__device__
float2 euler(float x, float y, float dt, float tFinal, float param, int sys)
{
    for (float t = 0.0f; t <= tFinal; t += dt)
    {
        x = x + y * dt;
        y = y + f(x, y, param, sys) * dt;
    }
    return make_float2(x, y);
}

__global__
void stabilityKernel(uchar4* d_out, int width, int height, float param, int sys)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height)
        return;
    // index
    const int i = row * width + col;
    // initial state
    const float x0 = scale(col, width);
    const float y0 = scale(row, height);
    // initial distance from stable solution (0,0)
    const float dist0 = sqrt(x0 * x0 + y0 * y0);
    const float2 finalPos = euler(x0, y0, dt, finalTime, param, sys);
    const float distFinal = sqrt(finalPos.x * finalPos.x + finalPos.y * finalPos.y);
    // assign color based on distance change
    const float distRatio = distFinal / dist0;

    d_out[i].x = clip(distRatio * 255); // red - growth
    d_out[i].y = (col == width / 2 || row == height / 2) ? 166 : 0; // green - axes
    d_out[i].z = clip((1.0f / distRatio) * 255); // blue - 1/growth
    d_out[i].w = 255; // alpha channel
}

void stabilityKernelLauncher(uchar4* d_out, int width, int height, float param, int sys)
{
    const dim3 blockSize = dim3(TX, TY);
    const dim3 gridSize = dim3((width + TX - 1) / TX, (height + TY - 1) / TY);
    stabilityKernel << <gridSize, blockSize >> > (d_out, width, height, param, sys);
    CUDA(cudaGetLastError()); // Check for launch errors
    CUDA(cudaDeviceSynchronize()); // Check for runtime errors
}


__global__
void distKernel(uchar4* d_out, int w, int h, int2 pos)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if ((c >= w) || (r >= h))
        return;
    const int i = r * w + c;

    const int d = sqrtf((c - pos.x) * (c - pos.x) + (r - pos.y) * (r - pos.y));

    const unsigned char intensity = clip(255 - d);
    d_out[i].x = intensity;
    d_out[i].y = intensity;
    d_out[i].z = 0;
    d_out[i].w = 255;
}

void distKernelLauncher(uchar4* d_out, int w, int h, int2 pos)
{
    const dim3 blockSize(TX, TY);
    const int bx = (w + TX - 1) / TX;
    const int by = (h + TY - 1) / TY;
    const dim3 gridSize(bx, by);
    distKernel << <gridSize, blockSize >> > (d_out, w, h, pos);
    CUDA(cudaGetLastError()); // Check for launch errors
    CUDA(cudaDeviceSynchronize()); // Check for runtime errors
}