
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cmath> 
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>

#define START 1
#define END 500000
#define STEP 0.01
#define MAX_RANDOM_VALUE 10000
#define THREADS_PER_BLOCK 1024

__device__ double getFunctionValue(double x) {
    return 1/x;
}

__device__ double getRandonPoint(long seed) {
    curandState_t state;
    curand_init(seed, 0, 0, &state);
    double fraction = 1.0 / (RAND_MAX + 1.0);
    double result = (curand(&state) % MAX_RANDOM_VALUE) * fraction * (START - END + 1) + END;
    return result;
}

__device__ double sum(double* values, unsigned int n) {
    for (int i = 1; i < n; i++) {
        values[0] += values[i];
    }
    return values[0];
}

__global__ void monteCarlo(double* integral, unsigned int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    curandState_t state;
    curand_init(tid, 0, 0, &state);
    double fraction = 1.0 / (RAND_MAX + 1.0);
    double result = (curand(&state) % MAX_RANDOM_VALUE) * fraction * (START - END + 1) + END;
    if (tid <= n)
    {
        double x = result * STEP;
        integral[tid] = getFunctionValue(x);
    }

    if (tid == 0)
        integral[0] = sum(integral, n);
}

__global__ void monteCarloWithShared(double* integral, unsigned int n) {
    __shared__ double sums[THREADS_PER_BLOCK];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int sharedIndex = threadIdx.x;
    double result = getRandonPoint(tid);
    sums[sharedIndex] = 0;
    if (tid <= n)
    {
        double x = result * STEP;
        sums[sharedIndex] = getFunctionValue(x);
    }

    __syncthreads();
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (sharedIndex < i)
            sums[sharedIndex] += sums[sharedIndex + i];

        __syncthreads();
        i /= 2;
    }
    if (sharedIndex == 0)
        integral[blockIdx.x] = sums[0];
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}
template <unsigned int blockSize>
__global__ void monteCarloWithReduce(double* integral, unsigned int n) {

    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        double x = getRandonPoint(i);
        sdata[tid] += getFunctionValue(x);
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) integral[blockIdx.x] = sdata[0];
}

int main()
{
    int n = (END - START + 1) / STEP;
    int blocksPerGrid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    double* mas = new double[n];
    double* dev_mas;
    double result = 1;

    double* c = new double[n];
    double* dev_c;
    cudaMalloc((void**)&dev_mas, n * sizeof(double));
    cudaMalloc((void**)&dev_c, n * sizeof(double));

    cudaMemcpy(dev_mas, mas, n * sizeof(double), cudaMemcpyHostToDevice);

    auto start = std::chrono::system_clock::now();
    monteCarlo <<< blocksPerGrid, THREADS_PER_BLOCK >>> (dev_c, n);
    // monteCarloWithShared <<< blocksPerGrid, THREADS_PER_BLOCK >>> (dev_c, n);
    // monteCarloWithReduce <THREADS_PER_BLOCK><<< blocksPerGrid, THREADS_PER_BLOCK >>> (dev_c, n);
    
    cudaMemcpy(c, dev_c, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, dev_c, n * sizeof(double), cudaMemcpyDeviceToHost);
    auto end = std::chrono::system_clock::now();
    cudaFree(dev_mas);
    cudaFree(dev_c);
    std::cout << "Result: " << sum << "\n";
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " sec.";


    return 0;
}
