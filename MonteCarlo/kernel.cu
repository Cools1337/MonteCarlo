
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
#define MAX 10000000
#define THREADS_PER_BLOCK 1024

__device__ double getFunctionValue(double x) {
    return 1 / x;
}

__global__ void monteCarlo(double* integral, unsigned int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x - 1;
    curandState_t state;
    curand_init(tid, /* seed контролирует последовательность значений, которые генерируются*/
        0, /* порядковый номер важен только с несколькими ядрами*/
        0,
        &state); /* curand работает как rand - за исключением того, что он принимает состояние как параметр*/

    double result = curand(&state) % MAX;
    if (tid > 0)
    {
        double x = result*STEP;
        integral[tid] = getFunctionValue(x);
    }
}

__global__ void monteCarloWithShared(double* integral, unsigned int n) {
    __shared__ double cache[THREADS_PER_BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    double x, temp = 0;

    while (tid <= n && tid > 0)
    {
        x = tid;
        temp += getFunctionValue(x);
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();
    int i = blockDim.x / 2;

    while (i != 0)
    {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        integral[blockIdx.x] = cache[0];
}

int main()
{
    int n = (END - START + 1) / STEP;
    int blocksPerGrid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    double* mas = new double[n];
    double* dev_mas;
    
    double* c = new double[n];

    double* dev_c;
    auto start = std::chrono::system_clock::now();
    cudaMalloc((void**)&dev_mas, n * sizeof(double));
    cudaMalloc((void**)&dev_c, n * sizeof(double));

    cudaMemcpy(dev_mas, mas, n * sizeof(double), cudaMemcpyHostToDevice);

    // monteCarlo <<< blocksPerGrid, THREADS_PER_BLOCK >>> (dev_c, n);
    monteCarloWithShared <<< blocksPerGrid, THREADS_PER_BLOCK >>> (dev_c, n);

    cudaMemcpy(c, dev_c, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dev_mas);
    cudaFree(dev_c);
    auto end = std::chrono::system_clock::now();
    double sum = 0;
    for (int i = 0; i < blocksPerGrid; i++)
        sum += c[i];
    std::cout << "Result: " << sum << "\n";
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " sec.";


    return 0;
}
