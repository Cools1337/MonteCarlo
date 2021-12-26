
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
//
//__device__ double atomicAdd(double* address, double val)
//{
//    unsigned long long int* address_as_ull =
//        (unsigned long long int*)address;
//    unsigned long long int old = *address_as_ull, assumed;
//
//    do {
//        assumed = old;
//        old = atomicCAS(address_as_ull, assumed,
//            __double_as_longlong(val +
//                __longlong_as_double(assumed)));
//
//        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//    } while (assumed != old);
//
//    return __longlong_as_double(old);
//}

__global__ void monteCarlo(double* integral, unsigned int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x - 1;
    curandState_t state;
    curand_init(tid, /* seed контролирует последовательность значений, которые генерируются*/
        0, /* порядковый номер важен только с несколькими ядрами*/
        0, &state); /* curand работает как rand - за исключением того, что он принимает состояние как параметр*/
    double fraction = 1.0 / (RAND_MAX + 1.0);
    double result = (curand(&state)% MAX_RANDOM_VALUE) * fraction * (START - END + 1) + END ;
    if (tid <= n && tid > 0)
    {
        double x = result * STEP;
        integral[tid] = getFunctionValue(x);
    }

    /*__syncthreads();

    if (tid == 0)
    {
        double x = result * STEP;
        for (size_t i = 0; i < 48829; i++)
        {
           integral[0] += integral[i];
        }
    }*/
}

__global__ void monteCarloWithShared(double* integral, unsigned int n) {
    __shared__ double sums[THREADS_PER_BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x - 1;
    int cacheIndex = threadIdx.x;
    double x;
    curandState_t state;
    curand_init(tid, 0, 0, &state);
    double fraction = 1.0 / (RAND_MAX + 1.0);
    double result = (curand(&state) % MAX_RANDOM_VALUE) * fraction * (START - END + 1) + END;
    if (tid <= n && tid > 0)
    {
        x = result * STEP;
        sums[cacheIndex] = getFunctionValue(x);
    }

    __syncthreads();

    if (cacheIndex == 0) {
        for (int i = 1; i < THREADS_PER_BLOCK; i++) {
            sums[0] += sums[i];
        }
        integral[blockIdx.x] = sums[0];
    }
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
    //monteCarloWithShared << < blocksPerGrid, THREADS_PER_BLOCK >> > (dev_c, n);
    
    cudaMemcpy(c, dev_c, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, dev_c, n * sizeof(double), cudaMemcpyDeviceToHost);
    auto end = std::chrono::system_clock::now();
    cudaFree(dev_mas);
    cudaFree(dev_c);
    double sum = 0;
    for (int i = 0; i < blocksPerGrid; i++)
        sum += c[i];
    std::cout << "Result: " << sum << "\n";
    //std::cout << "Result: " << c[0] << "\n";
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " sec.";


    return 0;
}
