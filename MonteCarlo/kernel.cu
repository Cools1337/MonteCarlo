
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
#define MAX 100

__global__ void kernel_simpson(double* a, double* c, unsigned int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x-1;
    if (tid > 0)
    {
        double x = tid;
        c[tid] = 1 / x;
    }
}


__global__ void reduce(double* v, double* per_block_sum, unsigned int n)
{
    __shared__ double sdata[1024];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        sdata[tid] = v[i];
        __syncthreads();
        for (int s = 1; s < blockDim.x; s *= 2) {
            if (tid % (2 * s) == 0)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            per_block_sum[blockIdx.x] = sdata[0];
    }

}

int main()
{
    int n = (END - START + 1) / STEP;
    int threadsPerBlock = 512;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    double* mas = new double[n];
    double* dev_mas;
    
    double* c = new double[n];

    double* dev_c;
    auto start = std::chrono::system_clock::now();
    cudaMalloc((void**)&dev_mas, n * sizeof(double));
    cudaMalloc((void**)&dev_c, n * sizeof(double));

    cudaMemcpy(dev_mas, mas, n * sizeof(double), cudaMemcpyHostToDevice);

    kernel_simpson <<<blocksPerGrid, threadsPerBlock >>> (dev_mas, dev_c, n);

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
    //double* dev_v, * per_block_sum;
    //double* sums = new double[blocksPerGrid];
    //cudaMalloc((void**)&per_block_sum, sizeof(double) * blocksPerGrid);
    //cudaMalloc((void**)&dev_v, sizeof(double) * n);
    //cudaMemcpy(dev_v, c, sizeof(double) * n, cudaMemcpyHostToDevice);
    //reduce <<<blocksPerGrid, threadsPerBlock >>> (dev_v, per_block_sum, n);
    //cudaDeviceSynchronize();
    //cudaMemcpy(sums, per_block_sum, sizeof(double) * blocksPerGrid, cudaMemcpyDeviceToHost);
    //cudaFree(dev_v);
    //cudaFree(per_block_sum);
    ////auto end = std::chrono::system_clock::now();
    //sum = 0;
    //for (int i = 0; i < blocksPerGrid; i++)
    //    sum += sums[i];


    //
    //std::cout << "Result: " << sum * ((double)STEP / (double)3) << "\n";
    //std::cout << "Time: " << elapsed.count() << " sec.";


    return 0;
}
