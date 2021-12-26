
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

__global__ void monteCarlo(double* a, double* c, unsigned int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x - 1;
    curandState_t state;
    curand_init(tid, /* seed контролирует последовательность значений, которые генерируются*/

        0, /* порядковый номер важен только с несколькими ядрами*/

        0,

        &state);
    /* curand работает как rand - за исключением того, что он принимает состояние как параметр*/

    double result = curand(&state) % MAX;
    if (tid > 0)
    {
        double x = result*STEP;
        c[tid] = 1 / x;
    }
}

int main()
{
    int n = (END - START + 1) / STEP;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    double* mas = new double[n];
    double* dev_mas;
    
    double* c = new double[n];

    double* dev_c;
    auto start = std::chrono::system_clock::now();
    cudaMalloc((void**)&dev_mas, n * sizeof(double));
    cudaMalloc((void**)&dev_c, n * sizeof(double));

    cudaMemcpy(dev_mas, mas, n * sizeof(double), cudaMemcpyHostToDevice);

    monteCarlo <<<blocksPerGrid, threadsPerBlock >>> (dev_mas, dev_c, n);

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
