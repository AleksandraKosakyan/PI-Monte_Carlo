#include <cstdlib>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>
#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <device_functions.h> 
#include "device_launch_parameters.h"


#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
printf("Cuda error: %s\n", cudaGetErrorString(err));    \
printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
}       

const long N = 33554432; 


__global__ void calculationPiGPU(float *x, float *y, int *blocksCounts) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	int bias = gridDim.x * blockDim.x;

	
	__shared__ int sharedCounts[512]; 

	int countPointsInCircle = 0;
	for (int i = idx; i < N; i += bias) {
		if (x[i] * x[i] + y[i] * y[i] < 1) {
			countPointsInCircle++;
		}
	}
	sharedCounts[threadIdx.x] = countPointsInCircle;

	__syncthreads();

	
	if (threadIdx.x == 0) {
		int total = 0;
		for (int j = 0; j < 512; j++) {
			total += sharedCounts[j];
		}
		blocksCounts[blockIdx.x] = total;
	}
}


float calculationPiCPU(float *x, float *y) {
	int countPointsInCircle = 0; 
	for (int i = 0; i < N; i++) {
		if (x[i] * x[i] + y[i] * y[i] < 1) {
			countPointsInCircle++;
		}
	}
	return float(countPointsInCircle) * 4 / N;
}



int main()
{
	setlocale(LC_ALL, "RUS");
	float *X, *Y, *devX, *devY;

	
	X = (float *)calloc(N, sizeof(float));
	Y = (float *)calloc(N, sizeof(float));

	
	CUDA_CHECK_ERROR(cudaMalloc((void **)&devX, N * sizeof(float)));
	CUDA_CHECK_ERROR(cudaMalloc((void **)&devY, N * sizeof(float)));

	curandGenerator_t curandGenerator; 
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32); 
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL); 
	curandGenerateUniform(curandGenerator, devX, N); 
	curandGenerateUniform(curandGenerator, devY, N);
	curandDestroyGenerator(curandGenerator); 

	
	CUDA_CHECK_ERROR(cudaMemcpy(X, devX, N * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERROR(cudaMemcpy(Y, devY, N * sizeof(float), cudaMemcpyDeviceToHost));

	clock_t  start_time = clock();
    float cpu_result = calculationPiCPU(X, Y);
	clock_t  end_time = clock();
	std::cout << "Время на CPU = " << (double)((end_time - start_time) * 1000 / CLOCKS_PER_SEC) << " мсек" << std::endl;
	std::cout << "result: " << cpu_result << std::endl;
	
	int *dev_blocks_counts = 0, *blocks_counts = 0;
	float gpuTime = 0;

	cudaEvent_t start;
	cudaEvent_t stop;

	int blockDim = 512; 
	int gridDim = N / (128 * blockDim); 


	blocks_counts = (int *)calloc(gridDim, sizeof(int));

	CUDA_CHECK_ERROR(cudaMalloc((void **)&dev_blocks_counts, 512 * sizeof(int)));

	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));

	cudaEventRecord(start, 0);

	calculationPiGPU << <gridDim, blockDim >> >(devX, devY, dev_blocks_counts);

	
	CUDA_CHECK_ERROR(cudaMemcpy(blocks_counts, dev_blocks_counts, gridDim * sizeof(int), cudaMemcpyDeviceToHost));

	int countPointsInCircle = 0;
	for (int i = 0; i < gridDim; i++) {
		countPointsInCircle += blocks_counts[i];
	}

	
	float gpu_result = (float) countPointsInCircle * 4 / N;

	
	cudaEventRecord(stop, 0);

	
	cudaEventSynchronize(stop);

	
	cudaEventElapsedTime(&gpuTime, start, stop);

	std::cout << "Время на GPU = " << gpuTime << " мсек" << std::endl;
	std::cout << "result: " << gpu_result << std::endl;

	
	CUDA_CHECK_ERROR(cudaEventDestroy(start));
	CUDA_CHECK_ERROR(cudaEventDestroy(stop));

	CUDA_CHECK_ERROR(cudaFree(devX));
	CUDA_CHECK_ERROR(cudaFree(devY));
	CUDA_CHECK_ERROR(cudaFree(dev_blocks_counts));

	
	delete X;
	delete Y;

	system("pause");
	return 0;
}
