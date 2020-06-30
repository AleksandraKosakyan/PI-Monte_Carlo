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

const long N = 33554432; // Количество точек 


__global__ void calculationPiGPU(float *x, float *y, int *blocksCounts) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x; // номер элемента

	int bias = gridDim.x * blockDim.x;// cмещение по векторам

	/*
	разделяемая память в пределах одного блока, т.к всего в блоке 512 потоков, то размерность массива можно задать явно,
	сюда каждый поток будет записывать количесто точек пренадлежащих окружности
	*/
	__shared__ int sharedCounts[512]; 

	int countPointsInCircle = 0;
	for (int i = idx; i < N; i += bias) {
		if (x[i] * x[i] + y[i] * y[i] < 1) {
			countPointsInCircle++;
		}
	}
	sharedCounts[threadIdx.x] = countPointsInCircle;

	/*
	Эта функция заставит каждый поток ждать, пока
	(а) все остальные потоки этого блока достигнут этой точки и
	(б) все операции по доступу к разделяемой и глобальной памяти, совершенные потоками этого блока, завершатся и
	станут видны потокам этого блока.
	*/
	__syncthreads();

	// Первый поток каждого block`а будет вычислять суммарное количество точек, попавших в круг в каждом блоке 
	if (threadIdx.x == 0) {
		int total = 0;
		for (int j = 0; j < 512; j++) {
			total += sharedCounts[j];
		}
		blocksCounts[blockIdx.x] = total;
	}
}


float calculationPiCPU(float *x, float *y) {
	int countPointsInCircle = 0; //Количество точек попавших в круг
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

	//Выделяем память вектора на хосте
	X = (float *)calloc(N, sizeof(float));
	Y = (float *)calloc(N, sizeof(float));

	//Выделяем глобальную память для храния данных на девайсе
	CUDA_CHECK_ERROR(cudaMalloc((void **)&devX, N * sizeof(float)));
	CUDA_CHECK_ERROR(cudaMalloc((void **)&devY, N * sizeof(float)));

	curandGenerator_t curandGenerator; //создаем новый генератор
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32); // выбираем тип генератора, пусть будет алгоритм Мерсенна Твистера
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL); //«основа», на которой будут строиться случайные ряды
	curandGenerateUniform(curandGenerator, devX, N); // генерируем числа в количестве size и кладем их в а
	curandGenerateUniform(curandGenerator, devY, N);// генерируем числа в количестве size и кладем их в b
	curandDestroyGenerator(curandGenerator); //уничтожаем генератор

	//Копируем заполненные вектора с девайса на хост
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

	int blockDim = 512; // размер одного блока в потоках
	int gridDim = N / (128 * blockDim); // размер сетки


	blocks_counts = (int *)calloc(gridDim, sizeof(int));

	CUDA_CHECK_ERROR(cudaMalloc((void **)&dev_blocks_counts, 512 * sizeof(int)));

	//Создаем event'ы для синхронизации и замера времени работы GPU
	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));
	//Отмечаем старт расчетов на GPU
	cudaEventRecord(start, 0);

	calculationPiGPU << <gridDim, blockDim >> >(devX, devY, dev_blocks_counts);

	//Копируем результат с девайса на хост в blocks_counts
	CUDA_CHECK_ERROR(cudaMemcpy(blocks_counts, dev_blocks_counts, gridDim * sizeof(int), cudaMemcpyDeviceToHost));

	int countPointsInCircle = 0;
	for (int i = 0; i < gridDim; i++) {
		countPointsInCircle += blocks_counts[i];
	}

	// Полученное на GPU число π 
	float gpu_result = (float) countPointsInCircle * 4 / N;

	//Отмечаем окончание расчета
	cudaEventRecord(stop, 0);

	//Синхронизируемя с моментом окончания расчетов
	cudaEventSynchronize(stop);

	//Рассчитываем время работы GPU
	cudaEventElapsedTime(&gpuTime, start, stop);

	std::cout << "Время на GPU = " << gpuTime << " мсек" << std::endl;
	std::cout << "result: " << gpu_result << std::endl;

	//Чистим ресурсы на видеокарте
	CUDA_CHECK_ERROR(cudaEventDestroy(start));
	CUDA_CHECK_ERROR(cudaEventDestroy(stop));

	CUDA_CHECK_ERROR(cudaFree(devX));
	CUDA_CHECK_ERROR(cudaFree(devY));
	CUDA_CHECK_ERROR(cudaFree(dev_blocks_counts));

	//Чистим память на хосте
	delete X;
	delete Y;

	system("pause");
	return 0;
}
