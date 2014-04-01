// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA SDK samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>

#include <stdio.h>
#include <stdlib.h>
////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions (in addition to helper_cuda.h)

void inline checkError(cublasStatus_t status, const char *msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("%s", msg);
        exit(EXIT_FAILURE);
    }
}

float *projection(int devID, float* data, float *U, int height, int width, int pc)
{
	float *projectedData;
	float *loadingMatrix;

	if(pc > width)
	{
		printf("Number of principal components cannnot be > than matrix width");
		exit(1);
	}

	unsigned int size_A = height * width;
    unsigned int mem_size_A = sizeof(float) * size_A;

	unsigned int size_B = width * pc;
    unsigned int mem_size_B = sizeof(float) * size_B;

	unsigned int size_C = height * pc;
    unsigned int mem_size_C = sizeof(float) * size_C;

	projectedData = (float *)malloc(mem_size_C);
	loadingMatrix = (float *)malloc(mem_size_B);

	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < pc; j++)
		{
			loadingMatrix[i*pc+j] = U[i*width+j];
		}
	}

	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < pc; j++)
		{
				loadingMatrix[i*pc+j] *= -1;
		}
	}
	
	/*
	printf("------------Matrix U------------\n");
	for(int i = 0; i < width*width; i++)
	{
		printf("%f\t", U[i]);

		if( (i+1) % width == 0)
			printf("\n");
	}

	printf("\n------------Matrix Loading------------\n");
	for(int i = 0; i < width*pc; i++)
	{
		printf("%f\t", loadingMatrix[i]);

		if( (i+1) % pc == 0)
			printf("\n");
	}
	
	printf("\n");
	printf("-------------------------------------------\n");
	*/
	cudaDeviceProp deviceProp;
    cudaError_t error;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;

    // allocate device memory
    float *d_A, *d_B, *d_C;

    error = cudaMalloc((void **) &d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, data, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy d_A data returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, loadingMatrix, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy d_B loadingMatrix returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // setup execution parameters
    // dim3 threads(block_size, block_size);
    // dim3 grid(width_B / threads.x, height_A / threads.y);

    // create and start timer
    //printf("Computing result using CUBLAS...\n");

    // CUBLAS version 2.0
    {
        cublasHandle_t handle;

        cublasStatus_t ret;

        ret = cublasCreate(&handle);

        if (ret != CUBLAS_STATUS_SUCCESS)
        {
            printf("cublasCreate returned error code %d, line(%d)\n", ret, __LINE__);
            exit(EXIT_FAILURE);
        }

        const float alpha = 1.0f;
        const float beta  = 0.0f;

		cudaError_t error;
		// Allocate CUDA events that we'll use for timing
		cudaEvent_t start;
		error = cudaEventCreate(&start);

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		cudaEvent_t stop;
		error = cudaEventCreate(&stop);

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		// Record the start event
		error = cudaEventRecord(start, NULL);

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		// C = A * B
		ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, pc, height, width, &alpha, d_B, pc, d_A, width, &beta, d_C, pc);

        if (ret != CUBLAS_STATUS_SUCCESS)
        {
            printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__);
            exit(EXIT_FAILURE);
        }

		// Record the stop event
        error = cudaEventRecord(stop, NULL);

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        // Wait for the stop event to complete
        error = cudaEventSynchronize(stop);

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        float msecTotal = 0.0f;
        error = cudaEventElapsedTime(&msecTotal, start, stop);

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
		
		printf("Projection time is: %.3f ms\n\n", msecTotal);

        // copy result from device to host
		error = cudaMemcpy(projectedData, d_C, mem_size_C, cudaMemcpyDeviceToHost);

        if (error != cudaSuccess)
        {
            printf("cudaMemcpy h_CUBLAS d_C returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
    }

	
    // clean up memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();

	/*
	for(int i = 0; i < height*pc; i++)
	{
		printf("%f\t", projectedData[i]);

		if( (i+1) % pc == 0)
			printf("\n");
	}
	*/

	return projectedData;
}