
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cula_lapack.h>

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA SDK samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>

void checkStatus(culaStatus status)
{
    char buf[256];

    if(!status)
        return;

    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    printf("%s\n", buf);

    culaShutdown();
    exit(EXIT_FAILURE);
}


float* svd(float *cov, int width)
{
    culaStatus status;

    printf("Initializing CULA\n\n");
    status = culaInitialize();
    checkStatus(status);

	float *covariance = (float *)malloc(sizeof(float) * width*width);
	
	//copy coavariance matrix
	for(int i = 0; i < width*width; i++)
		covariance[i] = cov[i];
	
	/* printing covariance matrix: for debuging purpose
	///////////////////////////////////////////////////
	printf("---------------------Before SVD---------------------\n");
	for(int i = 0; i < width*width; i++)
		printf("%f\t", covariance[i]);
	printf("\n");
	*/

	//allocate memory for host
	float *singularValue = (float *)malloc(sizeof(float) * width);
	float *U = (float *)malloc(sizeof(float) * width*width);
	float *VT = (float *)malloc(sizeof(float) * width*width);
	
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

	//call SVD function in CULA
	status = culaSgesvd('S', 'S', width, width, covariance, width, singularValue, U, width, VT, width);
	checkStatus(status);

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
		
	printf("Singular value decomposition time is: %.3f ms\n\n", msecTotal);

	/* Printing matrix U: for debuging purpose
	//////////////////////////////////////////
	printf("---------------------After SVD---------------------\n");
	for(int i = 0; i < width*width; i++)
	{
		printf("%f\t", U[i]);
		if((i+1) % width == 0)
			printf("\n");
	}

	printf("\n");

	*/

	return VT;
}

