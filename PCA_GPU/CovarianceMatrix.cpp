// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA SDK samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

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
// end of CUDA Helper Functions

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

void initializeCUDA(int &devID)
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    devID = 0;
    
	cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
}

////////////////////////////////////////////////////////////////////////////////
//! matrix multiplication with its transpose (C = A^T * A) using CUBLAS
////////////////////////////////////////////////////////////////////////////////
float* covarianceCalucation(int devID, float *A, int height_A, int width_A)
{
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
	unsigned int size_A = height_A * width_A;
    unsigned int mem_size_A = sizeof(float) * size_A;

	unsigned int size_B = height_A * width_A;
    unsigned int mem_size_B = sizeof(float) * size_B;

    unsigned int size_C = width_A * width_A;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float *h_C      = (float *) malloc(mem_size_C);
    float *h_CUBLAS = (float *) malloc(mem_size_C);

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
    error = cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy d_A A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, A, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy d_B A returned error code %d, line(%d)\n", error, __LINE__);
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
		//ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width_B, height_A, width_A, &alpha, d_B, width_B, d_A, width_A, &beta, d_C, width_B);
        
		// C = AT * A
        ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, width_A, width_A, height_A, &alpha, d_B, width_A, d_A, width_A, &beta, d_C, width_A);

		// C =  A * AT
        //ret = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, height_A, height_A, width_A, &alpha, d_B, width_A, d_A, width_A, &beta, d_C, height_A);

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
		
		printf("matrix multiplication time is: %.3f ms\n\n", msecTotal);

        // copy result from device to host
        error = cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost);

        if (error != cudaSuccess)
        {
            printf("cudaMemcpy h_CUBLAS d_C returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
    }

    // clean up memory
    free(A);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();

    return h_CUBLAS;
}