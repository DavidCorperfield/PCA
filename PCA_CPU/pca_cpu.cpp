
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <sys\timeb.h>


#define p 3 //number of principal components
#define file "rndData.txt"

double* readData(char* fileName, int height, int width);
void printData(double* data, int height, int width);
void normalize(double *A, int height_A, int width_A);

int main()
{
	int matrix_height = 7000; 
	int matrix_width = 7000;
	double *data;
	struct timeb start, end; //timer
	float diff; 
	int info;
	int lwork = 5*matrix_width;
	char rob = 'S';

	float *singularValue = (float *)malloc(sizeof(float) * matrix_width); //memeroy for storing singluar values
	float *svd = (float *)malloc(sizeof(float) * matrix_width*matrix_width); //memory for storing U in SVD
	float *VT = (float *)malloc(sizeof(float) * matrix_width*matrix_width); //memory for storing V in SVD
	float * cov = (float *)malloc( matrix_width*matrix_width*sizeof( float )); //memory for storing covariance matrix in float format
	double *C = (double *)malloc( matrix_width*matrix_width*sizeof( double )); //memory for storing covariance matrix in double format
	double *data_buf = (double *)malloc(sizeof(double) * matrix_height*matrix_width); //memory for storing copy of original data input
	float *work = (float *)malloc(sizeof(float) * lwork); 

	//read data from files
	data = readData(file, matrix_height, matrix_width);

	//normilze data
	normalize(data, matrix_height, matrix_width);

	//store the copy of the data
	memcpy(data_buf, data, sizeof(double) * matrix_height*matrix_width);
	
	printf("Data loading finishied!\n\n");
	
	printf("covariance matrix calculation started!\n\n");

	//record start time for covariance matrix calucation
	ftime(&start);

	// C = A * A^T, matrix multiplication with its transpose using blas library
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, matrix_width,matrix_width,matrix_height,1.0,data,matrix_width,data,matrix_width,0.0,C,matrix_width);
	
	//record end time for covariance matrix calucation
	ftime(&end);
	
	//calculate the elapsed time for covariance matrix calucation in milisecond
	diff = (float) (1000.0 * (end.time - start.time) + (end.millitm - start.millitm));

	printf("covariance matrix calculation time is: %.3f\n\n", diff);

	printf("covariance matrix calculation finished!\n\n");

	//copy the covariance matrix into float format
	for(int i = 0; i < matrix_width*matrix_width; i++)
	{
		cov[i] = (float)C[i];
	}

	printf("SVD calculation started!\n\n");
	
	//record start time for svd
	ftime(&start);

	//svd using lapack library
	sgesvd(&rob, &rob, &matrix_width, &matrix_width, cov, &matrix_width, singularValue, svd, &matrix_width, VT, &matrix_width, work, &lwork, &info);
	
	//record end time for svd
	ftime(&end);
	
	//calculate the elapsed time for svd in milisecond
	diff = (float) (1000.0 * (end.time - start.time) + (end.millitm - start.millitm));

	printf("svd calculation time is: %.3f\n\n", diff);

	printf("Singular value decomposition finished!!!\n\n");

	printf("Projection start.......\n\n");

	double *projectedData = (double *)mkl_malloc( matrix_height*p*sizeof( double ), 64 );
	double *loadingMatrix = (double *)mkl_malloc( matrix_width*p*sizeof( double ), 64 );

	//constructing loading matrix, by selection first p column from the VT
	for(int i = 0; i < matrix_width; i++)
	{
		for(int j = 0; j < p; j++)
		{
			loadingMatrix[i*p+j] = VT[i*matrix_width+j];
		}
	}

	for(int i = 0; i < matrix_width; i++)
	{
		for(int j = 0; j < p; j++)
		{
				loadingMatrix[i*p+j] *= -1;
		}
	}

	ftime(&start);

	// C = A * B, multiplicate original matrix to loading matrix using blas library
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matrix_height,p,matrix_width,1.0,data_buf,matrix_width,loadingMatrix,p,0.0,projectedData,p);

	ftime(&end);

	diff = (float) (1000.0 * (end.time - start.time) + (end.millitm - start.millitm));

	printf("projection time is: %.3f\n\n", diff);

	printf("projection finished!\n\n");

	
	return 0;
}

