#include <stdlib.h>
#include <stdio.h>

void normalize(float *A, int height_A, int width_A)
{
		// calculate mean value for each column
	float *mean = (float*)malloc(sizeof(float)*width_A);

	for (int j = 0; j < width_A; j++)
	{
		mean[j] = 0.0;
		for (int i = 0; i < height_A; i++)
		{
			mean[j] += A[i*width_A+j];
		}
		mean[j] /= (float)height_A;
	}
	
	for (int i = 0; i < height_A; i++)
	{
		for (int j = 0; j < width_A; j++)
		{
			A[i*width_A+j] -= mean[j];
		}
	}

	/*
	//print array
	for(int i = 0; i < height_A; i++)
	{
		for(int j = 0; j < width_A; j++)
			printf("%f\t", A[i*width_A+j]);
		printf("\n");
	}
	*/
}