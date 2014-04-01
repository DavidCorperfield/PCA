#include <stdlib.h>
#include <stdio.h>
#include <time.h>

////////////////////////////////////////////////////////////////////////////////
// Read the matrix from .txt file
////////////////////////////////////////////////////////////////////////////////
float* readData(char* fileName, int height, int width)
{
	FILE* stream;
	float in_value;

	if ((stream = fopen(fileName,"r")) == NULL)
	{
		printf("cannot open file\n");
		exit(1);
	}

	//allocate host memory
	unsigned int size = width * height;
    unsigned int mem_size = sizeof(float) * size;
    float *data = (float *)malloc(mem_size);

	for (int i = 0; i < size; i++)
	{
			fscanf(stream, "%f", &in_value);
			data[i] = in_value;
	}
	
	return data;
}

////////////////////////////////////////////////////////////////////////////////
// Read the matrix from .txt file
////////////////////////////////////////////////////////////////////////////////
float *init(int height, int width)
{	
	//allocate host memory
	unsigned int size = width * height;
    unsigned int mem_size = sizeof(float) * size;
    float *data = (float *)malloc(mem_size);

	srand(time(NULL));

	for(int i = 0; i < height*width; i++)
		data[i] = (float)rand()/(float)RAND_MAX;

	return data;
}

void writeRandomDataToFile(float *data, int height, int width)
{
	FILE *f = fopen("rndData.txt", "w");

	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	/* print integers and floats */
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
			fprintf(f, "%f\t",  data[i*width+j]);
		fprintf(f, "\n");
	}

	fclose(f);
}

////////////////////////////////////////////////////////////////////////////////
// print the data in the matrix, this function is used for debuging purpose
////////////////////////////////////////////////////////////////////////////////
void printData(float* data, int height, int width)
{
	for(int i = 0; i < height;  i++)
	{
		for(int j = 0; j < width; j++)
			printf("%f\t", data[i*width+j]);
		printf("\n");
	}
}


