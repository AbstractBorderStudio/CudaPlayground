/*
	Daniel Bologna 2024
*/

// system includes
#include <stdio.h>

// cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// main functionality
cudaError_t AddVectors(const int* vec_a, const int* vec_b, int* vec_c, int size);

// kernel
__global__ void AddVectorsInGpu(int* result, const int* a, const int* b) {
	// get vectors a and b and add them in the result
	// first get thread idx
	int i = threadIdx.x;
	// sum vector in the i position
	result[i] = a[i] + b[i];
}

int main() {
	cudaError_t cudaStatus;

	// create vectors
	const int size = 5;
	const int A[size] = { 1,2,3,4,5 };
	const int B[size] = { 5,3,4,1,2 };
	int C[size] = { 0 };

	// pass them to the function
	cudaStatus = AddVectors(A, B, C, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error AddVectors!\n");
		return 1;
	}
	else {
		fprintf(stdout, "Successfully Added Vectors\n");
	}

	// print result
	printf("The sum between A[%d,%d,%d,%d,%d] and B[%d,%d,%d,%d,%d] is C[%d,%d,%d,%d,%d]\n\n",
		A[0], A[1], A[2], A[3], A[4],
		B[0], B[1], B[2], B[3], B[4],
		C[0], C[1], C[2], C[3], C[4]);

	// app exit
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error Resetting Cuda Device!\n");
		return 1;
	}
	else {
		fprintf(stdout, "Successfully Reset Device\n");
	}

	return 0;
}

cudaError_t AddVectors(const int* vec_a, const int* vec_b, int* vec_c, int size)
{
	cudaError_t cudaStatus;
	// create buffer to pass to the gpu
	const int* cuda_a;
	const int* cuda_b;
	int* cuda_c;

	// alloc memory for vector A in the GPU
	cudaStatus = cudaMalloc((void**)&cuda_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error cudaMalloc A!\n");
		goto Error;
	}
	else {
		fprintf(stdout, "Successfully Allocated Buffer A\n");
	}

	// alloc memory for vector B in the GPU
	cudaStatus = cudaMalloc((void**)&cuda_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error cudaMalloc B!\n");
		goto Error;
	}
	else {
		fprintf(stdout, "Successfully Allocated Buffer B\n");
	}

	// alloc memory for vector C in the GPU
	cudaStatus = cudaMalloc((void**)&cuda_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error cudaMalloc C!\n");
		goto Error;
	}
	else {
		fprintf(stdout, "Successfully Allocated Buffer C\n");
	}

	// copy the input vectors inside the cuda buffer just created
	// since we need to copy from cpu to gpu we use the flag cudaMemcpyHostToDevice
	cudaStatus = cudaMemcpy((void*)cuda_a, vec_a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error cudaMemcpy A!\n");
		goto Error;
	}
	else {
		fprintf(stdout, "Successfully copied data on cuda_a\n");
	}

	cudaStatus = cudaMemcpy((void*)cuda_b, vec_b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error cudaMemcpy B!\n");
		goto Error;
	}
	else {
		fprintf(stdout, "Successfully copied data on cuda_b\n");
	}

	// call the cuda kernel which handles the sum
	AddVectorsInGpu << <1, size >> > (cuda_c, cuda_a, cuda_b);

	// now cuda_c has all the data, so we need to copy back data from device to host
	cudaStatus = cudaMemcpy((void*)vec_c, cuda_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error cudaMemcpy C!\n");
		goto Error;
	}
	else {
		fprintf(stdout, "Successfully copied data back on vec_c\n");
	}

Error:
	// cleanup, dealloc variables in the gpu
	cudaFree(&cuda_a);
	cudaFree(&cuda_b);
	cudaFree(&cuda_c);

	return cudaStatus;
}
