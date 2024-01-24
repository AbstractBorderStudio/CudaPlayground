/*
	Daniel Bologna 2024
*/

// system includes
#include <stdio.h>

// cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// main func
cudaError_t MultMatrices(const int* A, const int* B, int* C, int m, int n, int k);

// kernel
__global__ void MultMatrixGPU(int* res, const int* A, const int* B, int m, int n, int k) {
	// each thread makes a vector-vector multiplication
}

int main() {
	cudaError_t cudaStatus;

	// create matrices // to fix indexing!
	const int m = 2;
	const int n = 2;
	const int k = 3;
	const int A[m][k] = {
		{1,2,3},
		{1,2,3}
	};
	const int B[k][n] = {
		{1,2},
		{1,2},
		{1,2}
	};
	int C[m][m] = { 0 }; // matrices filled with zeros

	// pass pointer to the first element
	cudaStatus = MultMatrices(A[0], B[0], C[0], m, n, k);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error MultMatrices!\n");
		return 1;
	}
	else {
		fprintf(stdout, "Successfully Multiplied Matrices\n");
	}

	// print result

	// exit app
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

cudaError_t MultMatrices(const int* A, const int* B, int* C, int m, int n, int k)
{
	cudaError_t cudaStatus = cudaSuccess;

	// create cuda buffers
	const int* cuda_a;
	const int* cuda_b;
	int* cuda_c;

	int size_a = m * k;
	int size_b = k * n;
	int size_c = m * n;

	// malloc in gpu
	cudaStatus = cudaMalloc((void**)&cuda_a, size_a * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error cudaMalloc cuda_a!\n");
		goto Error;
	}
	else {
		fprintf(stdout, "Successfully allocated cuda_a\n");
	}

	cudaStatus = cudaMalloc((void**)&cuda_b, size_b * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error cudaMalloc cuda_b!\n");
		goto Error;
	}
	else {
		fprintf(stdout, "Successfully allocated cuda_b\n");
	}
	
	cudaStatus = cudaMalloc((void**)&cuda_c, size_c * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error cudaMalloc cuda_c!\n");
		goto Error;
	}
	else {
		fprintf(stdout, "Successfully allocated cuda_c\n");
	}

	// copy mem from host to device
	cudaStatus = cudaMemcpy((void*)cuda_a, A, size_a * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error cudaMemcpy cuda_a!\n");
		goto Error;
	}
	else {
		fprintf(stdout, "Successfully copied data inside cuda_a\n");
	}

	// copy mem from host to device
	cudaStatus = cudaMemcpy((void*)cuda_b, B, size_b * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error cudaMemcpy cuda_b!\n");
		goto Error;
	}
	else {
		fprintf(stdout, "Successfully copied data inside cuda_b\n");
	}

	// launch kernel

	// coby data back
	// copy mem from host to device
	//cudaStatus = cudaMemcpy(C, (void*)cuda_c, size_c * sizeof(int), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "Error cudaMemcpy cuda_c!\n");
	//	goto Error;
	//}
	//else {
	//	fprintf(stdout, "Successfully copied back data inside cuda_c\n");
	//}

Error:
	cudaFree(&cuda_a);
	cudaFree(&cuda_b);
	cudaFree(&cuda_c);

	fprintf(stdout, "Successfully cleared cuda buffers\n");

	return cudaStatus;
}
 