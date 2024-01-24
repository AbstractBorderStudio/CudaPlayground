/*
	Daniel Bologna 2024
*/

// system includes
#include <stdio.h>

// cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main() {
	cudaError_t cudaStatus;

	// create matrices
	

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