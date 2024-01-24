# CudaPlayground
Cuda Samples to mess around with Cuda libs and GL

- Code boilerplate for only-cuda code

```cpp
// system includes
#include <stdio.h>

// cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main() {
	cudaError_t cudaStatus;

    // add code here //
	

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
```