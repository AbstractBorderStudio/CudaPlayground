- Includere le librerie
    
    ```cpp
    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"
    
    #include <stdio.h>
    ```
    
- Fare check degli errori ad ogni operazione
    
    Ogni funzione cuda ritorna lo stato del risultato. Fare un check sullo stato per vedere che tutto sia andato a buon fine:
    
    ```cpp
    // usare
    cudaError_t cudaStatus;
    
    ...
    
    cudaStatus = someCudaFunction(...);
    if (cudaStatus != cudaSuccess) {
    	// error occurred
    	return 1;
    }
    ```
    
- Creare dei buffer per la GPU
    
    ```cpp
    // dato da compiare nella gpu
    const int A[size] = { 1, 2, ... };
    
    // esempio voglio creare dei buffer per degli array
    // crea dei puntatori
    int *buffer_A = 0;
    
    /* uso cudaMalloc per allocare memoria nella gpu
    devo creare un vettore di int * size */
    
    // prima passo un puntatore a void** sulla variabile
    // poi la dimensione da allocare
    cudaStatus = cudaMalloc((void**)&buffer_A, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
    	// de-alloca le variabili.	
    	return 1;
    }
    ```
    
- Goto
    
    nelle funzioni c++ e cuda, viene usata la sintassi goto (come in assembly) per salta da un punto ad un altro del codice.
    
    ```cpp
    cudaError_t myFunction() {
    	cudaStatus = someCudaFunction(...);
    	if (cudaStatus != CudaSuccess) {
    		// errore!
    		goto Error;
    	}
    
    // si usa la keyword che identifica il punto del codice in cui saltare.
    Error:
    	// error handling
    	
    	return cudaStatus;
    }
    ```