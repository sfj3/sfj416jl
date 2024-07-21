#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA kernel for heat conduction calculation
__global__ void heatConductionKernel(float *T, float T1, float T2, float L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = 16; // Number of sections

    if (idx < N) {
        float x = (idx + 0.5f) * (L / N); // x at the center of each section
        T[idx] = (T2 - T1) / L * x + T1;
        printf("Section %d: x = %.3f, T = %.3f\n", idx, x, T[idx]);
    }
}

int main() {
    const int N = 16; // Number of sections
    float L = 1.0f; // Length of the plate
    float T1 = 100.0f; // Temperature at x = 0
    float T2 = 500.0f; // Temperature at x = L
    
    float *d_T; // Device array for temperatures
    float h_T[N]; // Host array for temperatures

    // Allocate memory on the GPU
    cudaCheckError(cudaMalloc(&d_T, N * sizeof(float)));

    // Define block and grid dimensions
    int threadsPerBlock = 16; // Use 16 threads per block
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Solving steady-state heat conduction over %d sections\n", N);
    printf("Plate length: %.2f, T1: %.2f, T2: %.2f\n", L, T1, T2);
    printf("Launching kernel with %d blocks, each with %d threads\n", blocksPerGrid, threadsPerBlock);

    // Launch the kernel
    heatConductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_T, T1, T2, L);
    cudaCheckError(cudaGetLastError());

    // Wait for GPU to finish
    cudaCheckError(cudaDeviceSynchronize());

    // Copy the result back to the CPU
    cudaCheckError(cudaMemcpy(h_T, d_T, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the results
    printf("\nFinal Results:\n");
    for (int i = 0; i < N; i++) {
        printf("Section %d: T = %.3f\n", i, h_T[i]);
    }

    // Free GPU memory
    cudaCheckError(cudaFree(d_T));

    return 0;
}
