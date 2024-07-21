#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA kernel for 1D heat conduction
__global__ void heatConductionKernel(float *T, float *T_new, float alpha, float dx, float dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < N - 1) {
        T_new[idx] = T[idx] + alpha * dt / (dx * dx) * (T[idx+1] - 2*T[idx] + T[idx-1]);
    }
}

int main() {
    const int N = 100; // Number of spatial points
    const float L = 1.0f; // Length of the rod in meters
    const float T_hot = 1000.0f; // Hot end temperature in Kelvin
    const float T_cold = 280.0f; // Cold end temperature in Kelvin
    const float k = 1.5f; // Thermal conductivity in W/(m·K)
    const float rho = 8000.0f; // Density in kg/m^3
    const float c = 500.0f; // Specific heat capacity in J/(kg·K)
    const float alpha = k / (rho * c); // Thermal diffusivity
    const float dx = L / (N - 1); // Spatial step
    const float dt = 0.1f * dx * dx / alpha; // Time step (for stability, dt <= 0.5 * dx^2 / alpha)
    const float total_time = 10.0f; // Total simulation time in seconds
    const int time_steps = ceil(total_time / dt);

    float *d_T, *d_T_new; // Device arrays for temperatures
    float *h_T; // Host array for temperatures

    // Allocate memory on the CPU
    h_T = (float*)malloc(N * sizeof(float));

    // Allocate memory on the GPU
    cudaCheckError(cudaMalloc(&d_T, N * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_T_new, N * sizeof(float)));

    // Initialize temperature distribution
    for (int i = 0; i < N; i++) {
        h_T[i] = T_cold + (T_hot - T_cold) * (1 - (float)i / (N - 1)); // Linear initial distribution
    }

    // Copy initial temperatures to GPU
    cudaCheckError(cudaMemcpy(d_T, h_T, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_T_new, h_T, N * sizeof(float), cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Simulating 1D heat conduction\n");
    printf("Rod length: %.4f m, T_hot: %.2f K, T_cold: %.2f K\n", L, T_hot, T_cold);
    printf("Thermal conductivity: %.2f W/(m·K), Thermal diffusivity: %.2e m^2/s\n", k, alpha);
    printf("Spatial steps: %d, Time step: %.6f s, Total time: %.2f s\n", N, dt, total_time);

    // Time stepping
    for (int step = 0; step < time_steps; step++) {
        heatConductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_T, d_T_new, alpha, dx, dt, N);
        cudaCheckError(cudaGetLastError());

        // Swap pointers
        float *temp = d_T;
        d_T = d_T_new;
        d_T_new = temp;

        // Reset boundary conditions
        cudaCheckError(cudaMemcpy(d_T, &T_hot, sizeof(float), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_T + N - 1, &T_cold, sizeof(float), cudaMemcpyHostToDevice));
    }

    // Copy final temperatures back to CPU
    cudaCheckError(cudaMemcpy(h_T, d_T, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    printf("\nFinal temperature distribution:\n");
    for (int i = 0; i < N; i += N/10) {
        printf("Position %.3f m: T = %.3f K\n", i * dx, h_T[i]);
    }

    // Calculate heat flux at the hot end
    float heat_flux = -k * (h_T[1] - h_T[0]) / dx;
    printf("\nHeat flux at the hot end: %.4f W/m^2\n", heat_flux);

    // Free memory
    free(h_T);
    cudaCheckError(cudaFree(d_T));
    cudaCheckError(cudaFree(d_T_new));

    return 0;
}
