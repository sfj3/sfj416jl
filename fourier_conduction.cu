#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define PI 3.14159265358979323846

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Corrected CUDA kernel for 1D heat conduction analytical solution
__global__ void heatConductionKernel(float *T, float T_hot, float T_cold, float L, float alpha, float t, int N, int n_terms) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = idx * L / (N - 1);
        float sum = 0.0f;
        for (int n = 1; n <= n_terms; n++) {
            sum += 1.0f / n * sinf(n * PI * x / L) * expf(-alpha * n * n * PI * PI * t / (L * L));
        }
        T[idx] = T_cold + (T_hot - T_cold) * (1.0f - x / L) + 
                 2.0f * (T_hot - T_cold) / PI * sum;
    }
}

int main() {
    const int N = 1000; // Number of spatial points
    const float L = 1.0f; // Length of the rod in meters
    const float T_hot = 600.0f; // Hot end temperature in Kelvin
    const float T_cold = 300.0f; // Cold end temperature in Kelvin
    const float k = 1.5f; // Thermal conductivity in W/(m·K)
    const float rho = 8000.0f; // Density in kg/m^3
    const float c = 500.0f; // Specific heat capacity in J/(kg·K)
    const float alpha = k / (rho * c); // Thermal diffusivity
    const float total_time = 600.0f; // Total simulation time in seconds
    const int time_steps = 10; // Number of time steps to display
    const int n_terms = 100; // Number of terms in the Fourier series

    float *d_T; // Device array for temperatures
    float *h_T; // Host array for temperatures

    // Allocate memory on the CPU
    h_T = (float*)malloc(N * sizeof(float));

    // Allocate memory on the GPU
    cudaCheckError(cudaMalloc(&d_T, N * sizeof(float)));

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Simulating 1D heat conduction\n");
    printf("Rod length: %.4f m, T_hot: %.2f K, T_cold: %.2f K\n", L, T_hot, T_cold);
    printf("Thermal conductivity: %.2f W/(m·K), Thermal diffusivity: %.2e m^2/s\n", k, alpha);
    printf("Spatial steps: %d, Total time: %.2f s, Time steps displayed: %d\n", N, total_time, time_steps);

    // Time stepping
    for (int step = 0; step <= time_steps; step++) {
        float t = step * total_time / time_steps;
        
        heatConductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_T, T_hot, T_cold, L, alpha, t, N, n_terms);
        cudaCheckError(cudaGetLastError());

        // Copy temperatures back to CPU
        cudaCheckError(cudaMemcpy(h_T, d_T, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Print results
        printf("\nTemperature distribution at t = %.2f s:\n", t);
        for (int i = 0; i < N; i += N/10) {
            float x = i * L / (N - 1);
            printf("x = %.3f m: T = %.3f K\n", x, h_T[i]);
        }

        // Calculate heat flux at the hot end
        float heat_flux = -k * (h_T[1] - h_T[0]) / (L / (N - 1));
        printf("Heat flux at the hot end: %.4f W/m^2\n", heat_flux);
    }

    // Free memory
    free(h_T);
    cudaCheckError(cudaFree(d_T));

    return 0;
}
