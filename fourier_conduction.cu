#include <stdio.h>
#include <cuda_runtime.h>

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void heatConductionKernel(float *T_old, float *T_new, float T_hot, float T_cold, float alpha, float dx, float dt, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        T_new[idx] = T_cold; // Cold end boundary condition
    } else if (idx == N - 1) {
        T_new[idx] = T_hot; // Hot end boundary condition
    } else if (idx < N) {
        T_new[idx] = T_old[idx] + alpha * dt / (dx * dx) * (T_old[idx + 1] - 2 * T_old[idx] + T_old[idx - 1]);
    }
}

int main() {
    const int N = 1200; // Number of spatial points
    const float L = 1.0f; // Length of the rod in meters
    const float T_hot = 500.0f; // Hot end temperature in Kelvin
    const float T_cold = 300.0f; // Cold end temperature in Kelvin
    const float k = 1.5f; // Thermal conductivity in W/(m·K)
    const float rho = 8000.0f; // Density in kg/m^3
    const float c = 500.0f; // Specific heat capacity in J/(kg·K)
    const float alpha = k / (rho * c); // Thermal diffusivity
    const float total_time = 60.0f; // Total simulation time in seconds
    const int time_steps = 10; // Number of time steps to display

    const float dx = L / (N - 1);
    const float dt = 0.5 * dx * dx / alpha; // Time step for stability

    float *d_T_old, *d_T_new; // Device arrays for temperatures
    float *h_T; // Host array for temperatures

    // Allocate memory on the CPU
    h_T = (float*)malloc(N * sizeof(float));

    // Allocate memory on the GPU
    cudaCheckError(cudaMalloc(&d_T_old, N * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_T_new, N * sizeof(float)));

    // Initialize temperatures on the host
    for (int i = 0; i < N; ++i) {
        h_T[i] = T_cold + (T_hot - T_cold) * (i / (float)(N - 1));
    }

    // Copy initial temperatures to the device
    cudaCheckError(cudaMemcpy(d_T_old, h_T, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_T_new, h_T, N * sizeof(float), cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Simulating 1D heat conduction\n");
    printf("Rod length: %.4f m, T_hot: %.2f K, T_cold: %.2f K\n", L, T_hot, T_cold);
    printf("Thermal conductivity: %.2f W/(m·K), Thermal diffusivity: %.2e m^2/s\n", k, alpha);
    printf("Spatial steps: %d, Total time: %.2f s, Time steps displayed: %d\n", N, total_time, time_steps);

    // Time stepping
    for (int step = 0; step <= time_steps; ++step) {
        float t = step * total_time / time_steps;

        int num_iterations = (int)(t / dt);
        for (int n = 0; n < num_iterations; ++n) {
            heatConductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_T_old, d_T_new, T_hot, T_cold, alpha, dx, dt, N);
            cudaCheckError(cudaGetLastError());
            cudaCheckError(cudaMemcpy(d_T_old, d_T_new, N * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        // Copy temperatures back to CPU
        cudaCheckError(cudaMemcpy(h_T, d_T_old, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Print results
        printf("\nTemperature distribution at t = %.2f s:\n", t);
        for (int i = 0; i < N; i += N / 10) {
            float x = i * L / (N - 1);
            printf("x = %.3f m: T = %.3f K\n", x, h_T[i]);
        }

        // Calculate heat flux at the hot end
        float heat_flux = -k * (h_T[1] - h_T[0]) / dx;
        printf("Heat flux at the cold end: %.4f W/m^2\n", heat_flux);
    }

    // Free memory
    free(h_T);
    cudaCheckError(cudaFree(d_T_old));
    cudaCheckError(cudaFree(d_T_new));

    return 0;
}
