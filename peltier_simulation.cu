#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

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

// CUDA kernel for 1D heat conduction numerical solution
__global__ void heatConductionKernel(float *T, float T_hot, float T_cold, float L, float alpha, float dt, float dx, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < N - 1) {
        float T_new = T[idx] + alpha * dt / (dx * dx) * (T[idx+1] - 2*T[idx] + T[idx-1]);
        T[idx] = T_new;
    }
}

// Structure to hold Peltier device properties
typedef struct {
    float area;
    float thickness;
    float k;
    float seebeck;
    float electrical_resistivity;
    float contact_resistance;
} PeltierDevice;

// Function to calculate Peltier device performance
void calculatePeltierPerformance(float T_hot, float T_cold, PeltierDevice device, float *power_out, float *efficiency) {
    float delta_T = T_hot - T_cold;
    
    // Calculate electrical resistance
    float electrical_resistance = device.electrical_resistivity * device.thickness / device.area + device.contact_resistance;
    
    // Calculate optimal current
    float optimal_current = device.seebeck * delta_T / (2 * electrical_resistance);
    
    // Calculate heat flow
    float q_hot = device.seebeck * T_hot * optimal_current - 0.5 * electrical_resistance * optimal_current * optimal_current + device.k * device.area * delta_T / device.thickness;
    
    // Calculate power output
    *power_out = device.seebeck * delta_T * optimal_current - electrical_resistance * optimal_current * optimal_current;
    
    // Calculate efficiency
    *efficiency = *power_out / q_hot;
}

int main() {
    const int N = 1000; // Number of spatial points
    const float L = 1.1f; // Length of the rod in meters
    const float T_hot = 500.0f; // Hot end temperature in Kelvin
    const float T_cold = 300.0f; // Cold end temperature in Kelvin
    const float k = 1.5f; // Thermal conductivity of the rod in W/(m·K)
    const float rho = 8000.0f; // Density of the rod in kg/m^3
    const float c = 500.0f; // Specific heat capacity of the rod in J/(kg·K)
    const float alpha = k / (rho * c); // Thermal diffusivity of the rod
    const float total_time = 10.0f; // Total simulation time in seconds
    const int time_steps = 1000; // Number of time steps
    const float dt = total_time / time_steps; // Time step size
    const float dx = L / (N - 1); // Spatial step size

    PeltierDevice device = {
        .area = 0.0004f, // 2cm x 2cm
        .thickness = 0.004f, // 4mm
        .k = 0.8f, // W/(m·K)
        .seebeck = 0.0002f, // V/K
        .electrical_resistivity = 1e-5f, // Ohm·m
        .contact_resistance = 1e-5f // Ohm
    };

    float *d_T; // Device array for temperatures
    float *h_T; // Host array for temperatures

    // Allocate memory on the CPU
    h_T = (float*)malloc(N * sizeof(float));

    // Initialize temperature distribution
    for (int i = 0; i < N; i++) {
        h_T[i] = T_cold + (T_hot - T_cold) * i / (N - 1);
    }

    // Allocate memory on the GPU
    cudaCheckError(cudaMalloc(&d_T, N * sizeof(float)));
    cudaCheckError(cudaMemcpy(d_T, h_T, N * sizeof(float), cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Refined 1D heat conduction simulation with Peltier devices\n");
    printf("Rod length: %.4f m, T_hot: %.2f K, T_cold: %.2f K\n", L, T_hot, T_cold);
    printf("Thermal conductivity: %.2f W/(m·K), Thermal diffusivity: %.2e m^2/s\n", k, alpha);
    printf("Spatial steps: %d, Total time: %.2f s, Time steps: %d\n", N, total_time, time_steps);
    printf("Peltier device area: %.4f m^2, thickness: %.4f m, k: %.2f W/(m·K)\n", device.area, device.thickness, device.k);

    printf("\nTime (s) | T_hot (K) | T_cold (K) | Delta T (K) | Power Out (W) | Efficiency\n");
    printf("--------------------------------------------------------------------------\n");

    // Time stepping
    for (int step = 0; step <= time_steps; step++) {
        float t = step * dt;
        
        // Update temperature distribution
        heatConductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_T, T_hot, T_cold, L, alpha, dt, dx, N);
        cudaCheckError(cudaGetLastError());

        if (step % (time_steps / 10) == 0) {
            // Copy temperatures back to CPU
            cudaCheckError(cudaMemcpy(h_T, d_T, N * sizeof(float), cudaMemcpyDeviceToHost));

            float power_out, efficiency;
            calculatePeltierPerformance(h_T[N-1], h_T[0], device, &power_out, &efficiency);

            // Print results
            printf("%.2f     | %.2f    | %.2f     | %.2f      | %.4f        | %.4f\n", 
                   t, h_T[N-1], h_T[0], h_T[N-1] - h_T[0], power_out, efficiency);
        }
    }

    // Free memory
    free(h_T);
    cudaCheckError(cudaFree(d_T));

    return 0;
}
