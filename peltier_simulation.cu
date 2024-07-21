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

// CUDA kernel for 1D heat conduction analytical solution
__global__ void heatConductionKernel(float *T, float T_hot, float T_cold, float L, float alpha, float t, int N, int n_terms) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = idx * L / (N - 1);
        float sum = 0;
        for (int n = 1; n <= n_terms; n++) {
            float term = sin(n * PI * x / L) * exp(-alpha * n * n * PI * PI * t / (L * L)) / n;
            sum += term;
        }
        T[idx] = T_cold + (T_hot - T_cold) * (1 - x / L - 2 / PI * sum);
    }
}

// Function to calculate Peltier power
float calculatePeltierPower(float *T, float L, float k, int N, float peltier_area, float peltier_thickness, float peltier_k) {
    float heat_flux = -k * (T[1] - T[0]) / (L / (N - 1));
    float heat_throughput = heat_flux * peltier_area;
    
    // Calculate thermal resistance of the Peltier device
    float thermal_resistance = peltier_thickness / (peltier_k * peltier_area);
    
    // Calculate temperature difference across the Peltier device
    float delta_T = heat_throughput * thermal_resistance;
    
    // Calculate COP based on temperature difference
    // This is a simplified model; actual COP depends on many factors
    float T_cold = T[0];
    float T_hot = T_cold + delta_T;
    float COP = T_cold / (T_hot - T_cold);
    
    // Limit COP to a realistic range
    COP = fmaxf(0.1f, fminf(COP, 3.0f));
    
    // Calculate Peltier power consumption
    float peltier_power = heat_throughput / COP;
    
    return peltier_power;
}

int main() {
    const int N = 100; // Number of spatial points
    const float L = 1.0f; // Length of the rod in meters
    const float T_hot = 600.0f; // Hot end temperature in Kelvin
    const float T_cold = 280.0f; // Cold end temperature in Kelvin
    const float k = 1.5f; // Thermal conductivity of the rod in W/(m·K)
    const float rho = 8000.0f; // Density of the rod in kg/m^3
    const float c = 500.0f; // Specific heat capacity of the rod in J/(kg·K)
    const float alpha = k / (rho * c); // Thermal diffusivity of the rod
    const float total_time = 10.0f; // Total simulation time in seconds
    const int time_steps = 10; // Number of time steps to display
    const int n_terms = 100; // Number of terms in the Fourier series
    const float peltier_area = 0.01f; // Peltier device area in m^2 (10cm x 10cm)
    const float peltier_thickness = 0.004f; // Peltier device thickness in m (4mm)
    const float peltier_k = 0.8f; // Thermal conductivity of Peltier device in W/(m·K)

    float *d_T; // Device array for temperatures
    float *h_T; // Host array for temperatures

    // Allocate memory on the CPU
    h_T = (float*)malloc(N * sizeof(float));

    // Allocate memory on the GPU
    cudaCheckError(cudaMalloc(&d_T, N * sizeof(float)));

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Simulating 1D heat conduction with Peltier device\n");
    printf("Rod length: %.4f m, T_hot: %.2f K, T_cold: %.2f K\n", L, T_hot, T_cold);
    printf("Thermal conductivity: %.2f W/(m·K), Thermal diffusivity: %.2e m^2/s\n", k, alpha);
    printf("Spatial steps: %d, Total time: %.2f s, Time steps displayed: %d\n", N, total_time, time_steps);
    printf("Peltier device area: %.4f m^2, thickness: %.4f m, k: %.2f W/(m·K)\n", peltier_area, peltier_thickness, peltier_k);

    printf("\nTime (s) | Heat Flux (W/m^2) | Heat Throughput (W) | Peltier Power (W) | COP\n");
    printf("-------------------------------------------------------------------------\n");

    // Time stepping
    for (int step = 0; step <= time_steps; step++) {
        float t = step * total_time / time_steps;
        
        heatConductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_T, T_hot, T_cold, L, alpha, t, N, n_terms);
        cudaCheckError(cudaGetLastError());

        // Copy temperatures back to CPU
        cudaCheckError(cudaMemcpy(h_T, d_T, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Calculate heat flux at the hot end
        float heat_flux = -k * (h_T[1] - h_T[0]) / (L / (N - 1));

        // Calculate heat throughput
        float heat_throughput = heat_flux * peltier_area;

        // Calculate Peltier power
        float peltier_power = calculatePeltierPower(h_T, L, k, N, peltier_area, peltier_thickness, peltier_k);

        // Calculate COP
        float COP = heat_throughput / peltier_power;

        // Print results
        printf("%.2f     | %.4f          | %.4f             | %.4f           | %.4f\n", 
               t, heat_flux, heat_throughput, peltier_power, COP);
    }

    // Free memory
    free(h_T);
    cudaCheckError(cudaFree(d_T));

    return 0;
}
