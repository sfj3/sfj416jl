#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

#define IX(i,j) ((i)+(N+2)*(j))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}
#define MAX_VELOCITY 100.0f

const int N = 256;
const int SIZE = (N+2)*(N+2);

// CUDA kernel for fluid simulation step
__global__ void fluid_step_kernel(float *u, float *v, float *u0, float *v0, float visc, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i <= N && j > 0 && j <= N) {
        int idx = IX(i,j);
        
        // Apply viscosity and external forces
        u[idx] = u0[idx] + visc * (u0[IX(i+1,j)] + u0[IX(i-1,j)] + u0[IX(i,j+1)] + u0[IX(i,j-1)] - 4*u0[idx]) * dt;
        v[idx] = v0[idx] + visc * (v0[IX(i+1,j)] + v0[IX(i-1,j)] + v0[IX(i,j+1)] + v0[IX(i,j-1)] - 4*v0[idx]) * dt;

        // Ensure velocity doesn't exceed maximum
        float speed = sqrtf(u[idx]*u[idx] + v[idx]*v[idx]);
        if (speed > MAX_VELOCITY) {
            u[idx] *= MAX_VELOCITY / speed;
            v[idx] *= MAX_VELOCITY / speed;
        }
    }
}

// CUDA kernel for calculating vorticity (curl of velocity field)
__global__ void vorticity_kernel(float *u, float *v, float *vort)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N+1 && j > 0 && j < N+1) {
        int idx = IX(i,j);
        vort[idx] = (v[IX(i+1,j)] - v[IX(i-1,j)] - u[IX(i,j+1)] + u[IX(i,j-1)]) * 0.5f;
    }
}

// Host function to set up and run simulation
void fluid_simulation(float *u, float *v, float *u0, float *v0, float *vort, float visc, float dt, int iterations)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    float *d_u, *d_v, *d_u0, *d_v0, *d_vort;
    
    // Allocate device memory
    cudaMalloc((void**)&d_u, SIZE * sizeof(float));
    cudaMalloc((void**)&d_v, SIZE * sizeof(float));
    cudaMalloc((void**)&d_u0, SIZE * sizeof(float));
    cudaMalloc((void**)&d_v0, SIZE * sizeof(float));
    cudaMalloc((void**)&d_vort, SIZE * sizeof(float));

    // Copy initial conditions to device
    cudaMemcpy(d_u, u, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    for (int step = 0; step < iterations; step++) {
        // Swap velocity fields
        SWAP(d_u0, d_u); SWAP(d_v0, d_v);

        // Perform fluid simulation step
        fluid_step_kernel<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_u0, d_v0, visc, dt);

        // Calculate vorticity
        vorticity_kernel<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_vort);

        // You can add more kernels here for pressure calculation, advection, etc.
    }

    // Copy results back to host
    cudaMemcpy(u, d_u, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vort, d_vort, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_u0); cudaFree(d_v0); cudaFree(d_vort);
}

// Host function to initialize the simulation
void init_simulation(float *u, float *v, float *vort, float initial_wind_x, float initial_wind_y)
{
    for (int i = 0; i < N+2; i++) {
        for (int j = 0; j < N+2; j++) {
            u[IX(i,j)] = initial_wind_x;
            v[IX(i,j)] = initial_wind_y;
            vort[IX(i,j)] = 0.0f;
        }
    }
}

int main()
{
    float *u = new float[SIZE];
    float *v = new float[SIZE];
    float *vort = new float[SIZE];

    float initial_wind_x = 10.0f;
    float initial_wind_y = 5.0f;
    float Reynolds = 1000.0f;
    float visc = 1.0f / Reynolds;
    float dt = 0.1f;
    int iterations = 1000;

    init_simulation(u, v, vort, initial_wind_x, initial_wind_y);
    fluid_simulation(u, v, u, v, vort, visc, dt, iterations);

    // Output results (you can modify this to save to a file or visualize)
    for (int i = 1; i <= N; i += N/10) {
        for (int j = 1; j <= N; j += N/10) {
            printf("Velocity at (%d,%d): (%f, %f), Vorticity: %f\n", 
                   i, j, u[IX(i,j)], v[IX(i,j)], vort[IX(i,j)]);
        }
    }

    delete[] u;
    delete[] v;
    delete[] vort;

    return 0;
}