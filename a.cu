#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

#define IX(i,j,k) ((i) + (N+2)*(j) + (N+2)*(N+2)*(k))
#define SWAP(x0,x) {float *tmp=x0;x0=x;x=tmp;}
#define MAX_VELOCITY 100.0f

const int N = 100;  // Increased grid size
const int SIZE = (N+2)*(N+2)*(N+2);

// Struct to hold simulation parameters
struct SimParams {
    float dt;
    float visc;
    float diff;
};

// CUDA kernel for advection step
__global__ void advect_kernel(float *d, float *d0, float *u, float *v, float *w, SimParams params)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <= N) {
        float x = i - params.dt*N*u[IX(i,j,k)];
        float y = j - params.dt*N*v[IX(i,j,k)];
        float z = k - params.dt*N*w[IX(i,j,k)];
       
        if (x < 0.5f) x = 0.5f; if (x > N + 0.5f) x = N + 0.5f;
        int i0 = (int)x; int i1 = i0 + 1;
        if (y < 0.5f) y = 0.5f; if (y > N + 0.5f) y = N + 0.5f;
        int j0 = (int)y; int j1 = j0 + 1;
        if (z < 0.5f) z = 0.5f; if (z > N + 0.5f) z = N + 0.5f;
        int k0 = (int)z; int k1 = k0 + 1;

        float s1 = x - i0; float s0 = 1 - s1;
        float t1 = y - j0; float t0 = 1 - t1;
        float u1 = z - k0; float u0 = 1 - u1;

        d[IX(i,j,k)] =
            s0 * (t0 * (u0 * d0[IX(i0,j0,k0)] + u1 * d0[IX(i0,j0,k1)]) +
                  t1 * (u0 * d0[IX(i0,j1,k0)] + u1 * d0[IX(i0,j1,k1)])) +
            s1 * (t0 * (u0 * d0[IX(i1,j0,k0)] + u1 * d0[IX(i1,j0,k1)]) +
                  t1 * (u0 * d0[IX(i1,j1,k0)] + u1 * d0[IX(i1,j1,k1)]));
    }
}

// CUDA kernel for diffusion step
__global__ void diffuse_kernel(float *x, float *x0, float diff, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <= N) {
        float a = dt * diff * N * N * N;
        x[IX(i,j,k)] = (x0[IX(i,j,k)] + a * (
            x[IX(i-1,j,k)] + x[IX(i+1,j,k)] +
            x[IX(i,j-1,k)] + x[IX(i,j+1,k)] +
            x[IX(i,j,k-1)] + x[IX(i,j,k+1)]
        )) / (1 + 6 * a);
    }
}

// CUDA kernel for projection step (compute divergence)
__global__ void project_kernel1(float *u, float *v, float *w, float *p, float *div)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 1 && i < N && j > 1 && j < N && k > 1 && k < N) {
        div[IX(i,j,k)] = -0.5f * (
            u[IX(i+1,j,k)] - u[IX(i-1,j,k)] +
            v[IX(i,j+1,k)] - v[IX(i,j-1,k)] +
            w[IX(i,j,k+1)] - w[IX(i,j,k-1)]
        ) / N;
        p[IX(i,j,k)] = 0;
    }
}

// CUDA kernel for projection step (solve pressure)
__global__ void project_kernel2(float *u, float *v, float *w, float *p, float *div)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 1 && i < N && j > 1 && j < N && k > 1 && k < N) {
        p[IX(i,j,k)] = (div[IX(i,j,k)] +
            p[IX(i-1,j,k)] + p[IX(i+1,j,k)] +
            p[IX(i,j-1,k)] + p[IX(i,j+1,k)] +
            p[IX(i,j,k-1)] + p[IX(i,j,k+1)]
        ) / 6.0f;
    }
}

// CUDA kernel for projection step (correct velocities)
__global__ void project_kernel3(float *u, float *v, float *w, float *p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 1 && i < N && j > 1 && j < N && k > 1 && k < N) {
        u[IX(i,j,k)] -= 0.5f * N * (p[IX(i+1,j,k)] - p[IX(i-1,j,k)]);
        v[IX(i,j,k)] -= 0.5f * N * (p[IX(i,j+1,k)] - p[IX(i,j-1,k)]);
        w[IX(i,j,k)] -= 0.5f * N * (p[IX(i,j,k+1)] - p[IX(i,j,k-1)]);
    }
}

// CUDA kernel for vorticity calculation
__global__ void vorticity_kernel(float *u, float *v, float *w, float *vort)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 1 && i < N && j > 1 && j < N && k > 1 && k < N) {
        vort[IX(i,j,k)] = sqrtf(
            powf((w[IX(i,j+1,k)] - w[IX(i,j-1,k)]) - (v[IX(i,j,k+1)] - v[IX(i,j,k-1)]), 2) +
            powf((u[IX(i,j,k+1)] - u[IX(i,j,k-1)]) - (w[IX(i+1,j,k)] - w[IX(i-1,j,k)]), 2) +
            powf((v[IX(i+1,j,k)] - v[IX(i-1,j,k)]) - (u[IX(i,j+1,k)] - u[IX(i,j-1,k)]), 2)
        ) * 0.5f / N;
    }
}

// Host function to set up and run simulation
void fluid_simulation(float *u, float *v, float *w, float *vort, SimParams params, int iterations)
{
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    float *d_u, *d_v, *d_w, *d_u0, *d_v0, *d_w0, *d_vort, *d_p, *d_div;
   
    // Allocate device memory
    cudaMalloc((void**)&d_u, SIZE * sizeof(float));
    cudaMalloc((void**)&d_v, SIZE * sizeof(float));
    cudaMalloc((void**)&d_w, SIZE * sizeof(float));
    cudaMalloc((void**)&d_u0, SIZE * sizeof(float));
    cudaMalloc((void**)&d_v0, SIZE * sizeof(float));
    cudaMalloc((void**)&d_w0, SIZE * sizeof(float));
    cudaMalloc((void**)&d_vort, SIZE * sizeof(float));
    cudaMalloc((void**)&d_p, SIZE * sizeof(float));
    cudaMalloc((void**)&d_div, SIZE * sizeof(float));

    // Copy initial conditions to device
    cudaMemcpy(d_u, u, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    for (int step = 0; step < iterations; step++) {
        // Velocity step
        SWAP(d_u0, d_u); SWAP(d_v0, d_v); SWAP(d_w0, d_w);
       
        // Use different streams for each velocity component
        diffuse_kernel<<<numBlocks, threadsPerBlock, 0, stream1>>>(d_u, d_u0, params.visc, params.dt);
        diffuse_kernel<<<numBlocks, threadsPerBlock, 0, stream2>>>(d_v, d_v0, params.visc, params.dt);
        diffuse_kernel<<<numBlocks, threadsPerBlock, 0, stream3>>>(d_w, d_w0, params.visc, params.dt);
       
        cudaDeviceSynchronize();

        project_kernel1<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_w, d_p, d_div);
        for (int k = 0; k < 20; k++) {
            project_kernel2<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_w, d_p, d_div);
        }
        project_kernel3<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_w, d_p);

        SWAP(d_u0, d_u); SWAP(d_v0, d_v); SWAP(d_w0, d_w);
       
        advect_kernel<<<numBlocks, threadsPerBlock, 0, stream1>>>(d_u, d_u0, d_u0, d_v0, d_w0, params);
        advect_kernel<<<numBlocks, threadsPerBlock, 0, stream2>>>(d_v, d_v0, d_u0, d_v0, d_w0, params);
        advect_kernel<<<numBlocks, threadsPerBlock, 0, stream3>>>(d_w, d_w0, d_u0, d_v0, d_w0, params);
       
        cudaDeviceSynchronize();

        project_kernel1<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_w, d_p, d_div);
        for (int k = 0; k < 20; k++) {
            project_kernel2<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_w, d_p, d_div);
        }
        project_kernel3<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_w, d_p);

        // Calculate vorticity
        vorticity_kernel<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_w, d_vort);
    }

    // Copy results back to host
    cudaMemcpy(u, d_u, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w, d_w, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vort, d_vort, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory and destroy streams
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_w);
    cudaFree(d_u0); cudaFree(d_v0); cudaFree(d_w0);
    cudaFree(d_vort); cudaFree(d_p); cudaFree(d_div);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
}

__global__ void init_complex_conditions(float *u, float *v, float *w, unsigned int seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < N+2 && j < N+2 && k < N+2) {
        curandState state;
        curand_init(seed, IX(i,j,k), 0, &state);

        // Create vortex-like structures
        float x = (float)i / N - 0.5f;
        float y = (float)j / N - 0.5f;
        float z = (float)k / N - 0.5f;
        float r = sqrtf(x*x + y*y + z*z);

        // Base flow
        u[IX(i,j,k)] = 10.0f * sinf(2.0f * M_PI * y) * cosf(2.0f * M_PI * z);
        v[IX(i,j,k)] = 10.0f * sinf(2.0f * M_PI * z) * cosf(2.0f * M_PI * x);
        w[IX(i,j,k)] = 10.0f * sinf(2.0f * M_PI * x) * cosf(2.0f * M_PI * y);

        // Add some randomness
        u[IX(i,j,k)] += 2.0f * (curand_uniform(&state) - 0.5f);
        v[IX(i,j,k)] += 2.0f * (curand_uniform(&state) - 0.5f);
        w[IX(i,j,k)] += 2.0f * (curand_uniform(&state) - 0.5f);

        // Add vortex-like structure
        float vortex_strength = 20.0f * expf(-r*r / 0.1f);
        u[IX(i,j,k)] += vortex_strength * (-y + z);
        v[IX(i,j,k)] += vortex_strength * (x - z);
        w[IX(i,j,k)] += vortex_strength * (-x + y);
    }
}

// Host function to initialize the simulation
void init_simulation(float *u, float *v, float *w, float *vort)
{
    float *d_u, *d_v, *d_w;
    cudaMalloc((void**)&d_u, SIZE * sizeof(float));
    cudaMalloc((void**)&d_v, SIZE * sizeof(float));
    cudaMalloc((void**)&d_w, SIZE * sizeof(float));

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((N+2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N+2 + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (N+2 + threadsPerBlock.z - 1) / threadsPerBlock.z);

    unsigned int seed = time(NULL);
    init_complex_conditions<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_w, seed);

    cudaMemcpy(u, d_u, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w, d_w, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);

    // Initialize vorticity to zero
    memset(vort, 0, SIZE * sizeof(float));
}

int main()
{
    float *u = new float[SIZE];
    float *v = new float[SIZE];
    float *w = new float[SIZE];
    float *vort = new float[SIZE];

    SimParams params;
    params.dt = 0.1f;
    params.visc = 0.0001f;
    params.diff = 0.0f;

    int iterations = 200;

    init_simulation(u, v, w, vort);
    fluid_simulation(u, v, w, vort, params, iterations);

    // Output results (you can modify this to save to a file or visualize)
    for (int i = 1; i <= N; i += N/8) {
        for (int j = 1; j <= N; j += N/8) {
            for (int k = 1; k <= N; k += N/8) {
                printf("Velocity at (%d,%d,%d): (%f, %f, %f), Vorticity: %f\n",
                       i, j, k, u[IX(i,j,k)], v[IX(i,j,k)], w[IX(i,j,k)], vort[IX(i,j,k)]);
            }
        }
    }

    delete[] u;
    delete[] v;
    delete[] w;
    delete[] vort;

    return 0;
}
