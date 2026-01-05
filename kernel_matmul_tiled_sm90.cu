#include <cuda_runtime.h>

// Professional Tiled Matrix Multiplication
// Optimization: Shared memory prevents global memory bottlenecks
__global__ void tiled_matmul_kernel(float* A, float* B, float* C, int N) {
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    float sum = 0;

    for (int t = 0; t < (N/32); ++t) {
        // Coalesced loading into Shared Memory
        tileA[threadIdx.y][threadIdx.x] = A[row * N + t * 32 + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * 32 + threadIdx.y) * N + col];
        __syncthreads(); // Critical Synchronization

        for (int i = 0; i < 32; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
