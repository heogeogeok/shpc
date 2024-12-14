#include "matmul_single.h"
#include "util.h"

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }
#define TILESIZE 32

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K) {
  __shared__ float tA[TILESIZE][TILESIZE];
  __shared__ float tB[TILESIZE][TILESIZE];

  int i = threadIdx.x;
  int j = threadIdx.y;

  int row = i + blockIdx.x * blockDim.x;
  int col = j + blockIdx.y * blockDim.y;

  int ntile = (K + TILESIZE - 1) / TILESIZE;

  float acc = 0.0;
  for (int t = 0; t < ntile; t++)
  {
    int ti = col * K + TILESIZE * t + i;
    int tj = (TILESIZE * t + j) * N + row;
    
    tA[j][i] = A[ti];
    tB[j][i] = B[tj];

    __syncthreads();

    for (int k = 0; k < TILESIZE; k++)
      acc += tA[j][k] * tB[k][i];

    __syncthreads();
  }
  
  C[col * N + row] = acc;
}

// Array of device (GPU) pointers
static float *a_d;
static float *b_d;
static float *c_d;

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  // Upload A and B matrix to every GPU
  CUDA_CALL(cudaMemcpy(a_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(b_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // Launch kernel on every GPU
  dim3 threads(TILESIZE, TILESIZE);
  dim3 blocks((N + TILESIZE -1) / TILESIZE, (M + TILESIZE -1) / TILESIZE);

  matmul_kernel<<<blocks, threads>>>(a_d, b_d, c_d, M, N, K);

  CUDA_CALL(cudaDeviceSynchronize());

  // Download C matrix from GPUs
  CUDA_CALL(cudaMemcpy(C, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_initialize(int M, int N, int K) {
  
  int num_devices;
  // Only root process do something
  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  if (num_devices <= 0) {
    printf("No CUDA device found. Aborting\n");
    exit(1);
  }

  // Allocate device memory 
  CUDA_CALL(cudaMalloc(&a_d, M * K * sizeof(float)));
  CUDA_CALL(cudaMalloc(&b_d, K * N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&c_d, M * N * sizeof(float)));
}

void matmul_finalize() {

  // Free GPU memory
  CUDA_CALL(cudaFree(a_d));
  CUDA_CALL(cudaFree(b_d));
  CUDA_CALL(cudaFree(c_d));
}
