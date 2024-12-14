#include "matmul.h"
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

#define MAX_NUM_GPU 4
#define TILESIZE 32
#define MAX_NUM_STREAM 16

int num_streams = 16;
int num_devices;

cudaStream_t streams[MAX_NUM_GPU][MAX_NUM_STREAM];

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

static float *a_d[MAX_NUM_GPU];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  #pragma omp parallel for num_threads(num_devices)
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    int sizeS = ((Mend[i] - Mbegin[i]) / num_streams) * K * sizeof(float);

    for (int j = 0; j < num_streams; j++) {
      int stream_offset = ((Mend[i] - Mbegin[i]) / num_streams) * j;

      CUDA_CALL(cudaMemcpyAsync((void *) ((size_t) a_d[i] + sizeS * j),
                                (void *) ((size_t) A + (Mbegin[i] + stream_offset) * K * sizeof(float)),
                                sizeS, cudaMemcpyHostToDevice, streams[i][j]));

      if (j == 0) { 
        CUDA_CALL(cudaMemcpyAsync(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice, streams[i][j]));
      }

      dim3 threads(TILESIZE, TILESIZE);
      dim3 blocks(N / TILESIZE, (Mend[i] - Mbegin[i]) / TILESIZE / MAX_NUM_STREAM);
      
      matmul_kernel<<<blocks, threads, 0, streams[i][j]>>>(
          (float *) ((size_t) a_d[i] + sizeS * j),
          b_d[i],
          (float *) ((size_t) c_d[i] + sizeS * j),
          (Mend[i] - Mbegin[i]) / num_streams,
          N, K);

      CUDA_CALL(cudaMemcpyAsync((void *) ((size_t) C + (Mbegin[i] + stream_offset) * N * sizeof(float)),
                                (void *) ((size_t) c_d[i] + sizeS * j),
                                sizeS, cudaMemcpyDeviceToHost, streams[i][j]));
    }

    for (int j = 0; j < num_streams; j++) {
      CUDA_CALL(cudaSetDevice(i));
      CUDA_CALL(cudaStreamSynchronize(streams[i][j]));
    }
  }
}

void matmul_initialize(int M, int N, int K) {

  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  printf("Using %d devices\n", num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));

    // Try printing more detailed information here
    printf("GPU %d: %s\n", i, prop.name);
  }

  if (num_devices <= 0) {
    printf("No CUDA device found. Aborting\n");
    exit(1);
  }

  for (int i = 0; i < num_devices; i++) {
    Mbegin[i] = (M / num_devices) * i;
    Mend[i] = (M / num_devices) * (i + 1);
  }
  Mend[num_devices - 1] = M;

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    for (int j = 0; j < num_streams; j++) {
      CUDA_CALL(cudaStreamCreate(&streams[i][j]));
      }
  }
}

void matmul_finalize() {

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));

    for (int j = 0; j < num_streams; j++)
      CUDA_CALL(cudaStreamDestroy(streams[i][j]));
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaFree(a_d[i]));
    CUDA_CALL(cudaFree(b_d[i]));
    CUDA_CALL(cudaFree(c_d[i]));
  }
}
