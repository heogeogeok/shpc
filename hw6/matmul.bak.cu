#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

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
#define NGPU 4

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
  __shared__ float tA[TILESIZE][TILESIZE + 16];
  __shared__ float tB[TILESIZE][TILESIZE + 16];

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
    
    if (ti < M * K)
      tA[j][i] = A[ti];
    else
      tA[j][i] = 0.0;
    if (tj < K * N)
      tB[j][i] = B[tj];
    else
      tB[j][i] = 0.0;

    __syncthreads();

    for (int k = 0; k < TILESIZE; k++)
      acc += tA[j][k] * tB[k][i];

    __syncthreads();
  }
  
  if (col < M && row < N) 
    C[col * N + row] = acc;
}

int ngpu;
int Nbegin, Nend;
static int Mbegin[NGPU], Mend[NGPU];
static float *a_d[NGPU], *b_d[NGPU], *c_d[NGPU];
static int mpi_rank, mpi_world_size;

static cudaStream_t streams[NGPU], streams_mem[NGPU];
cudaEvent_t event_htod[NGPU], event_dtoh[NGPU];

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  #pragma omp parallel for num_threads(ngpu)
  for (int i = 0; i < ngpu; i++) {
    CUDA_CALL(cudaMemcpyAsync(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice, streams_mem[i])); }
  
  #pragma omp parallel for num_threads(ngpu)
  for (int i = 0; i < ngpu; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMemcpyAsync(a_d[i], &A[Mbegin[i] * K], (Mend[i] - Mbegin[i]) * K * sizeof(float), cudaMemcpyHostToDevice, streams_mem[i]));

    CUDA_CALL(cudaEventRecord(event_htod[i], streams_mem[i]));
    CUDA_CALL(cudaStreamWaitEvent(streams[i], event_htod[i], 0));

    dim3 blockDim(TILESIZE, TILESIZE);
    dim3 gridDim((N + TILESIZE - 1) / TILESIZE, (Mend[i] - Mbegin[i] + TILESIZE - 1) / TILESIZE);
    matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(a_d[i], b_d[i], c_d[i], Mend[i] - Mbegin[i], N, K);

    CUDA_CALL(cudaEventRecord(event_dtoh[i], streams[i]));
    CUDA_CALL(cudaStreamWaitEvent(streams_mem[i], event_dtoh[i], 0));

    CUDA_CALL(cudaMemcpyAsync(&C[Mbegin[i] * N], c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float), cudaMemcpyDeviceToHost, streams_mem[i]));
  }

  for (int i = 0; i < ngpu; i++) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
  }
}

void matmul_initialize(int M, int N, int K) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  CUDA_CALL(cudaGetDeviceCount(&ngpu));
  
  Nbegin = M * mpi_rank / mpi_world_size;
  Nend = M * (mpi_rank + 1) / mpi_world_size;
  
  for (int i = 0; i < ngpu; i++) {
    Mbegin[i] = Nbegin + (Nend - Nbegin) * i / ngpu;
    Mend[i] = Nbegin + (Nend - Nbegin) * (i + 1) / ngpu;
    if (i == ngpu - 1) Mend[i] = Nend;
  }

  for (int i = 0; i < ngpu; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaStreamCreate(&streams[i]));
    CUDA_CALL(cudaStreamCreate(&streams_mem[i]));
    CUDA_CALL(cudaEventCreate(&event_htod[i]));
    CUDA_CALL(cudaEventCreate(&event_dtoh[i]));
  }

  for (int i = 0; i < ngpu; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
  }
}

void matmul_finalize() {
  for (int i = 0; i < ngpu; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaFree(a_d[i]));
    CUDA_CALL(cudaFree(b_d[i]));
    CUDA_CALL(cudaFree(c_d[i]));
    CUDA_CALL(cudaStreamDestroy(streams[i]));
    CUDA_CALL(cudaStreamDestroy(streams_mem[i]));
    CUDA_CALL(cudaEventDestroy(event_htod[i]));
    CUDA_CALL(cudaEventDestroy(event_dtoh[i]));
  }
}