#include "layer.h"

#define ceil_div(x, y) (((x) + (y) - 1) / (y))

/** BLOCK SIZE **/
#define CONV1D_K3_OC 16
#define CONV1D_K3_C 8
#define CONV1D_K3_OS 8

#define CONV1D_K5_OC 8
#define CONV1D_K5_C 4
#define CONV1D_K5_OS 32

#define CONV1D_K7_OC 8
#define CONV1D_K7_C 4
#define CONV1D_K7_OS 32

#define CONV1D_K9_OC 8
#define CONV1D_K9_C 4
#define CONV1D_K9_OS 32

#define LINEAR_BM 4

#define LINEAR_RELU_BM 16
#define LINEAR_RELU_BN 32

/** KERNELS **/
/* Embedding CUDA kernel */
__global__ void kembedding(const int *in, const float *w, float *out, size_t s, size_t H) {

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x; 
  
  if (i < s && j < H) { 
    out[i * H + j] = w[in[i] * H + j];
  }
}

/* Permute CUDA kernel */
__global__ void kpermute(const float *in, float *out, size_t s, size_t H) {

  int i = blockIdx.y * blockDim.y + threadIdx.y;  
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < s && j < H) {
    out[j * s + i] = in[i * H + j]; 
  }
}

/* Conv1D CUDA kernel */
__global__ void k3conv1d(float *in, float *w, float *b, float *out, 
                              int C, int K, int s, int OC, int os){
  const int BLOCK_OC = CONV1D_K3_OC;
  const int BLOCK_C = CONV1D_K3_C;
  const int BLOCK_OS = CONV1D_K3_OS;
  
  const int KERNEL_SIZE = 3;

  __shared__ float t_in[BLOCK_C][BLOCK_OS + KERNEL_SIZE - 1 + 4];
  __shared__ float t_w[BLOCK_OC][BLOCK_C][KERNEL_SIZE + 4];

  float val = 0.0f;

  int out_m = blockIdx.x * BLOCK_OC;
  int out_n = blockIdx.y * BLOCK_OS;

  int out_tm = threadIdx.x / min(BLOCK_OS, os - out_n);
  int out_tn = threadIdx.x % min(BLOCK_OS, os - out_n);

  for(int bk = 0; bk < C; bk += BLOCK_C)
  {
    // Load input
    int in_k = bk;
    int in_n = out_n;

    int in_tk = threadIdx.x / min(BLOCK_OS + KERNEL_SIZE - 1, s - in_n);
    int in_tn = threadIdx.x % min(BLOCK_OS + KERNEL_SIZE - 1, s - in_n);

    if (in_tk < min(BLOCK_C, C - in_k)){
      t_in[in_tk][in_tn] = in[(in_k + in_tk) * s + in_n + in_tn];
    }
    // Load weight
    int w_m = out_m;
    int w_k = bk;

    int w_tm = threadIdx.x / min(BLOCK_C, C - w_k);
    int w_tk = threadIdx.x % min(BLOCK_C, C - w_k);

    if(w_tm < min(BLOCK_OC, OC - w_m)) {
      for (int i = 0; i < KERNEL_SIZE; i++) {
        t_w[w_tm][w_tk][i] = w[(w_m + w_tm) * C * K + (w_k + w_tk) * K + i];
      }
    }

    __syncthreads();

    // Compute
    if (out_tm < min(BLOCK_OC, OC - out_m)) {
      for (int k = 0; k < BLOCK_C; k++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
          val += t_w[out_tm][k][i] *  t_in[k][out_tn + i];
        }  
      }
    }
    __syncthreads();
  }

  // Store
  if(out_tm < min(BLOCK_OC, OC - out_m)){
    val += b[out_m + out_tm];
    out[(out_m + out_tm) * os + out_n + out_tn] = val > 0.0f ? val : 0.0f;
  }
}

__global__ void k5conv1d(float *in, float *w, float *b, float *out, 
                              int C, int K, int s, int OC, int os){
  const int BLOCK_OC = CONV1D_K5_OC;
  const int BLOCK_C = CONV1D_K5_C;
  const int BLOCK_OS = CONV1D_K5_OS;
  
  const int KERNEL_SIZE = 5;

  __shared__ float t_in[BLOCK_C][BLOCK_OS + KERNEL_SIZE - 1 + 4];
  __shared__ float t_w[BLOCK_OC][BLOCK_C][KERNEL_SIZE + 4];

  float val = 0.0f;

  int out_m = blockIdx.x * BLOCK_OC;
  int out_n = blockIdx.y * BLOCK_OS;

  int out_tm = threadIdx.x / min(BLOCK_OS, os - out_n);
  int out_tn = threadIdx.x % min(BLOCK_OS, os - out_n);

  for(int bk = 0; bk < C; bk += BLOCK_C)
  {
    // Load input
    int in_k = bk;
    int in_n = out_n;

    int in_tk = threadIdx.x / min(BLOCK_OS + KERNEL_SIZE - 1, s - in_n);
    int in_tn = threadIdx.x % min(BLOCK_OS + KERNEL_SIZE - 1, s - in_n);

    if (in_tk < min(BLOCK_C, C - in_k)){
      t_in[in_tk][in_tn] = in[(in_k + in_tk) * s + in_n + in_tn];
    }
    // Load weight
    int w_m = out_m;
    int w_k = bk;

    int w_tm = threadIdx.x / min(BLOCK_C, C - w_k);
    int w_tk = threadIdx.x % min(BLOCK_C, C - w_k);

    if(w_tm < min(BLOCK_OC, OC - w_m)) {
      for (int i = 0; i < KERNEL_SIZE; i++) {
        t_w[w_tm][w_tk][i] = w[(w_m + w_tm) * C * K + (w_k + w_tk) * K + i];
      }
    }

    __syncthreads();

    // Compute
    if (out_tm < min(BLOCK_OC, OC - out_m)) {
      for (int k = 0; k < BLOCK_C; k++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
          val += t_w[out_tm][k][i] *  t_in[k][out_tn + i];
        }  
      }
    }
    __syncthreads();
  }

  // Store
  if(out_tm < min(BLOCK_OC, OC - out_m)){
    val += b[out_m + out_tm];
    out[(out_m + out_tm) * os + out_n + out_tn] = val > 0.0f ? val : 0.0f;
  }
}

__global__ void k7conv1d(float *in, float *w, float *b, float *out, 
                              int C, int K, int s, int OC, int os){
  const int BLOCK_OC = CONV1D_K7_OC;
  const int BLOCK_C = CONV1D_K7_C;
  const int BLOCK_OS = CONV1D_K7_OS;
  
  const int KERNEL_SIZE = 7;

  __shared__ float t_in[BLOCK_C][BLOCK_OS + KERNEL_SIZE - 1 + 4];
  __shared__ float t_w[BLOCK_OC][BLOCK_C][KERNEL_SIZE + 4];

  float val = 0.0f;

  int out_m = blockIdx.x * BLOCK_OC;
  int out_n = blockIdx.y * BLOCK_OS;

  int out_tm = threadIdx.x / min(BLOCK_OS, os - out_n);
  int out_tn = threadIdx.x % min(BLOCK_OS, os - out_n);

  for(int bk = 0; bk < C; bk += BLOCK_C)
  {
    // Load input
    int in_k = bk;
    int in_n = out_n;

    int in_tk = threadIdx.x / min(BLOCK_OS + KERNEL_SIZE - 1, s - in_n);
    int in_tn = threadIdx.x % min(BLOCK_OS + KERNEL_SIZE - 1, s - in_n);

    if (in_tk < min(BLOCK_C, C - in_k)){
      t_in[in_tk][in_tn] = in[(in_k + in_tk) * s + in_n + in_tn];
    }
    // Load weight
    int w_m = out_m;
    int w_k = bk;

    int w_tm = threadIdx.x / min(BLOCK_C, C - w_k);
    int w_tk = threadIdx.x % min(BLOCK_C, C - w_k);

    if(w_tm < min(BLOCK_OC, OC - w_m)) {
      for (int i = 0; i < KERNEL_SIZE; i++) {
        t_w[w_tm][w_tk][i] = w[(w_m + w_tm) * C * K + (w_k + w_tk) * K + i];
      }
    }

    __syncthreads();

    // Compute
    if (out_tm < min(BLOCK_OC, OC - out_m)) {
      for (int k = 0; k < BLOCK_C; k++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
          val += t_w[out_tm][k][i] *  t_in[k][out_tn + i];
        }  
      }
    }
    __syncthreads();
  }

  // Store
  if(out_tm < min(BLOCK_OC, OC - out_m)){
    val += b[out_m + out_tm];
    out[(out_m + out_tm) * os + out_n + out_tn] = val > 0.0f ? val : 0.0f;
  }
}

__global__ void k9conv1d(float *in, float *w, float *b, float *out, 
                              int C, int K, int s, int OC, int os){
  const int BLOCK_OC = CONV1D_K9_OC;
  const int BLOCK_C = CONV1D_K9_C;
  const int BLOCK_OS = CONV1D_K9_OS;
  
  const int KERNEL_SIZE = 9;

  __shared__ float t_in[BLOCK_C][BLOCK_OS + KERNEL_SIZE - 1 + 4];
  __shared__ float t_w[BLOCK_OC][BLOCK_C][KERNEL_SIZE + 4];

  float val = 0.0f;

  int out_m = blockIdx.x * BLOCK_OC;
  int out_n = blockIdx.y * BLOCK_OS;

  int out_tm = threadIdx.x / min(BLOCK_OS, os - out_n);
  int out_tn = threadIdx.x % min(BLOCK_OS, os - out_n);

  for(int bk = 0; bk < C; bk += BLOCK_C)
  {
    // Load input
    int in_k = bk;
    int in_n = out_n;

    int in_tk = threadIdx.x / min(BLOCK_OS + KERNEL_SIZE - 1, s - in_n);
    int in_tn = threadIdx.x % min(BLOCK_OS + KERNEL_SIZE - 1, s - in_n);

    if (in_tk < min(BLOCK_C, C - in_k)){
      t_in[in_tk][in_tn] = in[(in_k + in_tk) * s + in_n + in_tn];
    }
    // Load weight
    int w_m = out_m;
    int w_k = bk;

    int w_tm = threadIdx.x / min(BLOCK_C, C - w_k);
    int w_tk = threadIdx.x % min(BLOCK_C, C - w_k);

    if(w_tm < min(BLOCK_OC, OC - w_m)) {
      for (int i = 0; i < KERNEL_SIZE; i++) {
        t_w[w_tm][w_tk][i] = w[(w_m + w_tm) * C * K + (w_k + w_tk) * K + i];
      }
    }

    __syncthreads();

    // Compute
    if (out_tm < min(BLOCK_OC, OC - out_m)) {
      for (int k = 0; k < BLOCK_C; k++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
          val += t_w[out_tm][k][i] *  t_in[k][out_tn + i];
        }  
      }
    }
    __syncthreads();
  }

  // Store
  if(out_tm < min(BLOCK_OC, OC - out_m)){
    val += b[out_m + out_tm];
    out[(out_m + out_tm) * os + out_n + out_tn] = val > 0.0f ? val : 0.0f;
  }
}


/* GetMax CUDA kernel */
__global__ void kgetmax(const float *in, float *out, size_t s, size_t C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  

  if (i < C) {  
    float max_val = in[i * s]; 
    
    for (size_t j = 1; j < s; j++) {
      max_val = max(max_val, in[i * s + j]);
    }

    out[i] = max_val;
  }
}

/* Concat CUDA kernel */
__global__ void kconcat(const float *in1, const float *in2, const float *in3, const float *in4, 
                              float *out, size_t N1, size_t N2, size_t N3, size_t N4) {
  
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N1) {
    out[i] = in1[i];
  } else if (i < N1 + N2) {
    out[i] = in2[i - N1];
  } else if (i < N1 + N2 + N3) {
    out[i] = in3[i - N1 - N2];
  } else if (i < N1 + N2 + N3 + N4) {
    out[i] = in4[i - N1 - N2 - N3];
  }
}

/* Linear CUDA kernel */
__global__ void klinear(float *in, float *w, float *b, float *out, int N, int M) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < M) {
    float val = 0.f;
    for (int j = 0; j < N; j++) {
        val += in[j] * w[i * N + j];
    }

    val += b[i];
    out[i] = val;
  }
}

__global__ void klinear_relu(float *in, float *w, float *b, float *out, int N, int M, bool relu) {
  const int BM = LINEAR_RELU_BM;
  const int BN = LINEAR_RELU_BN;

  float val = 0.0f;
  int i = threadIdx.x;
  int row = blockIdx.x * BM;

  __shared__ float t_in[BN + 4];
  __shared__ float t_w[BM][BN + 4];

  for (int col = 0; col < N; col += BN) {
    // Load input
    for (int j = 0; j < BN / BM; j++) {
      t_in[i * BN / BM + j] = in[col + i * BN / BM + j];
    }

    // Load weight
    for (int j = 0; j < BN; j++) {
      t_w[i][j] = w[N * (row + i) + col + j];
    }

    __syncthreads();

    // Compute
    for (int j = 0; j < BN; j++) {
      val += t_w[i][j] * t_in[j];
    }
    __syncthreads();
  }

  // Store
  val += b[row + i];
  if (relu && val < 0.0f) val = 0.0f;
  out[row + i] = val;
}


/* Embedding
 * @param [in1]  in: [s]
 * @param [in2]   w: [NUM_VOCAB, H]
 * @param [out] out: [s, H]
 * 's' is the sequence length
 * 'H' is the embedding dimension
 */

void Embedding(int *in, float* w, float *out, size_t s, size_t H) {
  dim3 blockDim(8, 16);
  dim3 gridDim(ceil_div(H, blockDim.x), ceil_div(s, blockDim.y));
  kembedding<<<gridDim, blockDim>>>(in, w, out, s, H);
}

/* Permute
 * @param [in]   in: [M, N]
 * @param [out] out: [N, M]
 */
void Permute(float *in, float *out, size_t s, size_t H) {
  dim3 blockDim(32, 8);
  dim3 gridDim(ceil_div(H, blockDim.x), ceil_div(s, blockDim.y));
  kpermute<<<gridDim, blockDim>>>(in, out, s, H);
}

/* Conv1D 
 * @param [in1]  in: [C, s]
 * @param [in2]   w: [OC, C, K] 
 * @param [in3]   b: [OC]
 * @param [out] out: [OC, os]
 *    
 *    In this model, K is 3, 5, 7, or 9, 
 *    with stride = 1, pad = 0, dilation = 1.
 *    The formula for the output sequence length:
 *      os = (in - K + 2 * pad) / stride + 1
 *          = (s - K + 2 * 0) / 1 + 1
 *          = s - K + 1
 *
 * 'C' is the input channel size
 * 's' is the input sequence length
 * 'OC' is the output channel size
 * 'os' is the output sequence length
 * 'K' is the kernel (or filter) size
 */
void Conv1D_K3(float *in, float *w, float *b, float *out, size_t s, size_t C, size_t OC, size_t K, cudaStream_t stream){
  size_t os = s - K + 1;
  dim3 blockDim(CONV1D_K3_OC * CONV1D_K3_OS);
  dim3 gridDim(ceil_div(OC, CONV1D_K3_OC), ceil_div(os, CONV1D_K3_OS));
  k3conv1d<<<gridDim, blockDim, 0, stream>>>(in, w, b, out, C, K, s, OC, os);
}

void Conv1D_K5(float *in, float *w, float *b, float *out, size_t s, size_t C, size_t OC, size_t K, cudaStream_t stream){
  size_t os = s - K + 1;
  dim3 blockDim(CONV1D_K5_OC * CONV1D_K5_OS);
  dim3 gridDim(ceil_div(OC, CONV1D_K5_OC), ceil_div(os, CONV1D_K5_OS));
  k5conv1d<<<gridDim, blockDim, 0, stream>>>(in, w, b, out, C, K, s, OC, os);
}

void Conv1D_K7(float *in, float *w, float *b, float *out, size_t s, size_t C, size_t OC, size_t K, cudaStream_t stream){
  size_t os = s - K + 1;
  dim3 blockDim(CONV1D_K7_OC * CONV1D_K7_OS);
  dim3 gridDim(ceil_div(OC, CONV1D_K7_OC), ceil_div(os, CONV1D_K7_OS));
  k7conv1d<<<gridDim, blockDim, 0, stream>>>(in, w, b, out, C, K, s, OC, os);
}

void Conv1D_K9(float *in, float *w, float *b, float *out, size_t s, size_t C, size_t OC, size_t K, cudaStream_t stream){
  size_t os = s - K + 1;
  dim3 blockDim(CONV1D_K9_OC * CONV1D_K9_OS);
  dim3 gridDim(ceil_div(OC, CONV1D_K9_OC), ceil_div(os, CONV1D_K9_OS));
  k9conv1d<<<gridDim, blockDim, 0, stream>>>(in, w, b, out, C, K, s, OC, os);
}

/* GetMax
 * @param [in]   in: [C, s]
 * @param [out] out: [C]
 *    
 *    This layer is to get the max value along the sequence dim.
 *    The formula for this layer: out = max(in, dim=-1)
 * 
 * 'C' is the channel size
 * 's' is the sequence length
 */
void GetMax(float *in, float *out, size_t C, size_t s, cudaStream_t stream){
  dim3 blockDim(256);  
  dim3 gridDim(ceil_div(C, blockDim.x));
  kgetmax<<<gridDim, blockDim, 0, stream>>>(in, out, s, C);
}

/* Concat
 * @param [in1] in1: [N1]
 * @param [in2] in2: [N2]
 * @param [in3] in3: [N3]
 * @param [in4] in4: [N4]
 * @param [out] out: [N1 + N2 + N3 + N4]
 * 'N1', 'N2', 'N3', and 'N4' are the num of elems in the floats.
 */
void Concat(float *in1, float *in2, float *in3, float *in4, 
            float *out, size_t N1, size_t N2, size_t N3, size_t N4, cudaStream_t stream) {
  dim3 blockDim(256);
  dim3 gridDim(ceil_div((N1 + N2 + N3 + N4), blockDim.x));
  kconcat<<<gridDim, blockDim, 0, stream>>>(in1, in2, in3, in4, out, N1, N2, N3, N4);
}

/* Linear 
 * @param [in1]  in: [N]
 * @param [in2]   w: [M, N]
 * @param [in3]   b: [M]
 * @param [out] out: [M]
 * 'N' is the input feature size
 * 'M' is the output feature size
 */
void Linear_ReLU(float *in, float *w, float *b, float *out, int N, int M, cudaStream_t stream) {
    int blockDim(LINEAR_RELU_BM);
    int gridDim(ceil_div(M, blockDim));
    klinear_relu<<<gridDim, blockDim, 0, stream>>>(in, w, b, out, N, M, true);
}

// Final result
void Linear(float *in, float *w, float *b, float *out, int N, int M, cudaStream_t stream) {
    int blockDim(LINEAR_BM);
    int gridDim(ceil_div(M, blockDim));
    klinear<<<gridDim, blockDim, 0, stream>>>(in, w, b, out, N, M);
}
