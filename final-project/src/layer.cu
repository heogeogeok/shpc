#include "layer.h"

#define div(x, y) (((x) + (y) -1) / (y))

/** SECTION: Hyperparams **/
#define LINEAR_BM 32
#define C1D_K3_BM 16
#define C1D_K3_BN 8
#define C1D_K3_BK 8

#define C1D_K5_BM 16
#define C1D_K5_BN 8
#define C1D_K5_BK 8

#define C1D_K7_BM 16
#define C1D_K7_BN 8
#define C1D_K7_BK 8

#define C1D_K9_BM 8
#define C1D_K9_BN 32
#define C1D_K9_BK 4

/** SECTION: DEBUGGING **/
#define DEBUG 0
#if DEBUG == 1
double dbg_start_time, dbg_ce_init, dbg_ce_final;
#define DEBUG_PRINT(...) do { \
  printf(__VA_ARGS__); \
} while (0)
#else
#define DEBUG_PRINT(...)
#endif

/** SECTION: Kernels **/
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
  const int BK = C1D_K3_BK;
  const int BN = C1D_K3_BN;
  const int BM = C1D_K3_BM;
  const int KERNEL_SIZE = 3;

  __shared__ float t_in[BK][BN + KERNEL_SIZE - 1 + 4];
  __shared__ float t_w[BM][BK][KERNEL_SIZE + 4];

  float val = 0.0f;

  // output blocks
  int oblock_m_offset = blockIdx.x * BM;
  int oblock_n_offset = blockIdx.y * BN;

  int len_oblock_m = min(BM, OC - oblock_m_offset);
  int len_oblock_n = min(BN, os - oblock_n_offset);

  int othread_m_offset = threadIdx.x / len_oblock_n;
  int othread_n_offset = threadIdx.x % len_oblock_n;

  int othread_valid = othread_m_offset < len_oblock_m;

  for(int bk = 0; bk < C; bk += BK)
  {
    // Load input
    int iblock_k_offset = bk;
    int iblock_n_offset = oblock_n_offset;
    int len_iblock_k = min(BK, C - iblock_k_offset);
    int len_iblock_n = min(BN + KERNEL_SIZE - 1, s - iblock_n_offset);
    int ithread_k_offset = threadIdx.x / len_iblock_n;
    int ithread_n_offset = threadIdx.x % len_iblock_n;

    int ithread_valid = ithread_k_offset < len_iblock_k;

    if (ithread_valid){
      t_in[ithread_k_offset][ithread_n_offset] = in[(iblock_k_offset + ithread_k_offset) * s + iblock_n_offset + ithread_n_offset];
    }

    // Load weight
    int wblock_m_offset = oblock_m_offset;
    int wblock_k_offset = bk;
    int len_wblock_m = min(BM, OC - wblock_m_offset);
    int len_wblock_k = min(BK, C - wblock_k_offset);
    int wthread_m_offset = threadIdx.x / len_wblock_k;
    int wthread_k_offset = threadIdx.x % len_wblock_k;

    int wthread_valid = wthread_m_offset < len_wblock_m;

    if(wthread_valid) {
      for (int i = 0; i < KERNEL_SIZE; i++) {
        t_w[wthread_m_offset][wthread_k_offset][i] = w[(wblock_m_offset + wthread_m_offset) * C * K + (wblock_k_offset + wthread_k_offset) * K + i];
      }
    }

    __syncthreads();

    // Compute
    if (othread_valid) {
      for (int k = 0; k < BK; k++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
          val += t_w[othread_m_offset][k][i] *  t_in[k][othread_n_offset+ i];
        }  
      }
    }
    
    __syncthreads();
  }

  // Store
  if(othread_valid){
    val += b[oblock_m_offset + othread_m_offset];
    out[(oblock_m_offset + othread_m_offset) * os + oblock_n_offset + othread_n_offset] = val > 0.0f ? val : 0.0f;
  }
}

__global__ void k5conv1d(float *in, float *w, float *b, float *out, 
                              int C, int K, int s, int OC, int os){
  const int BK = C1D_K5_BK;
  const int BN = C1D_K5_BN;
  const int BM = C1D_K5_BM;
  const int KERNEL_SIZE = 5;

  __shared__ float t_in[BK][BN + KERNEL_SIZE - 1 + 4];
  __shared__ float t_w[BM][BK][KERNEL_SIZE + 4];

  float val = 0.0f;

  // output blocks
  int oblock_m_offset = blockIdx.x * BM;
  int oblock_n_offset = blockIdx.y * BN;

  int len_oblock_m = min(BM, OC - oblock_m_offset);
  int len_oblock_n = min(BN, os - oblock_n_offset);

  int othread_m_offset = threadIdx.x / len_oblock_n;
  int othread_n_offset = threadIdx.x % len_oblock_n;

  int othread_valid = othread_m_offset < len_oblock_m;

  for(int bk = 0; bk < C; bk += BK)
  {
    // Load input
    int iblock_k_offset = bk;
    int iblock_n_offset = oblock_n_offset;
    int len_iblock_k = min(BK, C - iblock_k_offset);
    int len_iblock_n = min(BN + KERNEL_SIZE - 1, s - iblock_n_offset);
    int ithread_k_offset = threadIdx.x / len_iblock_n;
    int ithread_n_offset = threadIdx.x % len_iblock_n;

    int ithread_valid = ithread_k_offset < len_iblock_k;

    if (ithread_valid){
      t_in[ithread_k_offset][ithread_n_offset] = in[(iblock_k_offset + ithread_k_offset) * s + iblock_n_offset + ithread_n_offset];
    }

    // Load weight
    int wblock_m_offset = oblock_m_offset;
    int wblock_k_offset = bk;
    int len_wblock_m = min(BM, OC - wblock_m_offset);
    int len_wblock_k = min(BK, C - wblock_k_offset);
    int wthread_m_offset = threadIdx.x / len_wblock_k;
    int wthread_k_offset = threadIdx.x % len_wblock_k;

    int wthread_valid = wthread_m_offset < len_wblock_m;

    if(wthread_valid) {
      for (int i = 0; i < KERNEL_SIZE; i++) {
        t_w[wthread_m_offset][wthread_k_offset][i] = w[(wblock_m_offset + wthread_m_offset) * C * K + (wblock_k_offset + wthread_k_offset) * K + i];
      }
    }

    __syncthreads();

    // Compute
    if (othread_valid) {
      for (int k = 0; k < BK; k++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
          val += t_w[othread_m_offset][k][i] *  t_in[k][othread_n_offset+ i];
        }  
      }
    }
    
    __syncthreads();
  }

  // store
  if(othread_valid){
    val += b[oblock_m_offset + othread_m_offset];
    out[(oblock_m_offset + othread_m_offset) * os + oblock_n_offset + othread_n_offset] = val > 0.0f ? val : 0.0f;
  }
}

__global__ void k7conv1d(float *in, float *w, float *b, float *out, 
                              int C, int K, int s, int OC, int os){
  const int BK = C1D_K7_BK;
  const int BN = C1D_K7_BN;
  const int BM = C1D_K7_BM;
  const int KERNEL_SIZE = 7;

  __shared__ float t_in[BK][BN + KERNEL_SIZE - 1 + 4];
  __shared__ float t_w[BM][BK][KERNEL_SIZE + 4];

  float val = 0.0f;

  // Output blocks
  int oblock_m_offset = blockIdx.x * BM;
  int oblock_n_offset = blockIdx.y * BN;

  int len_oblock_m = min(BM, OC - oblock_m_offset);
  int len_oblock_n = min(BN, os - oblock_n_offset);

  int othread_m_offset = threadIdx.x / len_oblock_n;
  int othread_n_offset = threadIdx.x % len_oblock_n;

  int othread_valid = othread_m_offset < len_oblock_m;

  for(int bk = 0; bk < C; bk += BK)
  {
    // Load input
    int iblock_k_offset = bk;
    int iblock_n_offset = oblock_n_offset;
    int len_iblock_k = min(BK, C - iblock_k_offset);
    int len_iblock_n = min(BN + KERNEL_SIZE - 1, s - iblock_n_offset);
    int ithread_k_offset = threadIdx.x / len_iblock_n;
    int ithread_n_offset = threadIdx.x % len_iblock_n;

    int ithread_valid = ithread_k_offset < len_iblock_k;

    if (ithread_valid){
      t_in[ithread_k_offset][ithread_n_offset] = in[(iblock_k_offset + ithread_k_offset) * s + iblock_n_offset + ithread_n_offset];
    }

    // Load weight
    int wblock_m_offset = oblock_m_offset;
    int wblock_k_offset = bk;
    int len_wblock_m = min(BM, OC - wblock_m_offset);
    int len_wblock_k = min(BK, C - wblock_k_offset);
    int wthread_m_offset = threadIdx.x / len_wblock_k;
    int wthread_k_offset = threadIdx.x % len_wblock_k;

    int wthread_valid = wthread_m_offset < len_wblock_m;

    if(wthread_valid) {
      for (int i = 0; i < KERNEL_SIZE; i++) {
        t_w[wthread_m_offset][wthread_k_offset][i] = w[(wblock_m_offset + wthread_m_offset) * C * K + (wblock_k_offset + wthread_k_offset) * K + i];
      }
    }

    __syncthreads();

    // compute
    if (othread_valid) {
      for (int k = 0; k < BK; k++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
          val += t_w[othread_m_offset][k][i] *  t_in[k][othread_n_offset+ i];
        }  
      }
    }
    
    __syncthreads();
  }

  // store
  if(othread_valid){
    val += b[oblock_m_offset + othread_m_offset];
    out[(oblock_m_offset + othread_m_offset) * os + oblock_n_offset + othread_n_offset] = val > 0.0f ? val : 0.0f;
  }
}

__global__ void k9conv1d(float *in, float *w, float *b, float *out, 
                              int C, int K, int s, int OC, int os){
  const int BK = C1D_K9_BK;
  const int BN = C1D_K9_BN;
  const int BM = C1D_K9_BM;
  const int KERNEL_SIZE = 9;

  __shared__ float t_in[BK][BN + KERNEL_SIZE - 1 + 4];
  __shared__ float t_w[BM][BK][KERNEL_SIZE + 4];

  float val = 0.0f;

  // output blocks
  int oblock_m_offset = blockIdx.x * BM;
  int oblock_n_offset = blockIdx.y * BN;

  int len_oblock_m = min(BM, OC - oblock_m_offset);
  int len_oblock_n = min(BN, os - oblock_n_offset);

  int othread_m_offset = threadIdx.x / len_oblock_n;
  int othread_n_offset = threadIdx.x % len_oblock_n;

  int othread_valid = othread_m_offset < len_oblock_m;

  for(int bk = 0; bk < C; bk += BK)
  {
    // Load input
    int iblock_k_offset = bk;
    int iblock_n_offset = oblock_n_offset;
    int len_iblock_k = min(BK, C - iblock_k_offset);
    int len_iblock_n = min(BN + KERNEL_SIZE - 1, s - iblock_n_offset);
    int ithread_k_offset = threadIdx.x / len_iblock_n;
    int ithread_n_offset = threadIdx.x % len_iblock_n;

    int ithread_valid = ithread_k_offset < len_iblock_k;

    if (ithread_valid){
      t_in[ithread_k_offset][ithread_n_offset] = in[(iblock_k_offset + ithread_k_offset) * s + iblock_n_offset + ithread_n_offset];
    }

    // Load weight
    int wblock_m_offset = oblock_m_offset;
    int wblock_k_offset = bk;
    int len_wblock_m = min(BM, OC - wblock_m_offset);
    int len_wblock_k = min(BK, C - wblock_k_offset);
    int wthread_m_offset = threadIdx.x / len_wblock_k;
    int wthread_k_offset = threadIdx.x % len_wblock_k;

    int wthread_valid = wthread_m_offset < len_wblock_m;

    if(wthread_valid) {
      for (int i = 0; i < KERNEL_SIZE; i++) {
        t_w[wthread_m_offset][wthread_k_offset][i] = w[(wblock_m_offset + wthread_m_offset) * C * K + (wblock_k_offset + wthread_k_offset) * K + i];
      }
    }

    __syncthreads();

    // Compute
    if (othread_valid) {
      for (int k = 0; k < BK; k++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
          val += t_w[othread_m_offset][k][i] *  t_in[k][othread_n_offset+ i];
        }  
      }
    }
    
    __syncthreads();
  }

  // store
  if(othread_valid){
    val += b[oblock_m_offset + othread_m_offset];
    out[(oblock_m_offset + othread_m_offset) * os + oblock_n_offset + othread_n_offset] = val > 0.0f ? val : 0.0f;
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
  
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N1) {
    out[idx] = in1[idx];
  } else if (idx < N1 + N2) {
    out[idx] = in2[idx - N1];
  } else if (idx < N1 + N2 + N3) {
    out[idx] = in3[idx - N1 - N2];
  } else if (idx < N1 + N2 + N3 + N4) {
    out[idx] = in4[idx - N1 - N2 - N3];
  }
}

/* Linear CUDA kernel */
__global__ void klinear(float *in, float *w, float *b, float *out, int N, int M, bool relu) {
  const int TILESIZE = LINEAR_BM;
  __shared__ float t_in[TILESIZE];
  __shared__ float t_w[TILESIZE][TILESIZE + 8];

  int i = threadIdx.x;
  int row = blockIdx.x * TILESIZE + i;

  float val = 0.0f;
  int ntiles = (N + TILESIZE - 1) / TILESIZE;

  for (int t = 0; t < ntiles; t++) {
    // Load input
    if (t * TILESIZE + i < N) {
      t_in[i] = in[t * TILESIZE + i];
    } else {
      t_in[i] = 0.0f;
    }
    // Load weight
    for (int j = 0; j < TILESIZE; j++) {
      int col = t * TILESIZE + j;
      if (row < M && col < N) {
        t_w[j][i] = w[row * N + col];
      } else {
        t_w[j][i] = 0.0f;
      }
    }
    __syncthreads();

    // Compute
    for (int j = 0; j < TILESIZE; j++) {
      val += t_in[j] * t_w[j][i];
    }
    __syncthreads();
  }

  // Store the result
  if (row < M) {
    out[row] = val + b[row];
    if (relu) out[row] = fmaxf(out[row], 0.0f);
  }
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
  dim3 gridDim(div(H, blockDim.x), div(s, blockDim.y));

  kembedding<<<gridDim, blockDim>>>(in, w, out, s, H);
}

/* Permute
 * @param [in]   in: [M, N]
 * @param [out] out: [N, M]
 */
void Permute(float *in, float *out, size_t s, size_t H) {
  dim3 blockDim(32, 8);
  dim3 gridDim(div(H, blockDim.x), div(s, blockDim.y));

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
  dim3 blockDim(C1D_K3_BM * C1D_K3_BN);
  dim3 gridDim(div(OC, C1D_K3_BM), div(os, C1D_K3_BN));

  k3conv1d<<<gridDim, blockDim, 0, stream>>>(in, w, b, out, C, K, s, OC, os);
}

void Conv1D_K5(float *in, float *w, float *b, float *out, size_t s, size_t C, size_t OC, size_t K, cudaStream_t stream){
  size_t os = s - K + 1;
  dim3 blockDim(C1D_K5_BM * C1D_K5_BN);
  dim3 gridDim(div(OC, C1D_K5_BM), div(os, C1D_K5_BN));
  k5conv1d<<<gridDim, blockDim, 0, stream>>>(in, w, b, out, C, K, s, OC, os);
}

void Conv1D_K7(float *in, float *w, float *b, float *out, size_t s, size_t C, size_t OC, size_t K, cudaStream_t stream){
  size_t os = s - K + 1;
  dim3 blockDim(C1D_K7_BM * C1D_K7_BN);
  dim3 gridDim(div(OC, C1D_K7_BM), div(os, C1D_K7_BN));
  k7conv1d<<<gridDim, blockDim, 0, stream>>>(in, w, b, out, C, K, s, OC, os);
}

void Conv1D_K9(float *in, float *w, float *b, float *out, size_t s, size_t C, size_t OC, size_t K, cudaStream_t stream){
  size_t os = s - K + 1;
  dim3 blockDim(C1D_K9_BM * C1D_K9_BN);
  dim3 gridDim(div(OC, C1D_K9_BM), div(os, C1D_K9_BN));
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
  dim3 gridDim(div(C, blockDim.x));
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
  dim3 gridDim(div((N1 + N2 + N3 + N4), blockDim.x));
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
    int blockDim(LINEAR_BM);
    int gridDim(div(M, blockDim));
    klinear<<<gridDim, blockDim, 0, stream>>>(in, w, b, out, N, M, true);
}

// Final result
void Linear(float *in, float *w, float *b, float *out, int N, int M, cudaStream_t stream) {
    int blockDim(LINEAR_BM);
    int gridDim(div(M, blockDim));
    klinear<<<gridDim, blockDim, 0, stream>>>(in, w, b, out, N, M, false);
}
