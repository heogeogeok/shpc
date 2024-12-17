#include "layer.h"

#define div(x, y) (((x) + (y) -1) / (y))

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

/** SECTION: GPU manipulation **/
#define NGPU    4

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
__global__ void kconv1d(float *in, float *w, float *b, float *out, 
                              int C, int K, int s, int OC, int os){
    int oc = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 

    if (oc < OC && j < os) {
        float val = 0.0f;

        for (int k = 0; k < C; k++) {          
            for (int l = 0; l < K; l++) {      
                val += in[k * s + j + l] * w[oc * C * K + k * K + l];
            }
        }

        val += b[oc];
        out[oc * os + j] = val > 0.0f ? val : 0.0f;
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M) {
        float val = 0.f;
        for (int j = 0; j < N; j++) {
            val += in[j] * w[i * N + j];
        }

        val += b[i];
        if (relu) {
            val = fmaxf(val, 0.0f);
        }
        out[i] = val;
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
  dim3 blockDim(16, 16);
  dim3 gridDim((H + blockDim.x - 1) / blockDim.x, (s + blockDim.y - 1) / blockDim.y);

  kembedding<<<gridDim, blockDim>>>(in, w, out, s, H);
}

/* Permute
 * @param [in]   in: [M, N]
 * @param [out] out: [N, M]
 */
void Permute(float *in, float *out, size_t s, size_t H) {
  dim3 blockDim(16, 16);
  dim3 gridDim((H + blockDim.x - 1) / blockDim.x, (s + blockDim.y - 1) / blockDim.y);

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
void Conv1D(float *in, float *w, float *b, float *out, size_t s, size_t C, size_t OC, size_t K){
  size_t os = s - K + 1;

  dim3 blockDim(16, 16);
  dim3 gridDim((OC + 16) / 16, (os + 16) / 16);

  kconv1d<<<gridDim, blockDim>>>(in, w, b, out, C, K, s, OC, os);
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
void GetMax(float *in, float *out, size_t C, size_t s){
  dim3 blockDim(256);  
  dim3 gridDim((C + blockDim.x - 1) / blockDim.x);

  kgetmax<<<gridDim, blockDim>>>(in, out, s, C);

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
            float *out, size_t N1, size_t N2, size_t N3, size_t N4) {
  dim3 blockDim(256);
  dim3 gridDim((N1 + N2 + N3 + N4 + blockDim.x - 1) / blockDim.x);

  kconcat<<<gridDim, blockDim>>>(in1, in2, in3, in4, out, N1, N2, N3, N4);
}

/* Linear 
 * @param [in1]  in: [N]
 * @param [in2]   w: [M, N]
 * @param [in3]   b: [M]
 * @param [out] out: [M]
 * 'N' is the input feature size
 * 'M' is the output feature size
 */
void Linear_ReLU(float *in, float *w, float *b, float *out, int N, int M){
    int blockDim = 256;
    int gridDim = (M + blockDim - 1) / blockDim;
    klinear<<<gridDim, blockDim>>>(in, w, b, out, N, M, true);
}

// Final result
void Linear(float *in, float *w, float *b, float *out, int N, int M) {
    int blockDim = 256;
    int gridDim = (M + blockDim - 1) / blockDim;

    klinear<<<gridDim, blockDim>>>(in, w, b, out, N, M, false);
}
