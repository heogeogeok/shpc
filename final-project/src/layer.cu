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
static cudaStream_t streams[NGPU];

/** SECTION: Kernels **/
/* Embedding CUDA kernel */
__global__ void kembedding(const int *in, const float *w, float *out, size_t s, size_t H, size_t embedding_dim) {

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

void Embedding(int *in, Tensor* w, Tensor *out) {
  size_t s = out->shape[0];
  size_t H = out->shape[1];

  int *d_in;
  float *d_w, *d_out;
  size_t in_size = s * sizeof(int);
  size_t w_size = w->shape[0] * w->shape[1] * sizeof(float);
  size_t out_size = s * H * sizeof(float);

  CHECK_CUDA(cudaMalloc(&d_in, in_size));
  CHECK_CUDA(cudaMalloc(&d_w, w_size));
  CHECK_CUDA(cudaMalloc(&d_out, out_size));

  CHECK_CUDA(cudaMemcpy(d_in, in, in_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_w, w->buf, w_size, cudaMemcpyHostToDevice));

  dim3 blockDim(16, 16);
  dim3 gridDim((H + blockDim.x - 1) / blockDim.x, (s + blockDim.y - 1) / blockDim.y);

  kembedding<<<gridDim, blockDim>>>(d_in, d_w, d_out, s, H, w->shape[1]);

  CHECK_CUDA(cudaMemcpy(out->buf, d_out, out_size, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_w));
  CHECK_CUDA(cudaFree(d_out));
}

/* Permute
 * @param [in]   in: [M, N]
 * @param [out] out: [N, M]
 */
void Permute(Tensor *in, Tensor *out) {
  size_t s = in->shape[0];  
  size_t H = in->shape[1];

  float *d_in, *d_out;
  size_t in_size = s * H * sizeof(float);
  size_t out_size = s * H * sizeof(float);

  CHECK_CUDA(cudaMalloc(&d_in, in_size));
  CHECK_CUDA(cudaMalloc(&d_out, out_size));

  CHECK_CUDA(cudaMemcpy(d_in, in->buf, in_size, cudaMemcpyHostToDevice));

  dim3 blockDim(16, 16);
  dim3 gridDim((H + blockDim.x - 1) / blockDim.x, (s + blockDim.y - 1) / blockDim.y);

  kpermute<<<gridDim, blockDim>>>(d_in, d_out, s, H);

  CHECK_CUDA(cudaMemcpy(out->buf, d_out, out_size, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
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
void Conv1D(Tensor *in, Tensor *w, Tensor *b, Tensor *out){

  size_t s = in->shape[1];
  size_t C = in->shape[0];
  size_t OC = w->shape[0];
  size_t K = w->shape[2];

  size_t os = s - K + 1;

  float *d_in, *d_w, *d_b, *d_out;
  size_t size_in = in->num_elem() * sizeof(float);
  size_t size_w = w->num_elem() * sizeof(float);
  size_t size_b = b->num_elem() * sizeof(float);
  size_t size_out = out->num_elem() * sizeof(float);

  for (int i = 0; i < NGPU; ++i) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
  }

  CHECK_CUDA(cudaMalloc(&d_in, size_in));
  CHECK_CUDA(cudaMalloc(&d_w, size_w));
  CHECK_CUDA(cudaMalloc(&d_b, size_b));
  CHECK_CUDA(cudaMalloc(&d_out, size_out));

  CHECK_CUDA(cudaMemcpy(d_in, in->buf, size_in, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_w, w->buf, size_w, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b->buf, size_b, cudaMemcpyHostToDevice));

  dim3 blockDim(16, 16);
  dim3 gridDim((OC + 16) / 16, (os + 16) / 16);

  kconv1d<<<gridDim, blockDim>>>(d_in, d_w, d_b, d_out, C, K, s, OC, os);
  
  CHECK_CUDA(cudaMemcpy(out->buf, d_out, size_out, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_w));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_out));
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
void GetMax(Tensor *in, Tensor *out){
  size_t C = in->shape[0];
  size_t s = in->shape[1];

  float *d_in, *d_out;
  size_t in_size = C * s * sizeof(float);
  size_t out_size = C * sizeof(float);

  cudaMalloc(&d_in, in_size);
  cudaMalloc(&d_out, out_size);

  cudaMemcpy(d_in, in->buf, in_size, cudaMemcpyHostToDevice);

  dim3 blockDim(256);  
  dim3 gridDim((C + blockDim.x - 1) / blockDim.x);

  kgetmax<<<gridDim, blockDim>>>(d_in, d_out, s, C);

  cudaMemcpy(out->buf, d_out, out_size, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

/* Concat
 * @param [in1] in1: [N1]
 * @param [in2] in2: [N2]
 * @param [in3] in3: [N3]
 * @param [in4] in4: [N4]
 * @param [out] out: [N1 + N2 + N3 + N4]
 * 'N1', 'N2', 'N3', and 'N4' are the num of elems in the tensors.
 */
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out) {
  
  size_t N1 = in1->shape[0];
  size_t N2 = in2->shape[0];
  size_t N3 = in3->shape[0];
  size_t N4 = in4->shape[0];

  float *d_in1, *d_in2, *d_in3, *d_in4, *d_out;
  size_t out_size = (N1 + N2 + N3 + N4) * sizeof(float);

  CHECK_CUDA(cudaMalloc(&d_in1, N1 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_in2, N2 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_in3, N3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_in4, N4 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, out_size));

  CHECK_CUDA(cudaMemcpy(d_in1, in1->buf, N1 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_in2, in2->buf, N2 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_in3, in3->buf, N3 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_in4, in4->buf, N4 * sizeof(float), cudaMemcpyHostToDevice));

  dim3 blockDim(256);
  dim3 gridDim((N1 + N2 + N3 + N4 + blockDim.x - 1) / blockDim.x);

  kconcat<<<gridDim, blockDim>>>(d_in1, d_in2, d_in3, d_in4, d_out, N1, N2, N3, N4);

  CHECK_CUDA(cudaMemcpy(out->buf, d_out, out_size, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_in1));
  CHECK_CUDA(cudaFree(d_in2));
  CHECK_CUDA(cudaFree(d_in3));
  CHECK_CUDA(cudaFree(d_in4));
  CHECK_CUDA(cudaFree(d_out));
}

/* Linear 
 * @param [in1]  in: [N]
 * @param [in2]   w: [M, N]
 * @param [in3]   b: [M]
 * @param [out] out: [M]
 * 'N' is the input feature size
 * 'M' is the output feature size
 */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
    
    float *d_in, *d_w, *d_b, *d_out;
    size_t size_in = in->num_elem() * sizeof(float);
    size_t size_w = w->num_elem() * sizeof(float);
    size_t size_b = b->num_elem() * sizeof(float);
    size_t size_out = out->num_elem() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_in, size_in));
    CHECK_CUDA(cudaMalloc(&d_w, size_w));
    CHECK_CUDA(cudaMalloc(&d_b, size_b));
    CHECK_CUDA(cudaMalloc(&d_out, size_out));

    CHECK_CUDA(cudaMemcpy(d_in, in->buf, size_in, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, w->buf, size_w, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b->buf, size_b, cudaMemcpyHostToDevice));

    int M = w->shape[0];
    int blockDim = 256;
    int gridDim = (M + blockDim - 1) / blockDim;
    klinear<<<gridDim, blockDim>>>(d_in, d_w, d_b, d_out, in->shape[0], M, false);

    CHECK_CUDA(cudaMemcpy(out->buf, d_out, size_out, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_out));
}

void Linear_ReLU(Tensor *in, Tensor *w, Tensor *b, Tensor *out){
    
    float *d_in, *d_w, *d_b, *d_out;
    size_t size_in = in->num_elem() * sizeof(float);
    size_t size_w = w->num_elem() * sizeof(float);
    size_t size_b = b->num_elem() * sizeof(float);
    size_t size_out = out->num_elem() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_in, size_in));
    CHECK_CUDA(cudaMalloc(&d_w, size_w));
    CHECK_CUDA(cudaMalloc(&d_b, size_b));
    CHECK_CUDA(cudaMalloc(&d_out, size_out));

    CHECK_CUDA(cudaMemcpy(d_in, in->buf, size_in, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, w->buf, size_w, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b->buf, size_b, cudaMemcpyHostToDevice));

    int M = w->shape[0];
    int blockDim = 256;
    int gridDim = (M + blockDim - 1) / blockDim;
    klinear<<<gridDim, blockDim>>>(d_in, d_w, d_b, d_out, in->shape[0], M, true);

    CHECK_CUDA(cudaMemcpy(out->buf, d_out, size_out, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_out));
}

