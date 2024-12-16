#include "layer.h"

#define div(x, y) (((x) + (y) -1) / (y))

/** SECTION: GPU manipulation **/
#define NGPU    4

/** SECTION: Kernels **/
/* ReLU CUDA kernel */
__global__ void krelu(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    inout[i] = inout[i] > 0 ? inout[i] : 0;
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

  #pragma omp parallel for
  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < H; j++) {
      out->buf[i * H + j] = w->buf[in[i] * H + j];
    }
  }
}

/* Permute
 * @param [in]   in: [M, N]
 * @param [out] out: [N, M]
 */
void Permute(Tensor *in, Tensor *out) {
  size_t s = in->shape[0];
  size_t H = in->shape[1];

  // #pragma omp parallel for
  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < H; j++) {
      out->buf[j * s + i] = in->buf[i * H + j];
    }
  }
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
void GetMax(Tensor *in, Tensor *out) {
  size_t C = in->shape[0];
  size_t s = in->shape[1];

  #pragma omp parallel for
  for (size_t i = 0; i < C; i++) {
    out->buf[i] = in->buf[i * s];
    for (size_t j = 1; j < s; j++) {
      out->buf[i] = in->buf[i * s + j] > out->buf[i] ? 
        in->buf[i * s + j] : out->buf[i];
    }
  }
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

  for (size_t i = 0; i < N1; i++) {
    out->buf[i] = in1->buf[i];
  }
  for (size_t i = 0; i < N2; i++) {
    out->buf[N1 + i] = in2->buf[i];
  }
  for (size_t i = 0; i < N3; i++) {
    out->buf[N1 + N2 + i] = in3->buf[i];
  }
  for (size_t i = 0; i < N4; i++) {
    out->buf[N1 + N2 + N3 + i] = in4->buf[i];
  }
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

