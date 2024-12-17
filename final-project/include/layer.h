#pragma once

#include "tensor.h"


/* Operations (layers) */
void Embedding(int *in, float* w, float *out, size_t s, size_t H) ;
void Permute(float *in, float *out, size_t s, size_t H);
void Conv1D(float *in, float *w, float *b, float *out, size_t s, size_t C, size_t OC, size_t K, cudaStream_t stream);
void GetMax(float *in, float *out, size_t C, size_t s, cudaStream_t stream);
void Concat(float *in1, float *in2, float *in3, float *in4, 
            float *out, size_t N1, size_t N2, size_t N3, size_t N4, cudaStream_t stream);
void Linear(float *in, float *w, float *b, float *out, int N, int M, cudaStream_t stream);

/* Example of using CUDA kernel */
void Linear_ReLU(float *in, float *w, float *b, float *out, int N, int M, cudaStream_t stream);