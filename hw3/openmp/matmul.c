#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

void matmul(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
    int ii, jj, kk, i, j, k;

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (ii = 0; ii < M; ii += M / num_threads)
        for (kk = 0; kk < K; kk += K / num_threads)
            for (jj = 0; jj < N; jj += N / num_threads)
                for (i = ii; i < min((ii + M / num_threads), M); i++)
                    for (k = kk; k < min((kk + K / num_threads), K); k++)
                        for (j = jj; j < min((jj + N / num_threads), N); j++)
                            C[i * N + j] += A[i * K + k] * B[k * N + j];
}
