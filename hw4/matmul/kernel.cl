__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  int i = get_local_id(0);
  int j = get_local_id(1);

  int gi = 32 * get_group_id(0) + i;
  int gj = 32 * get_group_id(1) + j;

  int ntile = (K + 32 - 1) / 32;

  __local float tA[32][32];
  __local float tB[32][32];

  float acc = 0.0;
  for (int t = 0; t < ntile; t++) {
    int ti = gj * K + 32 * t + i;
    int tj = (32 * t + j) * N + gi;

    if (ti < M * K)
      tA[j][i] = A[ti];
    else
      tA[j][i] = 0.0;
    
    if (tj < K * N)
      tB[j][i] = B[tj];
    else
      tB[j][i] = 0.0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < 32; k++)
        acc += tA[j][k] * tB[k][i];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (gj < M && gi < N)
    C[gj * N + gi] = acc;
}