# Optimizing CNN Model with CUDA

## 1. Optimization Methods

-   [x] Calculate each operator with CUDA: `Embedding`, `Conv1D`, `Permute`, `ReLU`, `GetMax`, `Linear`, etc.

    -   [x] Create CUDA version of each operators
        -   `Embedding`: Naive
        -   `Permute`: Navie
        -   `Conv1D`: Tiling
        -   `ReLU`: All merged into the other operators
        -   `GetMax`: Naive
        -   `Linear`: Tiling

-   [x] Create weakly fused operators: `Conv1D_ReLU`, `Linear_ReLU`

    -   [x] `Conv1D_ReLU`: integrated into `Conv1D`.
    -   [x] `Linear_ReLU`: integrated into `Linear`.

-   [x] Caculate each opeartor with multi-gpu
-   [x] Synchronously offload input to other nodes using MPI
-   [ ] Asynchronously offload input to other nodes using MPI
-   [ ] Calculate multiple batches at once
        
## 2. Optimization History

-   Baseline: 0.12 (sentences/sec)
-   GPU computation: 95.89 (sentences/sec)
-   Multi-GPU: 260.12 (sentences/sec)
-   Stream: 272.71 (sentences/sec)
-   Linear kernel code: 303.07 (sentences/sec)
-   MPI offload: 1100.58 (sentences/sec)
-   Convolution kernel code: 2826.95 (sentences/sec)
-   Linear kernel code: 3584.62 (sentences/sec)
-   Hyperparemeter: 4965.41 (sentences/sec)
