## Optimizing CNN Model with CUDA

## Applied Optimization Methods

-   [x] Calculate each operator with CUDA: `Embedding`, `Conv1D`, `Permute`, `ReLU`, `GetMax`, `Linear`, etc.

    -   [x] Create CUDA version of each operators
        -   `Conv1D`: Naive
        -   `Permute`: Naive
        -   `ReLU`: All merged into the other operators
        -   `GetMax`: Naive
        -   `Linear`: Naive
    -   [ ] Store most of intermediate features in global memory

-   [x] Create weakly fused operators: `Conv1D_ReLU`, `Linear_ReLU`, etc.

    -   [x] `Conv1D_ReLU`: integrated into `Conv1D`.
    -   [x] `Linear_ReLU`: integrated into `Linear`.

-   [x] Caculate each opeartor with multi-gpu
-   [ ] Calculate multiple batches at once
-   [x] Synchronously offload input to other nodes using MPI
-   [ ] Asynchronously offload input to other nodes using MPI

## Optimization History

-   Baseline: 0.12 (sentences/sec)
-   Synchronous offload: 8.19 (sentences/sec)
-   GPU computation: 95 (sentences/sec)
-   Multi-GPU: 260 (sentences/sec)
-   Stream: 272 (sentences/sec)
-   Linear kernel code: 303.073489 (sentences/sec)
-   Synchronously offload input to other nodes using MPI 1100.580692 (sentences/sec)
