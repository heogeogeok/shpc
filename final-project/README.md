## Optimizing CNN Model with CUDA

## Applied Optimization Methods

-   [x] Calculate each operator with CUDA: `Embedding`, `Conv1D`, `Permute`, `ReLU`, `GetMax`, `Linear`, etc.

    -   [x] Create CUDA version of each operators
        -   `Conv1D`: Rectangular blocking
        -   `Permute`: Naive
        -   `ReLU`: All merged into the other operators
        -   `GetMax`: Naive
        -   `Linear`: Naive
    -   [ ] Store most of intermediate features in global memory

-   [x] Create weakly fused operators: `Conv1D_ReLU`, `Linear_ReLU`, etc.

    -   [x] `Conv1D_ReLU`: integrated into `Conv1D`.
    -   [x] `Linear_ReLU`: integrated into `Linear`.

-   [ ] Caculate each opeartor with multi-gpu
-   [ ] Calculate multiple batches at once
-   [ ] Calculate multiple batches at once
-   [ ] Synchronously offload input to other nodes using MPI
-   [ ] Asynchronously offload input to other nodes using MPI

## Optimization History

-   Baseline: 0.12 input(s)/sec
-   Synchronous offload: 8.19 input(s)/sec