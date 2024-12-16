## Optimizing CNN Model with CUDA

## Applied Optimization Methods

-   [x] Synchronously offload input to other nodes using MPI
-   [ ] Asynchronously offload input to other nodes using MPI
-   [ ] Calculate multiple batches at once
-   [ ] Calculate each operators with CUDA: `Embedding`, `Conv1D`, `Permute`, `ReLU`, `GetMax`, `Linear`, etc.

    -   [ ] Create CUDA version of each operators
        -   `Conv1D`: Rectangular blocking
        -   `Permute`: Naive
        -   `ReLU`: All merged into the other operators
        -   `GetMax`: Naive
        -   `Linear`: Naive
    -   [ ] Store most of intermediate features in global memory

-   [ ] Create weakly fused operators: `Conv1D_ReLU`, `Linear_ReLU`, etc.

    -   [ ] `Conv1D_ReLU`: integrated into `Conv1D`.
    -   [ ] `Linear_ReLU`: integrated into `Linear`.

## Optimization History

-   Baseline: 0.12 input(s)/sec
-   Synchronous offload: 8.19 input(s)/sec
