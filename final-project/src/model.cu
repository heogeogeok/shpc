#include <mpi.h>
#include <omp.h>

#include <cstdio>

#include "layer.h"
#include "model.h"

int mpi_rank, mpi_size;

/** SECTION: DEBUGGING **/
#define DEBUG 0
#if DEBUG == 1
double dbg_start_time, dbg_ce_init, dbg_ce_final;
#define DEBUG_PRINT(...) do { \
  printf("(%s|rank=%d) ", mpi_rank); \
  printf(__VA_ARGS__); \
} while (0)
#else
#define DEBUG_PRINT(...)
#endif

/* [Model Parameters]
 * _w: Weight parameter
 * _b: Bias parameter
 */
Parameter *emb_w;
Parameter *conv0_w, *conv0_b;
Parameter *conv1_w, *conv1_b;
Parameter *conv2_w, *conv2_b;
Parameter *conv3_w, *conv3_b;
Parameter *linear0_w, *linear0_b;
Parameter *linear1_w, *linear1_b;
Parameter *linear2_w, *linear2_b;
Parameter *linear3_w, *linear3_b;

static float *emb_w_d;
static float *conv0_w_d, *conv0_b_d;
static float *conv1_w_d, *conv1_b_d;
static float *conv2_w_d, *conv2_b_d;
static float *conv3_w_d, *conv3_b_d;
static float *linear0_w_d, *linear0_b_d;
static float *linear1_w_d, *linear1_b_d;
static float *linear2_w_d, *linear2_b_d;
static float *linear3_w_d, *linear3_b_d;

void check_gpu_memory() {
    size_t free_mem, total_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU Memory: Free = %.2f MB, Total = %.2f MB\n", 
           free_mem / 1024.0 / 1024.0, total_mem / 1024.0 / 1024.0);
}

void alloc_and_set_parameters(float *param, size_t param_size) {
  size_t pos = 0;

  emb_w = new Parameter({21635, 4096}, param + pos);
  pos += 21635 * 4096; 

  conv0_w = new Parameter({1024, 4096, 3}, param + pos);
  pos += 1024 * 4096 * 3; 
  conv0_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv1_w = new Parameter({1024, 4096, 5}, param + pos);
  pos += 1024 * 4096 * 5; 
  conv1_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv2_w = new Parameter({1024, 4096, 7}, param + pos);
  pos += 1024 * 4096 * 7;
  conv2_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv3_w = new Parameter({1024, 4096, 9}, param + pos);
  pos += 1024 * 4096 * 9;
  conv3_b = new Parameter({1024}, param + pos);
  pos += 1024;

  linear0_w = new Parameter({2048, 4096}, param + pos);
  pos += 2048 * 4096;
  linear0_b = new Parameter({2048}, param + pos);
  pos += 2048;

  linear1_w = new Parameter({1024, 2048}, param + pos);
  pos += 1024 * 2048;
  linear1_b = new Parameter({1024}, param + pos);
  pos += 1024;

  linear2_w = new Parameter({512, 1024}, param + pos);
  pos += 512 * 1024;
  linear2_b = new Parameter({512}, param + pos);
  pos += 512;

  linear3_w = new Parameter({2, 512}, param + pos);
  pos += 2 * 512;
  linear3_b = new Parameter({2}, param + pos);
  pos += 2;

  if (pos != param_size) {
    fprintf(stderr, "Parameter size mismatched: %zu != %zu\n", 
            pos, param_size);
    exit(EXIT_FAILURE);
  }

  CHECK_CUDA(cudaMalloc(&emb_w_d, 21635 * 4096 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&conv0_w_d, 1024 * 4096 * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&conv0_b_d, 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&conv1_w_d, 1024 * 4096 * 5 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&conv1_b_d, 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&conv2_w_d, 1024 * 4096 * 7 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&conv2_b_d, 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&conv3_w_d, 1024 * 4096 * 9 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&conv3_b_d, 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&linear0_w_d, 2048 * 4096 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&linear0_b_d, 2048 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&linear1_w_d, 1024 * 2048 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&linear1_b_d, 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&linear2_w_d, 512 * 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&linear2_b_d, 512 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&linear3_w_d, 2 * 512 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&linear3_b_d, 2 * sizeof(float)));


  CHECK_CUDA(cudaMemcpyAsync(emb_w_d, emb_w->buf, 21635 * 4096 * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpyAsync(conv0_w_d, conv0_w->buf, 1024 * 4096 * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpyAsync(conv0_b_d, conv0_b->buf, 1024 * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpyAsync(conv1_w_d, conv1_w->buf, 1024 * 4096 * 5 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpyAsync(conv1_b_d, conv1_b->buf, 1024 * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpyAsync(conv2_w_d, conv2_w->buf, 1024 * 4096 * 7 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpyAsync(conv2_b_d, conv2_b->buf, 1024 * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpyAsync(conv3_w_d, conv3_w->buf, 1024 * 4096 * 9 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpyAsync(conv3_b_d, conv3_b->buf, 1024 * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpyAsync(linear0_w_d, linear0_w->buf, 2048 * 4096 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpyAsync(linear0_b_d, linear0_b->buf, 2048 * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpyAsync(linear1_w_d, linear1_w->buf, 1024 * 2048 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpyAsync(linear1_b_d, linear1_b->buf, 1024 * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpyAsync(linear2_w_d, linear2_w->buf, 512 * 1024 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpyAsync(linear2_b_d, linear2_b->buf, 512 * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpyAsync(linear3_w_d, linear3_w->buf, 2 * 512 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpyAsync(linear3_b_d, linear3_b->buf, 2 * sizeof(float), cudaMemcpyHostToDevice));
}

void free_parameters() {
  delete emb_w;
  delete conv0_w;
  delete conv0_b;
  delete conv1_w;
  delete conv1_b;
  delete conv2_w;
  delete conv2_b;
  delete conv3_w;
  delete conv3_b;
  delete linear0_w;
  delete linear0_b;
  delete linear1_w;
  delete linear1_b;
  delete linear2_w;
  delete linear2_b;
  delete linear3_w;
  delete linear3_b;

  CHECK_CUDA(cudaFree(emb_w_d));
  CHECK_CUDA(cudaFree(conv0_w_d));
  CHECK_CUDA(cudaFree(conv0_b_d));
  CHECK_CUDA(cudaFree(conv1_w_d));
  CHECK_CUDA(cudaFree(conv1_b_d));
  CHECK_CUDA(cudaFree(conv2_w_d));
  CHECK_CUDA(cudaFree(conv2_b_d));
  CHECK_CUDA(cudaFree(conv3_w_d));
  CHECK_CUDA(cudaFree(conv3_b_d));
  CHECK_CUDA(cudaFree(linear0_w_d));
  CHECK_CUDA(cudaFree(linear0_b_d));
  CHECK_CUDA(cudaFree(linear1_w_d));
  CHECK_CUDA(cudaFree(linear1_b_d));
  CHECK_CUDA(cudaFree(linear3_w_d));
  CHECK_CUDA(cudaFree(linear3_b_d));
}

/* [Model Activations] 
 * _a: Activation buffer
 */
Activation *emb_a;
Activation *permute_a;
Activation *conv0_a, *relu0_a, *pool0_a;
Activation *conv1_a, *relu1_a, *pool1_a;
Activation *conv2_a, *relu2_a, *pool2_a;
Activation *conv3_a, *relu3_a, *pool3_a;
Activation *concat_a;
Activation *linear0_a, *linear1_a, *linear2_a, *linear3_a;

static float *emb_a_d;
static float *permute_a_d;
static float *conv0_a_d, *pool0_a_d;
static float *conv1_a_d, *pool1_a_d;
static float *conv2_a_d, *pool2_a_d;
static float *conv3_a_d, *pool3_a_d;
static float *concat_a_d;
static float *linear0_a_d;
static float *linear1_a_d;
static float *linear2_a_d;
static float *linear3_a_d;

void alloc_activations() {
  emb_a = new Activation({SEQ_LEN, 4096});
  permute_a = new Activation({4096, SEQ_LEN});
  conv0_a = new Activation({1024, SEQ_LEN - 2});
  pool0_a = new Activation({1024});
  conv1_a = new Activation({1024, SEQ_LEN - 4});
  pool1_a = new Activation({1024});
  conv2_a = new Activation({1024, SEQ_LEN - 6});
  pool2_a = new Activation({1024});
  conv3_a = new Activation({1024, SEQ_LEN - 8});
  pool3_a = new Activation({1024});
  concat_a = new Activation({4096});
  linear0_a = new Activation({2048});
  linear1_a = new Activation({1024});
  linear2_a = new Activation({512});
  linear3_a = new Activation({2});

  CHECK_CUDA(cudaMalloc(&emb_a_d, SEQ_LEN * 4096 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&permute_a_d, 4096 * SEQ_LEN * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&conv0_a_d, 1024 * (SEQ_LEN - 2) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&pool0_a_d, 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&conv1_a_d, 1024 * (SEQ_LEN - 4) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&pool1_a_d, 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&conv2_a_d, 1024 * (SEQ_LEN - 6) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&pool2_a_d, 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&conv3_a_d, 1024 * (SEQ_LEN - 8) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&pool3_a_d, 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&concat_a_d, 4096 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&linear0_a_d, 2048 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&linear1_a_d, 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&linear2_a_d, 512 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&linear3_a_d, 2 * sizeof(float)));
}

void free_activations() {
  delete emb_a;
  delete permute_a;
  delete conv0_a;
  delete pool0_a;
  delete conv1_a;
  delete pool1_a;
  delete conv2_a;
  delete pool2_a;
  delete conv3_a;
  delete pool3_a;
  delete concat_a;
  delete linear0_a;
  delete linear1_a;
  delete linear2_a;
  delete linear3_a;

  CHECK_CUDA(cudaFree(emb_a_d));
  CHECK_CUDA(cudaFree(permute_a_d));
  CHECK_CUDA(cudaFree(conv0_a_d));
  CHECK_CUDA(cudaFree(pool0_a_d));
  CHECK_CUDA(cudaFree(conv1_a_d));
  CHECK_CUDA(cudaFree(pool1_a_d));
  CHECK_CUDA(cudaFree(conv2_a_d));
  CHECK_CUDA(cudaFree(pool2_a_d));
  CHECK_CUDA(cudaFree(conv3_a_d));
  CHECK_CUDA(cudaFree(pool3_a_d));
  CHECK_CUDA(cudaFree(concat_a_d));
  CHECK_CUDA(cudaFree(linear0_a_d));
  CHECK_CUDA(cudaFree(linear1_a_d));
  CHECK_CUDA(cudaFree(linear2_a_d));
  CHECK_CUDA(cudaFree(linear3_a_d));
}

/* [Model Computation: Sentiment Analysis Task] */
void predict_sentiment(int *inputs, float *outputs, size_t n_samples) {
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if (mpi_rank == 0) {
    /* Predict sentiment for each sentence */
    int *d_inputs;
    size_t input_size = n_samples * SEQ_LEN * sizeof(int);
    CHECK_CUDA(cudaMalloc(&d_inputs, input_size));
    CHECK_CUDA(cudaMemcpy(d_inputs, inputs, input_size, cudaMemcpyHostToDevice));

    for (size_t n = 0; n < n_samples; n++){
      /* Load a sentence from the inputs */
      int *single_input = d_inputs + n * SEQ_LEN;

      // Embedding
      Embedding(single_input, emb_w_d, emb_a_d, SEQ_LEN, 4096);

      // Permute
      Permute(emb_a_d, permute_a_d, SEQ_LEN, 4096);

      // Conv1D and GetMax
      Conv1D(permute_a_d, conv0_w_d, conv0_b_d, conv0_a_d, SEQ_LEN, 4096, 1024, 3);
      GetMax(conv0_a_d, pool0_a_d, 1024, SEQ_LEN - 2);
      Conv1D(permute_a_d, conv1_w_d, conv1_b_d, conv1_a_d, SEQ_LEN, 4096, 1024, 5);
      GetMax(conv1_a_d, pool1_a_d, 1024, SEQ_LEN - 4);
      Conv1D(permute_a_d, conv2_w_d, conv2_b_d, conv2_a_d, SEQ_LEN, 4096, 1024, 7);
      GetMax(conv2_a_d, pool2_a_d, 1024, SEQ_LEN - 6);
      Conv1D(permute_a_d, conv3_w_d, conv3_b_d, conv3_a_d, SEQ_LEN, 4096, 1024, 9);
      GetMax(conv3_a_d, pool3_a_d, 1024, SEQ_LEN - 8);

      // Concat
      Concat(pool0_a_d, pool1_a_d, pool2_a_d, pool3_a_d, concat_a_d, 1024, 1024, 1024, 1024);

      // Fully Connected Layers
      Linear_ReLU(concat_a_d, linear0_w_d, linear0_b_d, linear0_a_d, 4096, 2048);
      Linear_ReLU(linear0_a_d, linear1_w_d, linear1_b_d, linear1_a_d, 2048, 1024);
      Linear_ReLU(linear1_a_d, linear2_w_d, linear2_b_d, linear2_a_d, 1024, 512);
      Linear(linear2_a_d, linear3_w_d, linear3_b_d, linear3_a_d, 512, 2);


      // Copy the computation result to the outputs
      CHECK_CUDA(cudaMemcpy(outputs + n * 2, linear3_a_d, 2 * sizeof(float), cudaMemcpyDeviceToHost));
      // memcpy(outputs + n * 2, linear3_a->buf, 2 * sizeof(float));
    }
  }
}