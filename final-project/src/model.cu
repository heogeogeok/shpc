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
  printf("(rank=%d) ", mpi_rank); \
  printf(__VA_ARGS__); \
} while (0)
#else
#define DEBUG_PRINT(...)
#endif

/** SECTION: GPU manipulation **/
cudaStream_t _gpu_stream_comp[NGPU][4];
cudaStream_t _gpu_stream_h2d[NGPU];
cudaStream_t _gpu_stream_d2h[NGPU];
cudaEvent_t _event_h2d[NGPU], _event_comp[NGPU][4], _event_d2h[NGPU];

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

static float *emb_w_d[NGPU];
static float *conv0_w_d[NGPU], *conv0_b_d[NGPU];
static float *conv1_w_d[NGPU], *conv1_b_d[NGPU];
static float *conv2_w_d[NGPU], *conv2_b_d[NGPU];
static float *conv3_w_d[NGPU], *conv3_b_d[NGPU];
static float *linear0_w_d[NGPU], *linear0_b_d[NGPU];
static float *linear1_w_d[NGPU], *linear1_b_d[NGPU];
static float *linear2_w_d[NGPU], *linear2_b_d[NGPU];
static float *linear3_w_d[NGPU], *linear3_b_d[NGPU];

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

  for(int i = 0; i < NGPU; ++i){
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&(emb_w_d[i]), 21635 * 4096 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(conv0_w_d[i]), 1024 * 4096 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(conv0_b_d[i]), 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(conv1_w_d[i]), 1024 * 4096 * 5 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(conv1_b_d[i]), 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(conv2_w_d[i]), 1024 * 4096 * 7 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(conv2_b_d[i]), 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(conv3_w_d[i]), 1024 * 4096 * 9 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(conv3_b_d[i]), 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(linear0_w_d[i]), 2048 * 4096 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(linear0_b_d[i]), 2048 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(linear1_w_d[i]), 1024 * 2048 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(linear1_b_d[i]), 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(linear2_w_d[i]), 512 * 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(linear2_b_d[i]), 512 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(linear3_w_d[i]), 2 * 512 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(linear3_b_d[i]), 2 * sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(emb_w_d[i], emb_w->buf, 21635 * 4096 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(conv0_w_d[i], conv0_w->buf, 1024 * 4096 * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(conv0_b_d[i], conv0_b->buf, 1024 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(conv1_w_d[i], conv1_w->buf, 1024 * 4096 * 5 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(conv1_b_d[i], conv1_b->buf, 1024 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(conv2_w_d[i], conv2_w->buf, 1024 * 4096 * 7 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(conv2_b_d[i], conv2_b->buf, 1024 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(conv3_w_d[i], conv3_w->buf, 1024 * 4096 * 9 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(conv3_b_d[i], conv3_b->buf, 1024 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(linear0_w_d[i], linear0_w->buf, 2048 * 4096 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(linear0_b_d[i], linear0_b->buf, 2048 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(linear1_w_d[i], linear1_w->buf, 1024 * 2048 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(linear1_b_d[i], linear1_b->buf, 1024 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(linear2_w_d[i], linear2_w->buf, 512 * 1024 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(linear2_b_d[i], linear2_b->buf, 512 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(linear3_w_d[i], linear3_w->buf, 2 * 512 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(linear3_b_d[i], linear3_b->buf, 2 * sizeof(float), cudaMemcpyHostToDevice));
  }

  for (int i = 0; i < NGPU; ++i) {
    CHECK_CUDA(cudaDeviceSynchronize());
  }
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

  for(int i = 0; i < NGPU; ++i){
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(emb_w_d[i]));
    CHECK_CUDA(cudaFree(conv0_w_d[i]));
    CHECK_CUDA(cudaFree(conv0_b_d[i]));
    CHECK_CUDA(cudaFree(conv1_w_d[i]));
    CHECK_CUDA(cudaFree(conv1_b_d[i]));
    CHECK_CUDA(cudaFree(conv2_w_d[i]));
    CHECK_CUDA(cudaFree(conv2_b_d[i]));
    CHECK_CUDA(cudaFree(conv3_w_d[i]));
    CHECK_CUDA(cudaFree(conv3_b_d[i]));
    CHECK_CUDA(cudaFree(linear0_w_d[i]));
    CHECK_CUDA(cudaFree(linear0_b_d[i]));
    CHECK_CUDA(cudaFree(linear1_w_d[i]));
    CHECK_CUDA(cudaFree(linear1_b_d[i]));
    CHECK_CUDA(cudaFree(linear3_w_d[i]));
    CHECK_CUDA(cudaFree(linear3_b_d[i]));
  }

}

/* [Model Activations] 
 * _a: Activation buffer
 */
static float *emb_a_d[NGPU];
static float *permute_a_d[NGPU];
static float *conv0_a_d[NGPU], *pool0_a_d[NGPU];
static float *conv1_a_d[NGPU], *pool1_a_d[NGPU];
static float *conv2_a_d[NGPU], *pool2_a_d[NGPU];
static float *conv3_a_d[NGPU], *pool3_a_d[NGPU];
static float *concat_a_d[NGPU];
static float *linear0_a_d[NGPU];
static float *linear1_a_d[NGPU];
static float *linear2_a_d[NGPU];
static float *linear3_a_d[NGPU];
static int *d_inputs[NGPU];

void alloc_activations() {
  for (int i = 0; i < NGPU; ++i) {
    CHECK_CUDA(cudaSetDevice(i));
    for (int j = 0; j < 4; ++j) {
        CHECK_CUDA(cudaStreamCreate(&_gpu_stream_comp[i][j]));
        CHECK_CUDA(cudaEventCreate(&_event_comp[i][j]));
    }
    CHECK_CUDA(cudaStreamCreate(&_gpu_stream_h2d[i]));
    CHECK_CUDA(cudaStreamCreate(&_gpu_stream_d2h[i]));
    CHECK_CUDA(cudaEventCreate(&_event_h2d[i]));
    CHECK_CUDA(cudaEventCreate(&_event_d2h[i]));
  }

  for(int i = 0 ; i < NGPU; ++i){
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&(d_inputs[i]), MAX_SAMPLES * SEQ_LEN * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&(emb_a_d[i]), SEQ_LEN * 4096 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(permute_a_d[i]), 4096 * SEQ_LEN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(conv0_a_d[i]), 1024 * (SEQ_LEN - 2) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(pool0_a_d[i]), 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(conv1_a_d[i]), 1024 * (SEQ_LEN - 4) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(pool1_a_d[i]), 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(conv2_a_d[i]), 1024 * (SEQ_LEN - 6) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(pool2_a_d[i]), 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(conv3_a_d[i]), 1024 * (SEQ_LEN - 8) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(pool3_a_d[i]), 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(concat_a_d[i]), 4096 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(linear0_a_d[i]), 2048 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(linear1_a_d[i]), 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(linear2_a_d[i]), 512 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&(linear3_a_d[i]), 2 * sizeof(float)));
  }
}

void free_activations() {
  for(int i = 0; i < NGPU; ++i){
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(emb_a_d[i]));
    CHECK_CUDA(cudaFree(permute_a_d[i]));
    CHECK_CUDA(cudaFree(conv0_a_d[i]));
    CHECK_CUDA(cudaFree(pool0_a_d[i]));
    CHECK_CUDA(cudaFree(conv1_a_d[i]));
    CHECK_CUDA(cudaFree(pool1_a_d[i]));
    CHECK_CUDA(cudaFree(conv2_a_d[i]));
    CHECK_CUDA(cudaFree(pool2_a_d[i]));
    CHECK_CUDA(cudaFree(conv3_a_d[i]));
    CHECK_CUDA(cudaFree(pool3_a_d[i]));
    CHECK_CUDA(cudaFree(concat_a_d[i]));
    CHECK_CUDA(cudaFree(linear0_a_d[i]));
    CHECK_CUDA(cudaFree(linear1_a_d[i]));
    CHECK_CUDA(cudaFree(linear2_a_d[i]));
    CHECK_CUDA(cudaFree(linear3_a_d[i]));  
    for (int j = 0; j < 4; ++j) {
        CHECK_CUDA(cudaStreamDestroy(_gpu_stream_comp[i][j]));
        CHECK_CUDA(cudaEventDestroy(_event_comp[i][j]));
    }
    CHECK_CUDA(cudaStreamDestroy(_gpu_stream_h2d[i]));
    CHECK_CUDA(cudaStreamDestroy(_gpu_stream_d2h[i]));
    CHECK_CUDA(cudaEventDestroy(_event_h2d[i]));
    CHECK_CUDA(cudaEventDestroy(_event_d2h[i]));
  }
}

/* [Model Computation: Sentiment Analysis Task] */
void predict_sentiment(int *inputs, float *outputs, size_t n_samples) {

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  size_t inputs_size = MAX_SAMPLES * SEQ_LEN * sizeof(int);
  size_t outputs_size = MAX_SAMPLES * 2 * sizeof(float);

  if (mpi_rank != 0)
  {
    inputs = (int *) malloc(inputs_size);
    outputs = (float *) malloc(outputs_size);
  }

  MPI_Scatter(inputs, n_samples * SEQ_LEN / NNODE, MPI_INT,
                inputs, n_samples * SEQ_LEN / NNODE, MPI_INT,
                0, MPI_COMM_WORLD);

  // Predict sentiment for each sentence 
  #pragma omp parallel for num_threads(NGPU)
  for (int gpu_idx = 0; gpu_idx < NGPU; gpu_idx++){
    int start_idx = gpu_idx * (n_samples / NNODE / NGPU);
    int end_idx = (gpu_idx + 1) * (n_samples / NNODE / NGPU);

    CHECK_CUDA(cudaSetDevice(gpu_idx));
    CHECK_CUDA(cudaMemcpyAsync(d_inputs[gpu_idx], inputs + start_idx * SEQ_LEN, 
                          (n_samples / NNODE / NGPU) * SEQ_LEN * sizeof(int), cudaMemcpyHostToDevice, _gpu_stream_h2d[gpu_idx]));

    CHECK_CUDA(cudaEventRecord(_event_h2d[gpu_idx], _gpu_stream_h2d[gpu_idx]));
    for (int j = 0; j < 4; j++) {
      CHECK_CUDA(cudaStreamWaitEvent(_gpu_stream_comp[gpu_idx][j], _event_h2d[gpu_idx], 0));
    }

    for (int n = start_idx; n < end_idx; n++) {
      int *single_input = d_inputs[gpu_idx] + (n - start_idx) * SEQ_LEN;
      // Embedding
      Embedding(single_input, emb_w_d[gpu_idx], emb_a_d[gpu_idx], SEQ_LEN, 4096);

      // Permute
      Permute(emb_a_d[gpu_idx], permute_a_d[gpu_idx], SEQ_LEN, 4096);

      // Conv1D and GetMax
      Conv1D_K3(permute_a_d[gpu_idx], conv0_w_d[gpu_idx], conv0_b_d[gpu_idx], conv0_a_d[gpu_idx], SEQ_LEN, 4096, 1024, 3, _gpu_stream_comp[gpu_idx][0]);
      GetMax(conv0_a_d[gpu_idx], pool0_a_d[gpu_idx], 1024, SEQ_LEN - 2, _gpu_stream_comp[gpu_idx][0]);

      Conv1D_K5(permute_a_d[gpu_idx], conv1_w_d[gpu_idx], conv1_b_d[gpu_idx], conv1_a_d[gpu_idx], SEQ_LEN, 4096, 1024, 5, _gpu_stream_comp[gpu_idx][1]);
      GetMax(conv1_a_d[gpu_idx], pool1_a_d[gpu_idx], 1024, SEQ_LEN - 4, _gpu_stream_comp[gpu_idx][1]);

      Conv1D_K7(permute_a_d[gpu_idx], conv2_w_d[gpu_idx], conv2_b_d[gpu_idx], conv2_a_d[gpu_idx], SEQ_LEN, 4096, 1024, 7, _gpu_stream_comp[gpu_idx][2]);
      GetMax(conv2_a_d[gpu_idx], pool2_a_d[gpu_idx], 1024, SEQ_LEN - 6, _gpu_stream_comp[gpu_idx][2]);

      Conv1D_K9(permute_a_d[gpu_idx], conv3_w_d[gpu_idx], conv3_b_d[gpu_idx], conv3_a_d[gpu_idx], SEQ_LEN, 4096, 1024, 9, _gpu_stream_comp[gpu_idx][3]);
      GetMax(conv3_a_d[gpu_idx], pool3_a_d[gpu_idx], 1024, SEQ_LEN - 8, _gpu_stream_comp[gpu_idx][3]);

      for (int j = 0; j < 4; j++) {
        CHECK_CUDA(cudaEventRecord(_event_comp[gpu_idx][j], _gpu_stream_comp[gpu_idx][j]));
        CHECK_CUDA(cudaStreamWaitEvent(_gpu_stream_comp[gpu_idx][0], _event_comp[gpu_idx][j], 0));
      }

      // Concat 
      Concat(pool0_a_d[gpu_idx], pool1_a_d[gpu_idx], pool2_a_d[gpu_idx], pool3_a_d[gpu_idx], concat_a_d[gpu_idx], 1024, 1024, 1024, 1024, _gpu_stream_comp[gpu_idx][0]);

      // Fully Connected Layers
      Linear_ReLU(concat_a_d[gpu_idx], linear0_w_d[gpu_idx], linear0_b_d[gpu_idx], 
                  linear0_a_d[gpu_idx], 4096, 2048, _gpu_stream_comp[gpu_idx][0]);

      Linear_ReLU(linear0_a_d[gpu_idx], linear1_w_d[gpu_idx], linear1_b_d[gpu_idx], 
                  linear1_a_d[gpu_idx], 2048, 1024, _gpu_stream_comp[gpu_idx][0]);

      Linear_ReLU(linear1_a_d[gpu_idx], linear2_w_d[gpu_idx], linear2_b_d[gpu_idx], 
                  linear2_a_d[gpu_idx], 1024, 512, _gpu_stream_comp[gpu_idx][0]);

      Linear(linear2_a_d[gpu_idx], linear3_w_d[gpu_idx], linear3_b_d[gpu_idx], 
            linear3_a_d[gpu_idx], 512, 2, _gpu_stream_comp[gpu_idx][0]);

      CHECK_CUDA(cudaEventRecord(_event_d2h[gpu_idx], _gpu_stream_comp[gpu_idx][0]));
      CHECK_CUDA(cudaStreamWaitEvent(_gpu_stream_d2h[gpu_idx], _event_d2h[gpu_idx], 0));

      // Copy the computation result to the outputs
      CHECK_CUDA(cudaMemcpyAsync(outputs + n * N_CLASSES, linear3_a_d[gpu_idx], 
                            N_CLASSES * sizeof(float), cudaMemcpyDeviceToHost, _gpu_stream_d2h[gpu_idx]));
    }
  }
  for (int i = 0; i < NGPU; ++i) {
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  MPI_Gather(outputs, n_samples * N_CLASSES / NNODE, MPI_FLOAT,
               outputs, n_samples * N_CLASSES / NNODE, MPI_FLOAT,
               0, MPI_COMM_WORLD);
  if (mpi_rank != 0)
  {
    free(inputs);
    free(outputs);
  }
}