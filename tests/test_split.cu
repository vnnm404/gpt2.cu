#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gpt2/split.cuh"

config_t config = {.vocab_size = 50257,
                   .batch_size = 4, // B = 4 from test
                   .n_layer = 12,
                   .n_head = 12,
                   .n_embd = 768,
                   .n_positions = 1024,
                   .n_ctx = 1024};

gpt2_t model;
gpt2_t g_model;

train_buffers_t buffers;
train_buffers_t g_buffers;

adamw_state_t opt_state;

int main(int argc, char *argv[]) {
  printf("GPT-2 Training Test\n");

  // Initialize models
  if (gpt2_initialize(&model, &config) != 0) {
    fprintf(stderr, "Failed to initialize GPT-2 model\n");
    return 1;
  }

  if (gpt2_initialize(&g_model, &config) != 0) {
    fprintf(stderr, "Failed to initialize GPT-2 gradient model\n");
    gpt2_free(&model);
    return 1;
  }

  // Load model weights
  FILE *model_file = fopen("../models/gpt2-124M-weights.bin", "rb");
  if (model_file == NULL) {
    fprintf(stderr, "Error opening model file\n");
    gpt2_free(&model);
    gpt2_free(&g_model);
    return 1;
  }

  if (gpt2_load_weights(&model, model_file) != 0) {
    fprintf(stderr, "Failed to load GPT-2 weights\n");
    fclose(model_file);
    gpt2_free(&model);
    gpt2_free(&g_model);
    return 1;
  }
  fclose(model_file);

  printf("Model loaded successfully.\n");

  // Load debug state file
  FILE *state_file = fopen("../models/gpt2_124M_debug_state.bin", "rb");
  if (state_file == NULL) {
    fprintf(stderr, "Error opening state file\n");
    gpt2_free(&model);
    gpt2_free(&g_model);
    return 1;
  }

  int state_header[256];
  size_t items_read = fread(state_header, sizeof(int), 256, state_file);
  if (items_read != 256) {
    fprintf(stderr, "Failed to read state header\n");
    fclose(state_file);
    gpt2_free(&model);
    gpt2_free(&g_model);
    return 1;
  }
  if (state_header[0] != 20240327) {
    fprintf(stderr, "Bad magic state file\n");
    fclose(state_file);
    gpt2_free(&model);
    gpt2_free(&g_model);
    return 1;
  }
  // if (state_header[1] != 1) {
  //     fprintf(stderr, "Bad version in state file\n");
  //     fclose(state_file);
  //     gpt2_free(&model);
  //     gpt2_free(&g_model);
  //     return 1;
  // }
  printf("Version: %d\n", state_header[1]);

  int B = state_header[2];
  int T = state_header[3];
  printf("[State]\n");
  printf("batch_size: %d\n", B);
  printf("seq_len: %d\n", T);

  // Verify batch size matches
  if (B != config.batch_size) {
    fprintf(stderr, "Batch size mismatch: config=%d, state=%d\n",
            config.batch_size, B);
    fclose(state_file);
    gpt2_free(&model);
    gpt2_free(&g_model);
    return 1;
  }

  int V = config.vocab_size;

  // Allocate CPU memory for inputs
  int *x = (int *)malloc(B * T * sizeof(int));
  int *y = (int *)malloc(B * T * sizeof(int));

  // Read inputs from state file (skip logits, loss, and gradients)
  items_read = fread(x, sizeof(int), B * T, state_file);
  if (items_read != B * T) {
    fprintf(stderr, "Failed to read input x\n");
    fclose(state_file);
    free(x);
    free(y);
    gpt2_free(&model);
    gpt2_free(&g_model);
    return 1;
  }
  items_read = fread(y, sizeof(int), B * T, state_file);
  if (items_read != B * T) {
    fprintf(stderr, "Failed to read input y\n");
    fclose(state_file);
    free(x);
    free(y);
    gpt2_free(&model);
    gpt2_free(&g_model);
    return 1;
  }

  // Skip expected_logits, expected_loss, and expected_grads
  fseek(state_file, B * T * V * sizeof(float), SEEK_CUR); // skip logits
  fseek(state_file, sizeof(float), SEEK_CUR);             // skip loss
  fseek(state_file, model.num_parameters * sizeof(float),
        SEEK_CUR); // skip grads

  fclose(state_file);

  // Copy inputs to GPU
  int *d_input_tokens, *d_target_tokens;
  gpuErrchk(cudaMalloc(&d_input_tokens, B * T * sizeof(int)));
  gpuErrchk(cudaMalloc(&d_target_tokens, B * T * sizeof(int)));
  gpuErrchk(cudaMemcpy(d_input_tokens, x, B * T * sizeof(int),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_target_tokens, y, B * T * sizeof(int),
                       cudaMemcpyHostToDevice));

  // Setup training buffers
  setup_train_buffers(config, &buffers, T);
  setup_train_buffers(config, &g_buffers, T);

  // Initialize optimizer state
  opt_state.learning_rate = 1e-4f;
  opt_state.beta1 = 0.9f;
  opt_state.beta2 = 0.999f;
  opt_state.eps = 1e-8f;
  opt_state.weight_decay = 0.01f;
  opt_state.t = 0;
  opt_state.m_memory = NULL;
  opt_state.v_memory = NULL;

  float losses[10];

  // Run 10 training iterations
  for (int step = 0; step < 10; step++) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    gpt2_zero_grad(&g_model);
    zero_activation_grads(config, &g_buffers);

    forward(config, model, buffers, d_input_tokens, T);
    cross_entropy(config, buffers, d_target_tokens, T);
    backward(config, model, buffers, g_model, g_buffers, d_input_tokens,
             d_target_tokens, T);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    float mean_loss = compute_mean_loss(&buffers.losses, B, T);

    gpt2_update(&model, &g_model, &opt_state);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("step %d: loss %f (took %f ms)\n", step, mean_loss,
           time_elapsed_s * 1000);
    losses[step] = mean_loss;
  }

  // Check expected losses
  float expected_losses[10] = {5.270007133483887,  4.059706687927246,
                               3.3751230239868164, 2.8007826805114746,
                               2.315382242202759,  1.8490285873413086,
                               1.3946564197540283, 0.9991465210914612,
                               0.6240804195404053, 0.37651097774505615};

  int allok = 1;
  for (int i = 0; i < 10; i++) {
    if (fabs(losses[i] - expected_losses[i]) >= 1e-2) {
      printf("LOSS MISMATCH AT STEP %d: %f %f\n", i, losses[i],
             expected_losses[i]);
      allok = 0;
    } else {
      printf("loss ok at step %d: %f %f\n", i, losses[i], expected_losses[i]);
    }
  }

  printf("overall okay: %d\n", allok);

  free(x);
  free(y);
  cudaFree(d_input_tokens);
  cudaFree(d_target_tokens);
  free_train_buffers(&buffers);
  free_train_buffers(&g_buffers);
  if (opt_state.m_memory)
    cudaFree(opt_state.m_memory);
  if (opt_state.v_memory)
    cudaFree(opt_state.v_memory);
  gpt2_free(&model);
  gpt2_free(&g_model);

  return allok ? 0 : 1;
}
