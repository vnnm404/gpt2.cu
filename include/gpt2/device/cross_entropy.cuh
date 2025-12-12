#pragma once

#include <cuda_runtime.h>

__device__ void cross_entropy_forward_device(float *losses, const float *probs,
                                             const int *target, int batch_size,
                                             int seq_len, int vocab_size,
                                             int blockIdx_x) {
  int idx = blockIdx_x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * seq_len;

  if (idx < total_elements) {
    int t = idx % seq_len;
    int batch_idx = idx / seq_len;

    // Pointer to probability vector for this (batch, time) position
    const float *prob_bt =
        probs + batch_idx * seq_len * vocab_size + t * vocab_size;

    int target_token = target[batch_idx * seq_len + t];
    float prob = prob_bt[target_token];

    // Compute cross-entropy loss: -log(prob)
    losses[idx] = -logf(fmaxf(prob, 1e-10f)); // Avoid log(0)
  }
}

__device__ void cross_entropy_backward_device(float *g_logits, float *probs,
                                              const int *targets,
                                              int batch_size, int seq_len,
                                              int vocab_size, int blockIdx_x) {
  int idx = blockIdx_x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * seq_len;

  if (idx < total_elements) {
    int t = idx % seq_len;
    int batch_idx = idx / seq_len;

    // Pointers to logits gradient and probs for this (batch, time) position
    float *dlogits_bt =
        g_logits + batch_idx * seq_len * vocab_size + t * vocab_size;
    float *probs_bt = probs + batch_idx * seq_len * vocab_size + t * vocab_size;

    // Compute dloss directly (fused from init kernel)
    float dloss = 1.0f / (batch_size * seq_len);
    int target_token = targets[batch_idx * seq_len + t];

    // Compute gradient for all vocab positions
    for (int i = 0; i < vocab_size; i++) {
      float p = probs_bt[i];
      float indicator = (i == target_token) ? 1.0f : 0.0f;
      dlogits_bt[i] += (p - indicator) * dloss;
    }
  }
}