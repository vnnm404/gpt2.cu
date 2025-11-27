#include "gpt2/layers/cross_entropy.h"

__global__ void cross_entropy_forward(float *losses, const float *probs, const int *target, int batch_size, int seq_len, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len;

    if (idx < total_elements) {
        int t = idx % seq_len;
        int batch_idx = idx / seq_len;

        // Pointer to probability vector for this (batch, time) position
        const float *prob_bt = probs + batch_idx * seq_len * vocab_size + t * vocab_size;

        int target_token = target[batch_idx * seq_len + t];
        float prob = prob_bt[target_token];

        // Compute cross-entropy loss: -log(prob)
        losses[idx] = -logf(fmaxf(prob, 1e-10f)); // Avoid log(0)
    }
}
