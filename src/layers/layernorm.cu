/* Layer normalization implementation - C implementation */

#include "gpt2/layers/layernorm.h"

// out: [batch_size, seq_len, n_embd]
// input: [batch_size, seq_len, n_embd]
// gamma: [n_embd]
// beta: [n_embd]
__global__ void layernorm_forward(float *out, float *input, float *weight, float *bias, int seq_len, int n_embd) {
    int batch_idx = blockIdx.x;
    int seq_idx = threadIdx.x;

    if (seq_idx < n_embd) {
        float *x = input + (batch_idx * seq_len * n_embd) + (seq_idx * n_embd);

        // mean
        float mean = 0.0f;
        for (int i = 0; i < n_embd; i++) {
            mean += x[i];
        }
        mean /= n_embd;

        // variance
        float variance = 0.0f;
        for (int i = 0; i < n_embd; i++) {
            float diff = x[i] - mean;
            variance += diff * diff;
        }
        variance /= n_embd;

        // normalize and scale
        float epsilon = 1e-5f;
        float s = 1.0f / sqrtf(variance + epsilon);
        float *outx = out + (batch_idx * seq_len * n_embd) + (seq_idx * n_embd);

        for (int i = 0; i < n_embd; i++) {
            float n = (x[i] - mean) * s;
            outx[i] = weight[i] * n + bias[i];
        }
    }
}
