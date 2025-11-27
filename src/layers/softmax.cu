/* Softmax layer implementation - C implementation */

#include "gpt2/layers/softmax.h"

// out: [batch_size, seq_len, dim]
// input: [batch_size, seq_len, dim]
__global__ void softmax_forward(float *out, const float *input, int batch_size, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * dim;

    if (idx < total_elements) {
        int d = idx % dim;
        int t = (idx / dim) % seq_len;
        int batch_idx = idx / (seq_len * dim);

        // Pointer to input vector for this (batch, time) position
        const float *inp_bt = input + batch_idx * seq_len * dim + t * dim;

        // Compute max for numerical stability
        float max_val = inp_bt[0];
        for (int i = 1; i < dim; i++) {
            if (inp_bt[i] > max_val) {
                max_val = inp_bt[i];
            }
        }

        // Compute sum of exp
        float sum_exp = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum_exp += expf(inp_bt[i] - max_val);
        }

        // Compute softmax output
        out[idx] = expf(inp_bt[d] - max_val) / sum_exp;
    }
}
