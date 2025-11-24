/* Attention layer implementation - C implementation */

#include "gpt2/layers/attention.h"
#include <stdio.h>

// out: [batch_size, seq_len, 3 * n_embd]
// input: [batch_size, seq_len, n_embd]
// weights: [n_embd, 3 * n_embd]
// bias: [3 * n_embd]
// Launch with: qkv_projection<<<num_blocks, threads_per_block>>>(...)
// where num_blocks = (batch_size * seq_len * 3 * n_embd + threads_per_block - 1) / threads_per_block
// and threads_per_block = 256 (or similar)
__global__ void qkv_projection(float *out, const float *input, const float *weights, const float *bias, int batch_size, int seq_len, int n_embd) {
    // Global thread index across all blocks
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int OC = 3 * n_embd;
    int total_elements = batch_size * seq_len * OC;
    
    if (idx < total_elements) {
        // Decompose linear index into (batch, time, output_channel)
        int o = idx % OC;
        int t = (idx / OC) % seq_len;
        int b = idx / (seq_len * OC);
        
        // Pointer to input vector for this (batch, time) position
        const float *inp_bt = input + b * seq_len * n_embd + t * n_embd;
        
        // Compute dot product: input[:] @ weights[:, o]
        // weights are stored as [n_embd, 3*n_embd], so column o is at weights[i * OC + o]
        float val = bias[o];
        for (int i = 0; i < n_embd; i++) {
            val += inp_bt[i] * weights[i * OC + o];
        }
        
        // Write output
        // printf("out[%d, %d, %d] = %f\n", b, t, o, val);
        out[idx] = val;
    }
}
