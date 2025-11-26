/* MLP layer implementation - C implementation */

#include "gpt2/layers/mlp.h"

// out: [batch_size, seq_len, output_dim]
// input: [batch_size, seq_len, input_dim]
// w: [input_dim, output_dim]
// b: [output_dim]
// Launch with: mlp_forward<<<num_blocks, threads_per_block>>>(...)
// where num_blocks = (batch_size * seq_len * output_dim + threads_per_block - 1) / threads_per_block
// and threads_per_block = 256 (or similar)
__global__ void mlp_forward(float *out, const float *input, const float *w, const float *b, int batch_size, int seq_len, int input_dim, int output_dim) {
    // Global thread index across all blocks
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_elements = batch_size * seq_len * output_dim;
    
    if (idx < total_elements) {
        // Decompose linear index into (batch, time, output_channel)
        int o = idx % output_dim;
        int t = (idx / output_dim) % seq_len;
        int batch_idx = idx / (seq_len * output_dim);
        
        // Pointer to input vector for this (batch, time) position
        const float *inp_bt = input + batch_idx * seq_len * input_dim + t * input_dim;
        
        // Compute dot product: input[:] @ w[:, o]
        // weights are stored as [input_dim, output_dim], so column o is at w[i * output_dim + o]
        float val = (b == NULL) ? 0.0f : b[o];
        for (int i = 0; i < input_dim; i++) {
            val += inp_bt[i] * w[i * output_dim + o];
        }
        
        // Write output
        out[idx] = val;
    }
}

__device__ void mlp_forward_mk(float *out, const float *input, const float *w, const float *b, int batch_size, int seq_len, int input_dim, int output_dim, int bidx) {
    // Global thread index across all blocks
    int idx = bidx * blockDim.x + threadIdx.x;
    
    int total_elements = batch_size * seq_len * output_dim;
    
    if (idx < total_elements) {
        // Decompose linear index into (batch, time, output_channel)
        int o = idx % output_dim;
        int t = (idx / output_dim) % seq_len;
        int batch_idx = idx / (seq_len * output_dim);
        
        // Pointer to input vector for this (batch, time) position
        const float *inp_bt = input + batch_idx * seq_len * input_dim + t * input_dim;
        
        // Compute dot product: input[:] @ w[:, o]
        // weights are stored as [input_dim, output_dim], so column o is at w[i * output_dim + o]
        float val = (b == NULL) ? 0.0f : b[o];
        for (int i = 0; i < input_dim; i++) {
            val += inp_bt[i] * w[i * output_dim + o];
        }
        
        // Write output
        out[idx] = val;
    }
}

