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

// Backward pass into input
// g_input: [batch_size, seq_len, input_dim] - gradient w.r.t. input
// g_out: [batch_size, seq_len, output_dim] - gradient w.r.t. output
// weight: [input_dim, output_dim] - forward pass weights
// Launch with: mlp_backward_input<<<num_blocks, threads_per_block>>>(...)
// where num_blocks = (batch_size * seq_len + threads_per_block - 1) / threads_per_block
// Parallelized over batch_size * seq_len
__global__ void mlp_backward_input(float *g_input, const float *g_out, const float *weight, int batch_size, int seq_len, int input_dim, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len;
    
    if (idx < total_elements) {
        // Decompose linear index into (batch, time)
        int t = idx % seq_len;
        int batch_idx = idx / seq_len;
        
        // Pointers to gradients for this (batch, time) position
        const float *g_out_bt = g_out + batch_idx * seq_len * output_dim + t * output_dim;
        float *g_input_bt = g_input + batch_idx * seq_len * input_dim + t * input_dim;
        
        // Compute gradient: g_input = g_out @ weight^T
        // For each input dimension i, accumulate contributions from all output dimensions
        for (int o = 0; o < output_dim; o++) {
            float g_o = g_out_bt[o];
            // weight row o is at weight[i * output_dim + o] for each i
            for (int i = 0; i < input_dim; i++) {
                g_input_bt[i] += weight[i * output_dim + o] * g_o;
            }
        }
    }
}

// Backward pass into weight and bias
// g_weight: [input_dim, output_dim] - gradient w.r.t. weight
// g_bias: [output_dim] - gradient w.r.t. bias
// g_out: [batch_size, seq_len, output_dim] - gradient w.r.t. output
// input: [batch_size, seq_len, input_dim] - forward pass input
// Launch with: mlp_backward_weight<<<num_blocks, threads_per_block>>>(...)
// where num_blocks = (output_dim + threads_per_block - 1) / threads_per_block
// Parallelized over output_dim
__global__ void mlp_backward_weight(float *g_weight, float *g_bias, const float *g_out, const float *input, int batch_size, int seq_len, int input_dim, int output_dim) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (o < output_dim) {
        // Accumulate gradients over all (batch, time) positions
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < seq_len; t++) {
                // Pointers for this (batch, time) position
                const float *g_out_bt = g_out + b * seq_len * output_dim + t * output_dim;
                const float *input_bt = input + b * seq_len * input_dim + t * input_dim;
                
                float g_o = g_out_bt[o];
                
                // Accumulate bias gradient
                if (g_bias != NULL) {
                    g_bias[o] += g_o;
                }
                
                // Accumulate weight gradient
                // g_weight[:, o] += input_bt[:] * g_o
                for (int i = 0; i < input_dim; i++) {
                    g_weight[i * output_dim + o] += input_bt[i] * g_o;
                }
            }
        }
    }
}
