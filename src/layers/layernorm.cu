/* Layer normalization implementation - C implementation */

#include "gpt2/layers/layernorm.h"

// Device function for layernorm forward pass - Megakernel compatible
// out: [batch_size, seq_len, n_embd]
// input: [batch_size, seq_len, n_embd]
// gamma: [n_embd]
// beta: [n_embd]
__device__ void layernorm_forward_device(float *out, float *input, float *weight, float *bias, float *ln_mean, float *ln_rstd, 
                                         int seq_len, int n_embd,
                                         int blockIdx_x) {
    int batch_idx = blockIdx_x;
    int seq_idx = threadIdx.x;

    // threadIdx.x indexes sequence positions (time steps), so compare against seq_len
    if (seq_idx < seq_len) {
        float *x = input + (batch_idx * seq_len * n_embd) + (seq_idx * n_embd);

        // ln_mean
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

        // Store ln_mean and ln_rstd if provided
        if (ln_mean != nullptr && ln_rstd != nullptr) {
            ln_mean[batch_idx * seq_len + seq_idx] = mean;
            ln_rstd[batch_idx * seq_len + seq_idx] = s;
        }
    }
}

// out: [batch_size, seq_len, n_embd]
// input: [batch_size, seq_len, n_embd]
// gamma: [n_embd]
// beta: [n_embd]
__global__ void layernorm_forward(float *out, float *input, float *weight, float *bias, float *ln_mean, float *ln_rstd, int seq_len, int n_embd) {
    int batch_idx = blockIdx.x;
    int seq_idx = threadIdx.x;

    // threadIdx.x indexes sequence positions (time steps), so compare against seq_len
    if (seq_idx < seq_len) {
        float *x = input + (batch_idx * seq_len * n_embd) + (seq_idx * n_embd);

        // ln_mean
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

        // Store ln_mean and ln_rstd if provided
        if (ln_mean != nullptr && ln_rstd != nullptr) {
            ln_mean[batch_idx * seq_len + seq_idx] = mean;
            ln_rstd[batch_idx * seq_len + seq_idx] = s;
        }
    }
}

// Device function for layernorm backward pass - Megakernel compatible
// g_input: [batch_size, seq_len, n_embd]
// g_weight: [n_embd]
// g_bias: [n_embd]
// g_out: [batch_size, seq_len, n_embd]
// input: [batch_size, seq_len, n_embd]
// ln_mean: [batch_size, seq_len]
// ln_rstd: [batch_size, seq_len]
// weight: [n_embd]
__device__ void layernorm_backward_device(float *g_input, float *g_weight, float *g_bias, const float *g_out, const float *input, const float *weight, const float *ln_mean, const float *ln_rstd, 
                                          int batch_size, int seq_len, int n_embd,
                                          int blockIdx_x) {
    int batch_idx = blockIdx_x;
    int seq_idx = threadIdx.x;

    if (seq_idx < seq_len) {
        // Get pointers to the current batch and sequence position
        const float *g_out_bt = g_out + batch_idx * seq_len * n_embd + seq_idx * n_embd;
        const float *input_bt = input + batch_idx * seq_len * n_embd + seq_idx * n_embd;
        float *g_input_bt = g_input + batch_idx * seq_len * n_embd + seq_idx * n_embd;
        
        float mean_bt = ln_mean[batch_idx * seq_len + seq_idx];
        float rstd_bt = ln_rstd[batch_idx * seq_len + seq_idx];

        // First: two reduce operations (gradient reductions)
        float gnorm_mean = 0.0f;
        float gnorm_norm_mean = 0.0f;
        for (int i = 0; i < n_embd; i++) {
            float norm_bti = (input_bt[i] - mean_bt) * rstd_bt;
            float gnorm_i = weight[i] * g_out_bt[i];
            gnorm_mean += gnorm_i;
            gnorm_norm_mean += gnorm_i * norm_bti;
        }
        gnorm_mean = gnorm_mean / n_embd;
        gnorm_norm_mean = gnorm_norm_mean / n_embd;

        // Now iterate again and accumulate all the gradients
        for (int i = 0; i < n_embd; i++) {
            float norm_bti = (input_bt[i] - mean_bt) * rstd_bt;
            float gnorm_i = weight[i] * g_out_bt[i];
            
            // Gradient contribution to bias (atomic add for thread-safe accumulation)
            atomicAdd(&g_bias[i], g_out_bt[i]);
            
            // Gradient contribution to weight (atomic add for thread-safe accumulation)
            atomicAdd(&g_weight[i], norm_bti * g_out_bt[i]);
            
            // Gradient contribution to input
            float gval = 0.0f;
            gval += gnorm_i; // term 1
            gval -= gnorm_mean; // term 2
            gval -= norm_bti * gnorm_norm_mean; // term 3
            gval *= rstd_bt; // final scale
            g_input_bt[i] += gval;
        }
    }
}

// g_input: [batch_size, seq_len, n_embd]
// g_weight: [n_embd]
// g_bias: [n_embd]
// g_out: [batch_size, seq_len, n_embd]
// input: [batch_size, seq_len, n_embd]
// ln_mean: [batch_size, seq_len]
// ln_rstd: [batch_size, seq_len]
// weight: [n_embd]
__global__ void layernorm_backward(float *g_input, float *g_weight, float *g_bias, const float *g_out, const float *input, const float *weight, const float *ln_mean, const float *ln_rstd, int batch_size, int seq_len, int n_embd) {
    int batch_idx = blockIdx.x;
    int seq_idx = threadIdx.x;

    if (seq_idx < seq_len) {
        // Get pointers to the current batch and sequence position
        const float *g_out_bt = g_out + batch_idx * seq_len * n_embd + seq_idx * n_embd;
        const float *input_bt = input + batch_idx * seq_len * n_embd + seq_idx * n_embd;
        float *g_input_bt = g_input + batch_idx * seq_len * n_embd + seq_idx * n_embd;
        
        float mean_bt = ln_mean[batch_idx * seq_len + seq_idx];
        float rstd_bt = ln_rstd[batch_idx * seq_len + seq_idx];

        // First: two reduce operations (gradient reductions)
        float gnorm_mean = 0.0f;
        float gnorm_norm_mean = 0.0f;
        for (int i = 0; i < n_embd; i++) {
            float norm_bti = (input_bt[i] - mean_bt) * rstd_bt;
            float gnorm_i = weight[i] * g_out_bt[i];
            gnorm_mean += gnorm_i;
            gnorm_norm_mean += gnorm_i * norm_bti;
        }
        gnorm_mean = gnorm_mean / n_embd;
        gnorm_norm_mean = gnorm_norm_mean / n_embd;

        // Now iterate again and accumulate all the gradients
        for (int i = 0; i < n_embd; i++) {
            float norm_bti = (input_bt[i] - mean_bt) * rstd_bt;
            float gnorm_i = weight[i] * g_out_bt[i];
            
            // Gradient contribution to bias (atomic add for thread-safe accumulation)
            atomicAdd(&g_bias[i], g_out_bt[i]);
            
            // Gradient contribution to weight (atomic add for thread-safe accumulation)
            atomicAdd(&g_weight[i], norm_bti * g_out_bt[i]);
            
            // Gradient contribution to input
            float gval = 0.0f;
            gval += gnorm_i; // term 1
            gval -= gnorm_mean; // term 2
            gval -= norm_bti * gnorm_norm_mean; // term 3
            gval *= rstd_bt; // final scale
            g_input_bt[i] += gval;
        }
    }
}