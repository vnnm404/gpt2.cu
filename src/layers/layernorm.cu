/* Layer normalization implementation - Optimized CUDA with shared memory
 *
 * Uses parallel reduction across n_embd dimension for computing mean/variance
 * Each block handles one (batch, seq) position
 */

#include "gpt2/layers/layernorm.h"

#define LN_WARP_SIZE 32

// Warp-level reduction for sum
__device__ __forceinline__ float ln_warp_reduce_sum(float val)
{
    for (int offset = LN_WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
__device__ float ln_block_reduce_sum(float val, float *smem, int tid, int num_threads)
{
    int warp_id = tid / LN_WARP_SIZE;
    int lane_id = tid % LN_WARP_SIZE;
    int num_warps = (num_threads + LN_WARP_SIZE - 1) / LN_WARP_SIZE;

    // Warp-level reduction
    val = ln_warp_reduce_sum(val);

    // Store warp results to shared memory
    if (lane_id == 0)
    {
        smem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0)
    {
        val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        val = ln_warp_reduce_sum(val);
    }

    return val;
}

// Device function for layernorm forward pass - Megakernel compatible
// Uses shared memory for parallel reduction
// out: [batch_size, seq_len, n_embd]
// input: [batch_size, seq_len, n_embd]
// gamma: [n_embd]
// beta: [n_embd]
__device__ void layernorm_forward_device(float *out, float *input, float *weight, float *bias, float *ln_mean, float *ln_rstd,
                                         int seq_len, int n_embd,
                                         int blockIdx_x)
{
    int batch_idx = blockIdx_x / seq_len;
    int seq_idx = blockIdx_x % seq_len;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ float smem[];

    float *x = input + (batch_idx * seq_len * n_embd) + (seq_idx * n_embd);
    float *outx = out + (batch_idx * seq_len * n_embd) + (seq_idx * n_embd);

    // Step 1: Compute mean using parallel reduction
    float local_sum = 0.0f;
    for (int i = tid; i < n_embd; i += num_threads)
    {
        local_sum += x[i];
    }
    local_sum = ln_block_reduce_sum(local_sum, smem, tid, num_threads);

    __shared__ float s_mean;
    if (tid == 0)
    {
        s_mean = local_sum / n_embd;
    }
    __syncthreads();
    float mean = s_mean;

    // Step 2: Compute variance using parallel reduction
    float local_var = 0.0f;
    for (int i = tid; i < n_embd; i += num_threads)
    {
        float diff = x[i] - mean;
        local_var += diff * diff;
    }
    local_var = ln_block_reduce_sum(local_var, smem, tid, num_threads);

    __shared__ float s_rstd;
    if (tid == 0)
    {
        float variance = local_var / n_embd;
        float epsilon = 1e-5f;
        s_rstd = 1.0f / sqrtf(variance + epsilon);
    }
    __syncthreads();
    float rstd = s_rstd;

    // Step 3: Normalize and scale
    for (int i = tid; i < n_embd; i += num_threads)
    {
        float n = (x[i] - mean) * rstd;
        outx[i] = weight[i] * n + bias[i];
    }

    // Store ln_mean and ln_rstd if provided
    if (tid == 0 && ln_mean != nullptr && ln_rstd != nullptr)
    {
        ln_mean[batch_idx * seq_len + seq_idx] = mean;
        ln_rstd[batch_idx * seq_len + seq_idx] = rstd;
    }
}

// out: [batch_size, seq_len, n_embd]
// input: [batch_size, seq_len, n_embd]
// gamma: [n_embd]
// beta: [n_embd]
__global__ void layernorm_forward(float *out, float *input, float *weight, float *bias, float *ln_mean, float *ln_rstd, int seq_len, int n_embd)
{
    layernorm_forward_device(out, input, weight, bias, ln_mean, ln_rstd, seq_len, n_embd, blockIdx.x);
}

// Device function for layernorm backward pass - Megakernel compatible
// Uses shared memory for parallel reductions
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
                                          int blockIdx_x)
{
    int batch_idx = blockIdx_x / seq_len;
    int seq_idx = blockIdx_x % seq_len;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ float smem[];

    // Get pointers to the current batch and sequence position
    const float *g_out_bt = g_out + batch_idx * seq_len * n_embd + seq_idx * n_embd;
    const float *input_bt = input + batch_idx * seq_len * n_embd + seq_idx * n_embd;
    float *g_input_bt = g_input + batch_idx * seq_len * n_embd + seq_idx * n_embd;

    float mean_bt = ln_mean[batch_idx * seq_len + seq_idx];
    float rstd_bt = ln_rstd[batch_idx * seq_len + seq_idx];

    // Step 1: Compute gnorm_mean and gnorm_norm_mean using parallel reduction
    float local_gnorm_sum = 0.0f;
    float local_gnorm_norm_sum = 0.0f;
    for (int i = tid; i < n_embd; i += num_threads)
    {
        float norm_bti = (input_bt[i] - mean_bt) * rstd_bt;
        float gnorm_i = weight[i] * g_out_bt[i];
        local_gnorm_sum += gnorm_i;
        local_gnorm_norm_sum += gnorm_i * norm_bti;
    }

    // Reduce gnorm_sum
    local_gnorm_sum = ln_block_reduce_sum(local_gnorm_sum, smem, tid, num_threads);
    __shared__ float s_gnorm_mean;
    if (tid == 0)
    {
        s_gnorm_mean = local_gnorm_sum / n_embd;
    }
    __syncthreads();

    // Reduce gnorm_norm_sum
    local_gnorm_norm_sum = ln_block_reduce_sum(local_gnorm_norm_sum, smem, tid, num_threads);
    __shared__ float s_gnorm_norm_mean;
    if (tid == 0)
    {
        s_gnorm_norm_mean = local_gnorm_norm_sum / n_embd;
    }
    __syncthreads();

    float gnorm_mean = s_gnorm_mean;
    float gnorm_norm_mean = s_gnorm_norm_mean;

    // Step 2: Compute gradients
    for (int i = tid; i < n_embd; i += num_threads)
    {
        float norm_bti = (input_bt[i] - mean_bt) * rstd_bt;
        float gnorm_i = weight[i] * g_out_bt[i];

        // Gradient contribution to bias (atomic add for thread-safe accumulation)
        atomicAdd(&g_bias[i], g_out_bt[i]);

        // Gradient contribution to weight (atomic add for thread-safe accumulation)
        atomicAdd(&g_weight[i], norm_bti * g_out_bt[i]);

        // Gradient contribution to input
        float gval = gnorm_i - gnorm_mean - norm_bti * gnorm_norm_mean;
        gval *= rstd_bt;
        g_input_bt[i] += gval;
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
__global__ void layernorm_backward(float *g_input, float *g_weight, float *g_bias, const float *g_out, const float *input, const float *weight, const float *ln_mean, const float *ln_rstd, int batch_size, int seq_len, int n_embd)
{
    layernorm_backward_device(g_input, g_weight, g_bias, g_out, input, weight, ln_mean, ln_rstd, batch_size, seq_len, n_embd, blockIdx.x);
}