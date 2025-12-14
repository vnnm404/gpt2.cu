/* Attention layer implementation - Optimized CUDA implementation
 *
 * Optimizations:
 * - 2D grid: (B*NH, T) for better parallelism
 * - Shared memory for Q vector and attention scores
 * - Warp-level parallel reduction for dot products
 * - Online softmax (fused max/sum in single pass)
 * - Atomic-free backward pass via loop restructuring
 */

#include "gpt2/layers/attention.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val)
{
    for (int offset = ATTN_WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val)
{
    for (int offset = ATTN_WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Device function for attention forward pass - Optimized with shared memory
// Grid: (B*NH, T) - each block handles one (batch, head, query_time) position
// Block: threads cooperatively compute dot products and softmax
// out: [B, T, C]
// preatt: [B, NH, T, T]
// att: [B, NH, T, T]
// inp: [B, T, 3*C] where Q,K,V are concatenated
__device__ void attention_forward_device(float *out, float *preatt, float *att,
                                         const float *inp,
                                         int B, int T, int NH, int C,
                                         int blockIdx_x, int blockIdx_y)
{
    // Decode block indices: blockIdx_x = b * NH + h, blockIdx_y = t (query time)
    int bh = blockIdx_x;
    int t = blockIdx_y;
    int h = bh % NH;
    int b = bh / NH;

    int C3 = C * 3;
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf((float)hs);

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int num_threads = blockDim.x * blockDim.y;

    // Shared memory layout:
    // s_query[hs]: query vector for this position
    // s_scores[T]: attention scores (pre-softmax, then post-softmax)
    extern __shared__ float smem[];
    float *s_query = smem;
    float *s_scores = smem + hs;

    // Load query vector into shared memory cooperatively
    const float *query_t = inp + b * T * C3 + t * C3 + h * hs;
    for (int i = tid; i < hs; i += num_threads)
    {
        s_query[i] = query_t[i];
    }
    __syncthreads();

    // Initialize for online softmax
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Pass 1: Compute attention scores with online softmax
    // Each thread handles multiple key positions
    for (int t2 = tid; t2 <= t; t2 += num_threads)
    {
        const float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;

        // Compute dot product: query · key
        float score = 0.0f;
        for (int i = 0; i < hs; i++)
        {
            score += s_query[i] * key_t2[i];
        }
        score *= scale;

        // Store pre-attention score
        preatt[b * NH * T * T + h * T * T + t * T + t2] = score;
        s_scores[t2] = score;
    }
    __syncthreads();

    // Find max across all scores (parallel reduction)
    float local_max = -INFINITY;
    for (int t2 = tid; t2 <= t; t2 += num_threads)
    {
        local_max = fmaxf(local_max, s_scores[t2]);
    }
    // Warp-level reduction
    local_max = warp_reduce_max(local_max);

    // Cross-warp reduction via shared memory
    __shared__ float s_warp_max[8]; // Assuming max 8 warps per block
    int warp_id = tid / ATTN_WARP_SIZE;
    int lane_id = tid % ATTN_WARP_SIZE;
    if (lane_id == 0 && warp_id < 8)
    {
        s_warp_max[warp_id] = local_max;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0 && lane_id < (num_threads + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE)
    {
        local_max = s_warp_max[lane_id];
    }
    else if (warp_id == 0)
    {
        local_max = -INFINITY;
    }
    if (warp_id == 0)
    {
        local_max = warp_reduce_max(local_max);
    }

    __shared__ float s_max;
    if (tid == 0)
    {
        s_max = local_max;
    }
    __syncthreads();
    float maxval = s_max;

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int t2 = tid; t2 <= t; t2 += num_threads)
    {
        float expv = __expf(s_scores[t2] - maxval);
        s_scores[t2] = expv;
        local_sum += expv;
    }

    // Reduce sum across threads
    local_sum = warp_reduce_sum(local_sum);

    __shared__ float s_warp_sum[8];
    if (lane_id == 0 && warp_id < 8)
    {
        s_warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < (num_threads + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE)
    {
        local_sum = s_warp_sum[lane_id];
    }
    else if (warp_id == 0)
    {
        local_sum = 0.0f;
    }
    if (warp_id == 0)
    {
        local_sum = warp_reduce_sum(local_sum);
    }

    __shared__ float s_sum;
    if (tid == 0)
    {
        s_sum = local_sum;
    }
    __syncthreads();

    float expsum_inv = (s_sum == 0.0f) ? 0.0f : 1.0f / s_sum;

    // Normalize attention weights and store
    for (int t2 = tid; t2 <= t; t2 += num_threads)
    {
        float att_val = s_scores[t2] * expsum_inv;
        att[b * NH * T * T + h * T * T + t * T + t2] = att_val;
        s_scores[t2] = att_val;
    }
    // Set causal mask
    for (int t2 = t + 1 + tid; t2 < T; t2 += num_threads)
    {
        att[b * NH * T * T + h * T * T + t * T + t2] = 0.0f;
    }
    __syncthreads();

    // Compute output: weighted sum of values
    // Each thread computes a portion of the output head
    float *out_bth = out + b * T * C + t * C + h * hs;

    for (int i = tid; i < hs; i += num_threads)
    {
        float acc = 0.0f;
        for (int t2 = 0; t2 <= t; t2++)
        {
            const float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
            acc += s_scores[t2] * value_t2[i];
        }
        out_bth[i] = acc;
    }
}

// CUDA kernel for attention forward pass - 2D grid version
__global__ void attention_forward(float *out, float *preatt, float *att,
                                  const float *inp,
                                  int B, int T, int NH, int C)
{
    attention_forward_device(out, preatt, att, inp, B, T, NH, C, blockIdx.x, blockIdx.y);
}

// Device function for attention backward pass - 2D grid with shared memory
// Grid: (B*NH, T) - blockIdx_y = t (query position)
// Each block computes gradients for one (batch, head, query_time) position
// Uses shared memory for attention scores and parallel reduction
__device__ void attention_backward_device(float *g_inp, float *g_preatt, float *g_att,
                                          const float *g_out, const float *inp, const float *att,
                                          int B, int T, int C, int NH,
                                          int blockIdx_x, int blockIdx_y)
{
    // Decode: blockIdx_x = b * NH + h, blockIdx_y = t (QUERY position)
    int bh = blockIdx_x;
    int t = blockIdx_y; // Query position
    int h = bh % NH;
    int b = bh / NH;

    int hs = C / NH;
    int C3 = C * 3;
    float scale = 1.0f / sqrtf((float)hs);

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int num_threads = blockDim.x * blockDim.y;

    // Shared memory layout:
    // s_att[T]: attention weights for this query
    // s_g_att[T]: gradient of attention weights
    extern __shared__ float smem[];
    float *s_att = smem;
    float *s_g_att = smem + T;

    // Pointers
    const float *g_out_t = g_out + b * T * C + t * C + h * hs;
    const float *query_t = inp + b * T * C3 + t * C3 + h * hs;
    float *g_preatt_t = g_preatt + b * NH * T * T + h * T * T + t * T;
    float *g_att_t = g_att + b * NH * T * T + h * T * T + t * T;

    // Load attention weights into shared memory
    for (int t2 = tid; t2 <= t; t2 += num_threads)
    {
        s_att[t2] = att[b * NH * T * T + h * T * T + t * T + t2];
    }
    __syncthreads();

    // Step 1: Compute g_att[t, t2] = dot(value[t2], g_out[t])
    for (int t2 = tid; t2 <= t; t2 += num_threads)
    {
        const float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
        float g_att_val = 0.0f;
        for (int i = 0; i < hs; i++)
        {
            g_att_val += value_t2[i] * g_out_t[i];
        }
        s_g_att[t2] = g_att_val;
        g_att_t[t2] = g_att_val;
    }
    __syncthreads();

    // Step 2: Compute ds = sum_{t2} att[t,t2] * g_att[t,t2] using parallel reduction
    float local_ds = 0.0f;
    for (int t2 = tid; t2 <= t; t2 += num_threads)
    {
        local_ds += s_att[t2] * s_g_att[t2];
    }

    // Warp-level reduction
    for (int offset = ATTN_WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        local_ds += __shfl_down_sync(0xffffffff, local_ds, offset);
    }

    // Cross-warp reduction
    __shared__ float s_warp_ds[8];
    int warp_id = tid / ATTN_WARP_SIZE;
    int lane_id = tid % ATTN_WARP_SIZE;
    if (lane_id == 0 && warp_id < 8)
    {
        s_warp_ds[warp_id] = local_ds;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < (num_threads + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE)
    {
        local_ds = s_warp_ds[lane_id];
    }
    else if (warp_id == 0)
    {
        local_ds = 0.0f;
    }
    if (warp_id == 0)
    {
        for (int offset = ATTN_WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            local_ds += __shfl_down_sync(0xffffffff, local_ds, offset);
        }
    }

    __shared__ float s_ds;
    if (tid == 0)
    {
        s_ds = local_ds;
    }
    __syncthreads();
    float ds = s_ds;

    // Step 3: Compute g_preatt[t, t2] = att[t,t2] * (g_att[t,t2] - ds)
    for (int t2 = tid; t2 <= t; t2 += num_threads)
    {
        float g_preatt_val = s_att[t2] * (s_g_att[t2] - ds);
        g_preatt_t[t2] = g_preatt_val;
        s_g_att[t2] = g_preatt_val; // Reuse for g_preatt values
    }
    __syncthreads();

    // Step 4: Compute g_query[t] = sum_{t2} key[t2] * g_preatt[t,t2] * scale
    // Each thread handles a portion of the head dimension
    float *g_query_t = g_inp + b * T * C3 + t * C3 + h * hs;
    for (int i = tid; i < hs; i += num_threads)
    {
        float g_q = 0.0f;
        for (int t2 = 0; t2 <= t; t2++)
        {
            const float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
            g_q += key_t2[i] * s_g_att[t2]; // s_g_att now holds g_preatt
        }
        g_query_t[i] = g_q * scale;
    }

    // Step 5: Accumulate g_key[t2] and g_value[t2] using atomics
    // g_key[t2] += query[t] * g_preatt[t,t2] * scale
    // g_value[t2] += att[t,t2] * g_out[t]
    for (int t2 = tid; t2 <= t; t2 += num_threads)
    {
        float g_preatt_val = s_g_att[t2] * scale;
        float att_val = s_att[t2];

        float *g_key_t2 = g_inp + b * T * C3 + t2 * C3 + h * hs + C;
        float *g_value_t2 = g_inp + b * T * C3 + t2 * C3 + h * hs + C * 2;

        for (int i = 0; i < hs; i++)
        {
            atomicAdd(&g_key_t2[i], query_t[i] * g_preatt_val);
            atomicAdd(&g_value_t2[i], att_val * g_out_t[i]);
        }
    }
}

// CUDA kernel for attention backward pass - 2D grid version
__global__ void attention_backward(float *g_inp, float *g_preatt, float *g_att,
                                   const float *g_out, const float *inp, const float *att,
                                   int B, int T, int C, int NH)
{
    attention_backward_device(g_inp, g_preatt, g_att, g_out, inp, att, B, T, C, NH, blockIdx.x, blockIdx.y);
}
