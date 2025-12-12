#pragma once

#include <cuda_runtime.h>

__device__ float warpReduceMax(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

// out: [batch_size, seq_len, dim]
// input: [batch_size, seq_len, dim]
// Each block handles one (batch, seq_len) position
// Threads cooperate to compute softmax over dim
__device__ void softmax_forward_device(float *out, const float *input,
                                       int batch_size, int seq_len, int dim,
                                       int blockIdx_x, float *shared_mem) {
  // Each block processes one (batch, time) position
  int bt = blockIdx_x;
  if (bt >= batch_size * seq_len)
    return;

  const float *inp_bt = input + bt * dim;
  float *out_bt = out + bt * dim;

  // Shared memory for block-level reductions
  // __shared__ float shared_max[32];  // One per warp
  // __shared__ float shared_sum[32];
  float *shared_max = shared_mem;      // size 32
  float *shared_sum = shared_mem + 32; // size 32

  int tid = threadIdx.x;
  int warpId = tid / 32;
  int laneId = tid % 32;

  // Phase 1: Find maximum value for numerical stability
  float thread_max = -INFINITY;
  for (int i = tid; i < dim; i += blockDim.x) {
    thread_max = fmaxf(thread_max, inp_bt[i]);
  }

  // Warp-level reduction
  float warp_max = warpReduceMax(thread_max);

  // First thread in each warp writes to shared memory
  if (laneId == 0) {
    shared_max[warpId] = warp_max;
  }
  __syncthreads();

  // Final reduction across warps (first warp only)
  if (warpId == 0) {
    float val =
        (laneId < (blockDim.x + 31) / 32) ? shared_max[laneId] : -INFINITY;
    val = warpReduceMax(val);
    if (laneId == 0) {
      shared_max[0] = val;
    }
  }
  __syncthreads();

  float max_val = shared_max[0];

  // Phase 2: Compute sum of exponentials
  float thread_sum = 0.0f;
  for (int i = tid; i < dim; i += blockDim.x) {
    thread_sum += expf(inp_bt[i] - max_val);
  }

  // Warp-level reduction
  float warp_sum = warpReduceSum(thread_sum);

  // First thread in each warp writes to shared memory
  if (laneId == 0) {
    shared_sum[warpId] = warp_sum;
  }
  __syncthreads();

  // Final reduction across warps (first warp only)
  if (warpId == 0) {
    float val = (laneId < (blockDim.x + 31) / 32) ? shared_sum[laneId] : 0.0f;
    val = warpReduceSum(val);
    if (laneId == 0) {
      shared_sum[0] = val;
    }
  }
  __syncthreads();

  float sum_exp = shared_sum[0];

  // Phase 3: Compute and write softmax output
  for (int i = tid; i < dim; i += blockDim.x) {
    out_bt[i] = expf(inp_bt[i] - max_val) / sum_exp;
  }
}