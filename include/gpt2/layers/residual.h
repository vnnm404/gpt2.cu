#ifndef GPT2_LAYERS_RESIDUAL_H
#define GPT2_LAYERS_RESIDUAL_H

#include <cuda_runtime.h>

/* Residual layer structures and functions will be defined here */

__global__ void residual_forward(float *out, const float *input, const float *residual, int batch_size, int seq_len, int n_embd);

__device__ void residual_forward_mk(float *out, const float *input, const float *residual, int batch_size, int seq_len, int n_embd, int bidx);

#endif /* GPT2_LAYERS_RESIDUAL_H */
