#ifndef GPT2_LAYERS_LAYERNORM_H
#define GPT2_LAYERS_LAYERNORM_H

#include <cuda_runtime.h>

/* Layer normalization structures and functions will be defined here */

__global__ void layernorm_forward(float *out, float *input, float *weight, float *bias, float *ln_mean, float *ln_rstd, int seq_len, int n_embd);
__global__ void layernorm_backward(float *g_input, float *g_weight, float *g_bias, const float *g_out, const float *input, const float *weight, const float *ln_mean, const float *ln_rstd, int batch_size, int seq_len, int n_embd);

#endif /* GPT2_LAYERS_LAYERNORM_H */
