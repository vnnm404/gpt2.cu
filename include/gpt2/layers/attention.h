#ifndef GPT2_LAYERS_ATTENTION_H
#define GPT2_LAYERS_ATTENTION_H

#include <cuda_runtime.h>

/* Attention layer structures and functions will be defined here */

__global__ void qkv_projection(float *out, const float *input, const float *weights, const float *bias, int batch_size, int seq_len, int n_embd);

#endif /* GPT2_LAYERS_ATTENTION_H */
