#ifndef GPT2_LAYERS_ATTENTION_H
#define GPT2_LAYERS_ATTENTION_H

#include <cuda_runtime.h>

/* Attention layer structures and functions will be defined here */

__global__ void attention_forward(float *out, float *preatt, float *att, const float *input, int batch_size, int seq_len, int n_head, int n_embd);

#endif /* GPT2_LAYERS_ATTENTION_H */
