#ifndef GPT2_LAYERS_EMBEDDING_H
#define GPT2_LAYERS_EMBEDDING_H

/* Embedding layer structures and functions will be defined here */

#include <cuda_runtime.h>

__global__ void embedding_forward(float *out, const int *input_tokens, const float *wte, const float *wpe, int seq_len, int n_embd, int n_positions);

#endif /* GPT2_LAYERS_EMBEDDING_H */
