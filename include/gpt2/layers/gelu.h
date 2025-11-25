#ifndef GPT2_LAYERS_GELU_H
#define GPT2_LAYERS_GELU_H

#include <cuda_runtime.h>

/* GELU layer structures and functions will be defined here */

__global__ void gelu_forward(float *out, const float *input, int batch_size, int seq_len, int n_embd);

#endif /* GPT2_LAYERS_GELU_H */
