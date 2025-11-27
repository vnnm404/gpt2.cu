#ifndef GPT2_LAYERS_SOFTMAX_H
#define GPT2_LAYERS_SOFTMAX_H

#include <cuda_runtime.h>

/* Softmax layer structures and functions will be defined here */

__global__ void softmax_forward(float *out, const float *input, int batch_size, int seq_len, int dim);

#endif /* GPT2_LAYERS_SOFTMAX_H */
