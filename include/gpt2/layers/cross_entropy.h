#ifndef GPT2_LAYERS_CROSS_ENTROPY_H
#define GPT2_LAYERS_CROSS_ENTROPY_H

#include <cuda_runtime.h>

/* Cross-entropy layer structures and functions will be defined here */

__global__ void cross_entropy_forward(float *losses, const float *probs, const int *target, int batch_size, int seq_len, int vocab_size);
__global__ void cross_entropy_backward(float *dlogits, float *probs, const int *targets, int batch_size, int seq_len, int vocab_size);

__device__ void cross_entropy_forward_device(float *losses, const float *probs, const int *target, int batch_size, int seq_len, int vocab_size,
                                             int blockIdx_x);
__device__ void cross_entropy_backward_device(float *g_logits, float *probs, const int *targets, int batch_size, int seq_len, int vocab_size,
                                              int blockIdx_x);

#endif /* GPT2_LAYERS_CROSS_ENTROPY_H */
