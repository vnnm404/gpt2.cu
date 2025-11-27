#ifndef GPT2_LAYERS_CROSS_ENTROPY_H
#define GPT2_LAYERS_CROSS_ENTROPY_H

#include <cuda_runtime.h>

/* Cross-entropy layer structures and functions will be defined here */

__global__ void cross_entropy_forward(float *losses, const float *probs, const int *target, int batch_size, int seq_len, int vocab_size);

#endif /* GPT2_LAYERS_CROSS_ENTROPY_H */
