#ifndef GPT2_LAYERS_MLP_H
#define GPT2_LAYERS_MLP_H

#include <cuda_runtime.h>

/* MLP layer structures and functions will be defined here */

__global__ void mlp_forward(float *out, const float *input, const float *w, const float *b, int batch_size, int seq_len, int input_dim, int output_dim);

__global__ void mlp_backward_input(float *g_input, const float *g_out, const float *weight, int batch_size, int seq_len, int input_dim, int output_dim);

__global__ void mlp_backward_weight(float *g_weight, float *g_bias, const float *g_out, const float *input, int batch_size, int seq_len, int input_dim, int output_dim);

#endif /* GPT2_LAYERS_MLP_H */
