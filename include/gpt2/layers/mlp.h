#ifndef GPT2_LAYERS_MLP_H
#define GPT2_LAYERS_MLP_H

#include <cuda_runtime.h>

/* MLP layer structures and functions */

#define MLP_TILE_SIZE 32

// Helper macros for launching optimized MLP kernels
#define MLP_FORWARD_GRID(output_dim, batch_size, seq_len) \
    dim3(((output_dim) + MLP_TILE_SIZE - 1) / MLP_TILE_SIZE, \
         ((batch_size) * (seq_len) + MLP_TILE_SIZE - 1) / MLP_TILE_SIZE)

#define MLP_BACKWARD_INPUT_GRID(input_dim, batch_size, seq_len) \
    dim3(((input_dim) + MLP_TILE_SIZE - 1) / MLP_TILE_SIZE, \
         ((batch_size) * (seq_len) + MLP_TILE_SIZE - 1) / MLP_TILE_SIZE)

#define MLP_BACKWARD_WEIGHT_GRID(output_dim, input_dim) \
    dim3(((output_dim) + MLP_TILE_SIZE - 1) / MLP_TILE_SIZE, \
         ((input_dim) + MLP_TILE_SIZE - 1) / MLP_TILE_SIZE)

#define MLP_BLOCK_DIM dim3(MLP_TILE_SIZE, MLP_TILE_SIZE)

__global__ void mlp_forward(float *out, const float *input, const float *w, const float *b, int batch_size, int seq_len, int input_dim, int output_dim);

__global__ void mlp_backward_input(float *g_input, const float *g_out, const float *weight, int batch_size, int seq_len, int input_dim, int output_dim);

__global__ void mlp_backward_weight(float *g_weight, float *g_bias, const float *g_out, const float *input, int batch_size, int seq_len, int input_dim, int output_dim);

#endif /* GPT2_LAYERS_MLP_H */
