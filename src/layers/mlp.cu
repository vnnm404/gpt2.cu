/* MLP layer implementation - Optimized CUDA implementation */

#include "gpt2/layers/mlp.h"

// Device function for MLP forward pass - Megakernel compatible
// out: [batch_size, seq_len, output_dim]
// input: [batch_size, seq_len, input_dim]
// w: [input_dim, output_dim]
// b: [output_dim]
// shared_mem: pointer to shared memory (size: 2 * TILE_SIZE * TILE_SIZE * sizeof(float))
// Launch with: dim3 grid((output_dim + TILE_SIZE - 1) / TILE_SIZE, (batch_size * seq_len + TILE_SIZE - 1) / TILE_SIZE);
//              dim3 block(TILE_SIZE * TILE_SIZE);
__device__ void mlp_forward_device(float *out, const float *input, const float *w, const float *b, 
                                    int batch_size, int seq_len, int input_dim, int output_dim,
                                    int blockIdx_x, int blockIdx_y, float *shared_mem) {
    // Use provided shared memory for tiling
    float (*s_input)[TILE_SIZE] = (float (*)[TILE_SIZE])shared_mem;
    float (*s_weight)[TILE_SIZE] = (float (*)[TILE_SIZE])(shared_mem + TILE_SIZE * TILE_SIZE);
    
    // Thread indices within block - extract 2D from 1D
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    
    // Global output position
    int out_col = blockIdx_x * TILE_SIZE + tx;  // output dimension
    int out_row = blockIdx_y * TILE_SIZE + ty;  // batch * seq_len
    
    // Compute output element with tiling over input_dim
    float acc = 0.0f;
    
    int num_tiles = (input_dim + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load tile of input
        int input_col = tile * TILE_SIZE + tx;
        if (out_row < batch_size * seq_len && input_col < input_dim) {
            s_input[ty][tx] = input[out_row * input_dim + input_col];
        } else {
            s_input[ty][tx] = 0.0f;
        }
        
        // Load tile of weight
        int weight_row = tile * TILE_SIZE + ty;
        if (weight_row < input_dim && out_col < output_dim) {
            s_weight[ty][tx] = w[weight_row * output_dim + out_col];
        } else {
            s_weight[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += s_input[ty][k] * s_weight[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write output with bias
    if (out_row < batch_size * seq_len && out_col < output_dim) {
        float bias_val = (b == NULL) ? 0.0f : b[out_col];
        out[out_row * output_dim + out_col] = acc + bias_val;
    }
}

// Optimized forward pass using shared memory tiling
// out: [batch_size, seq_len, output_dim]
// input: [batch_size, seq_len, input_dim]
// w: [input_dim, output_dim]
// b: [output_dim]
// Launch with: dim3 grid((output_dim + TILE_SIZE - 1) / TILE_SIZE, (batch_size * seq_len + TILE_SIZE - 1) / TILE_SIZE);
//              dim3 block(TILE_SIZE * TILE_SIZE);
__global__ void mlp_forward(float *out, const float *input, const float *w, const float *b, int batch_size, int seq_len, int input_dim, int output_dim) {
    // Shared memory for tiling
    __shared__ float s_input[TILE_SIZE][TILE_SIZE];
    __shared__ float s_weight[TILE_SIZE][TILE_SIZE];
    
    // Thread indices within block - extract 2D from 1D
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    
    // Global output position
    int out_col = blockIdx.x * TILE_SIZE + tx;  // output dimension
    int out_row = blockIdx.y * TILE_SIZE + ty;  // batch * seq_len
    
    // Compute output element with tiling over input_dim
    float acc = 0.0f;
    
    int num_tiles = (input_dim + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load tile of input
        int input_col = tile * TILE_SIZE + tx;
        if (out_row < batch_size * seq_len && input_col < input_dim) {
            s_input[ty][tx] = input[out_row * input_dim + input_col];
        } else {
            s_input[ty][tx] = 0.0f;
        }
        
        // Load tile of weight
        int weight_row = tile * TILE_SIZE + ty;
        if (weight_row < input_dim && out_col < output_dim) {
            s_weight[ty][tx] = w[weight_row * output_dim + out_col];
        } else {
            s_weight[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += s_input[ty][k] * s_weight[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write output with bias
    if (out_row < batch_size * seq_len && out_col < output_dim) {
        float bias_val = (b == NULL) ? 0.0f : b[out_col];
        out[out_row * output_dim + out_col] = acc + bias_val;
    }
}

// Device function for MLP backward pass into input - Megakernel compatible
// g_input: [batch_size, seq_len, input_dim] - gradient w.r.t. input
// g_out: [batch_size, seq_len, output_dim] - gradient w.r.t. output
// weight: [input_dim, output_dim] - forward pass weights
// shared_mem: pointer to shared memory (size: 2 * TILE_SIZE * TILE_SIZE * sizeof(float))
// Computes: g_input = g_out @ weight^T
// Launch with: dim3 grid((input_dim + TILE_SIZE - 1) / TILE_SIZE, (batch_size * seq_len + TILE_SIZE - 1) / TILE_SIZE);
//              dim3 block(TILE_SIZE * TILE_SIZE);
__device__ void mlp_backward_input_device(float *g_input, const float *g_out, const float *weight, 
                                          int batch_size, int seq_len, int input_dim, int output_dim,
                                          int blockIdx_x, int blockIdx_y, float *shared_mem) {
    float (*s_g_out)[TILE_SIZE] = (float (*)[TILE_SIZE])shared_mem;
    float (*s_weight)[TILE_SIZE] = (float (*)[TILE_SIZE])(shared_mem + TILE_SIZE * TILE_SIZE);
    
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    
    int in_col = blockIdx_x * TILE_SIZE + tx;   // input_dim
    int in_row = blockIdx_y * TILE_SIZE + ty;   // batch * seq_len
    
    float acc = 0.0f;
    
    int num_tiles = (output_dim + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load tile of g_out
        int g_out_col = tile * TILE_SIZE + tx;
        if (in_row < batch_size * seq_len && g_out_col < output_dim) {
            s_g_out[ty][tx] = g_out[in_row * output_dim + g_out_col];
        } else {
            s_g_out[ty][tx] = 0.0f;
        }
        
        // Load tile of weight^T (transpose weight during load)
        int weight_col = tile * TILE_SIZE + ty;  // output_dim in original weight
        if (in_col < input_dim && weight_col < output_dim) {
            s_weight[ty][tx] = weight[in_col * output_dim + weight_col];
        } else {
            s_weight[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += s_g_out[ty][k] * s_weight[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write output (accumulate into g_input)
    if (in_row < batch_size * seq_len && in_col < input_dim) {
        atomicAdd(&g_input[in_row * input_dim + in_col], acc);
    }
}

// Optimized backward pass into input using shared memory tiling
// g_input: [batch_size, seq_len, input_dim] - gradient w.r.t. input
// g_out: [batch_size, seq_len, output_dim] - gradient w.r.t. output
// weight: [input_dim, output_dim] - forward pass weights
// Computes: g_input = g_out @ weight^T
// Launch with: dim3 grid((input_dim + TILE_SIZE - 1) / TILE_SIZE, (batch_size * seq_len + TILE_SIZE - 1) / TILE_SIZE);
//              dim3 block(TILE_SIZE * TILE_SIZE);
__global__ void mlp_backward_input(float *g_input, const float *g_out, const float *weight, int batch_size, int seq_len, int input_dim, int output_dim) {
    __shared__ float s_g_out[TILE_SIZE][TILE_SIZE];
    __shared__ float s_weight[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    
    int in_col = blockIdx.x * TILE_SIZE + tx;   // input_dim
    int in_row = blockIdx.y * TILE_SIZE + ty;   // batch * seq_len
    
    float acc = 0.0f;
    
    int num_tiles = (output_dim + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load tile of g_out
        int g_out_col = tile * TILE_SIZE + tx;
        if (in_row < batch_size * seq_len && g_out_col < output_dim) {
            s_g_out[ty][tx] = g_out[in_row * output_dim + g_out_col];
        } else {
            s_g_out[ty][tx] = 0.0f;
        }
        
        // Load tile of weight^T (transpose weight during load)
        int weight_col = tile * TILE_SIZE + ty;  // output_dim in original weight
        if (in_col < input_dim && weight_col < output_dim) {
            s_weight[ty][tx] = weight[in_col * output_dim + weight_col];
        } else {
            s_weight[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += s_g_out[ty][k] * s_weight[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write output (accumulate into g_input)
    if (in_row < batch_size * seq_len && in_col < input_dim) {
        atomicAdd(&g_input[in_row * input_dim + in_col], acc);
    }
}

// Device function for MLP backward pass into weight and bias - Megakernel compatible
// g_weight: [input_dim, output_dim] - gradient w.r.t. weight
// g_bias: [output_dim] - gradient w.r.t. bias
// g_out: [batch_size, seq_len, output_dim] - gradient w.r.t. output
// input: [batch_size, seq_len, input_dim] - forward pass input
// shared_mem: pointer to shared memory (size: 2 * TILE_SIZE * TILE_SIZE * sizeof(float))
// Computes: g_weight = input^T @ g_out
// Launch with: dim3 grid((output_dim + TILE_SIZE - 1) / TILE_SIZE, (input_dim + TILE_SIZE - 1) / TILE_SIZE);
//              dim3 block(TILE_SIZE * TILE_SIZE);
__device__ void mlp_backward_weight_device(float *g_weight, float *g_bias, const float *g_out, const float *input, 
                                           int batch_size, int seq_len, int input_dim, int output_dim,
                                           int blockIdx_x, int blockIdx_y, float *shared_mem) {
    float (*s_input)[TILE_SIZE] = (float (*)[TILE_SIZE])shared_mem;
    float (*s_g_out)[TILE_SIZE] = (float (*)[TILE_SIZE])(shared_mem + TILE_SIZE * TILE_SIZE);
    
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    
    int out_col = blockIdx_x * TILE_SIZE + tx;  // output_dim
    int out_row = blockIdx_y * TILE_SIZE + ty;  // input_dim
    
    float acc = 0.0f;
    
    int batch_seq = batch_size * seq_len;
    int num_tiles = (batch_seq + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load tile of input^T (transpose during load)
        int batch_seq_idx = tile * TILE_SIZE + tx;
        if (out_row < input_dim && batch_seq_idx < batch_seq) {
            s_input[ty][tx] = input[batch_seq_idx * input_dim + out_row];
        } else {
            s_input[ty][tx] = 0.0f;
        }
        
        // Load tile of g_out
        batch_seq_idx = tile * TILE_SIZE + ty;
        if (batch_seq_idx < batch_seq && out_col < output_dim) {
            s_g_out[ty][tx] = g_out[batch_seq_idx * output_dim + out_col];
        } else {
            s_g_out[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += s_input[ty][k] * s_g_out[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write weight gradient (accumulate)
    if (out_row < input_dim && out_col < output_dim) {
        atomicAdd(&g_weight[out_row * output_dim + out_col], acc);
    }
    
    // Compute bias gradient - use first row of threads
    if (ty == 0 && out_col < output_dim && g_bias != NULL) {
        float bias_grad = 0.0f;
        for (int bs = 0; bs < batch_seq; bs++) {
            bias_grad += g_out[bs * output_dim + out_col];
        }
        atomicAdd(&g_bias[out_col], bias_grad);
    }
}

// Optimized backward pass into weight and bias using shared memory tiling
// g_weight: [input_dim, output_dim] - gradient w.r.t. weight
// g_bias: [output_dim] - gradient w.r.t. bias
// g_out: [batch_size, seq_len, output_dim] - gradient w.r.t. output
// input: [batch_size, seq_len, input_dim] - forward pass input
// Computes: g_weight = input^T @ g_out
// Launch with: dim3 grid((output_dim + TILE_SIZE - 1) / TILE_SIZE, (input_dim + TILE_SIZE - 1) / TILE_SIZE);
//              dim3 block(TILE_SIZE * TILE_SIZE);
__global__ void mlp_backward_weight(float *g_weight, float *g_bias, const float *g_out, const float *input, int batch_size, int seq_len, int input_dim, int output_dim) {
    __shared__ float s_input[TILE_SIZE][TILE_SIZE];
    __shared__ float s_g_out[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    
    int out_col = blockIdx.x * TILE_SIZE + tx;  // output_dim
    int out_row = blockIdx.y * TILE_SIZE + ty;  // input_dim
    
    float acc = 0.0f;
    
    int batch_seq = batch_size * seq_len;
    int num_tiles = (batch_seq + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load tile of input^T (transpose during load)
        int batch_seq_idx = tile * TILE_SIZE + tx;
        if (out_row < input_dim && batch_seq_idx < batch_seq) {
            s_input[ty][tx] = input[batch_seq_idx * input_dim + out_row];
        } else {
            s_input[ty][tx] = 0.0f;
        }
        
        // Load tile of g_out
        batch_seq_idx = tile * TILE_SIZE + ty;
        if (batch_seq_idx < batch_seq && out_col < output_dim) {
            s_g_out[ty][tx] = g_out[batch_seq_idx * output_dim + out_col];
        } else {
            s_g_out[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += s_input[ty][k] * s_g_out[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write weight gradient (accumulate)
    if (out_row < input_dim && out_col < output_dim) {
        atomicAdd(&g_weight[out_row * output_dim + out_col], acc);
    }
    
    // Compute bias gradient - use first row of threads
    if (ty == 0 && out_col < output_dim && g_bias != NULL) {
        float bias_grad = 0.0f;
        for (int bs = 0; bs < batch_seq; bs++) {
            bias_grad += g_out[bs * output_dim + out_col];
        }
        atomicAdd(&g_bias[out_col], bias_grad);
    }
}
