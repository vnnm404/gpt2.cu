/* Embedding layer implementation - C implementation */

#include "gpt2/layers/embedding.h"
#include <cuda_runtime.h>

// Device function for embedding forward pass - Megakernel compatible
// output: [batch_size, seq_len, n_embd]
// input tokens: [batch_size, seq_len]
// wte: [n_embd, vocab_size]
// wpe: [n_positions, n_embd]
__device__ void embedding_forward_device(float *out, const int *input_tokens, const float *wte, const float *wpe, 
                                         int seq_len, int n_embd, int vocab_size, int n_positions,
                                         int blockIdx_x) {
    int batch_idx = blockIdx_x;
    int seq_idx = threadIdx.x;

    if (seq_idx < seq_len) {
        int token_id = input_tokens[batch_idx * seq_len + seq_idx];

        float *outx = out + (batch_idx * seq_len * n_embd) + (seq_idx * n_embd);
        for (int embd_idx = 0; embd_idx < n_embd; embd_idx++) {
            float token_embedding = wte[embd_idx * vocab_size + token_id];
            float position_embedding = wpe[seq_idx * n_embd + embd_idx];
            outx[embd_idx] = token_embedding + position_embedding;
        }
    }
}

// output: [batch_size, seq_len, n_embd]
// input tokens: [batch_size, seq_len]
// wte: [n_embd, vocab_size]
// wpe: [n_positions, n_embd]
__global__ void embedding_forward(float *out, const int *input_tokens, const float *wte, const float *wpe, int seq_len, int n_embd, int vocab_size, int n_positions) {
    int batch_idx = blockIdx.x;
    int seq_idx = threadIdx.x;

    if (seq_idx < seq_len) {
        int token_id = input_tokens[batch_idx * seq_len + seq_idx];

        float *outx = out + (batch_idx * seq_len * n_embd) + (seq_idx * n_embd);
        for (int embd_idx = 0; embd_idx < n_embd; embd_idx++) {
            float token_embedding = wte[embd_idx * vocab_size + token_id];
            float position_embedding = wpe[seq_idx * n_embd + embd_idx];
            outx[embd_idx] = token_embedding + position_embedding;
        }
    }
}

// Device function for embedding backward pass - Megakernel compatible
// g_wte: [C, vocab_size] gradients for token embeddings
// g_wpe: [n_positions, C] gradients for position embeddings
// g_out: [B, T, C] gradients from output
// inp: [B, T] input token indices
__device__ void embedding_backward_device(float *g_wte, float *g_wpe, const float *g_out, const int *inp, 
                                          int B, int T, int C,
                                          int blockIdx_x) {
    int idx = blockIdx_x * blockDim.x + threadIdx.x;
    int total_threads = B * T;
    
    if (idx >= total_threads) return;
    
    int b = idx / T;
    int t = idx % T;
    
    const float *g_out_bt = g_out + b * T * C + t * C;
    int ix = inp[b * T + t];
    float *g_wte_ix = g_wte + ix * C;
    float *g_wpe_t = g_wpe + t * C;
    
    for (int i = 0; i < C; i++) {
        float d = g_out_bt[i];
        atomicAdd(&g_wte_ix[i], d);
        atomicAdd(&g_wpe_t[i], d);
    }
}

// CUDA kernel for embedding backward pass
// g_wte: [C, vocab_size] gradients for token embeddings
// g_wpe: [n_positions, C] gradients for position embeddings
// g_out: [B, T, C] gradients from output
// inp: [B, T] input token indices
__global__ void embedding_backward(float *g_wte, float *g_wpe, const float *g_out, const int *inp, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T;
    
    if (idx >= total_threads) return;
    
    int b = idx / T;
    int t = idx % T;
    
    const float *g_out_bt = g_out + b * T * C + t * C;
    int ix = inp[b * T + t];
    float *g_wte_ix = g_wte + ix * C;
    float *g_wpe_t = g_wpe + t * C;
    
    for (int i = 0; i < C; i++) {
        float d = g_out_bt[i];
        atomicAdd(&g_wte_ix[i], d);
        atomicAdd(&g_wpe_t[i], d);
    }
}
