/* Embedding layer implementation - C implementation */

#include "gpt2/layers/embedding.h"
#include <cuda_runtime.h>

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
