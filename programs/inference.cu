/* GPT-2 inference executable - C implementation */

#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>

#include "gpt2/gpt2.h"
#include "gpt2/layers/embedding.h"
#include "gpt2/layers/layernorm.h"
#include "gpt2/layers/attention.h"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

config_t config = {
    .vocab_size = 50257,
    .padded_vocab_size = 50304,
    .n_layer = 12,
    .n_head = 12,
    .n_embd = 768,
    .n_positions = 1024,
    .n_ctx = 1024
};
gpt2_t model;

typedef struct {
    tensor_t *res;
    tensor_t *out;
} inference_buffers_t;

int prepare_input_tokens(int *input_tokens, int seq_len, int **d_input_tokens) {
    int *input_tokens_host = (int *) malloc(sizeof(int) * seq_len);
    memcpy(input_tokens_host, input_tokens, sizeof(int) * seq_len);
    
    cudaError_t err = cudaMalloc(d_input_tokens, sizeof(int) * seq_len);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for input tokens: %s\n", cudaGetErrorString(err));
        free(input_tokens_host);
        return -1;
    }
    
    err = cudaMemcpy(*d_input_tokens, input_tokens_host, sizeof(int) * seq_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed for input tokens: %s\n", cudaGetErrorString(err));
        cudaFree(*d_input_tokens);
        free(input_tokens_host);
        return -1;
    }
    
    free(input_tokens_host);
    return 0;
}

int setup_inference_buffers(inference_buffers_t *buffers) {
    int res_shape[] = {1, config.n_ctx, config.n_embd};
    buffers->res = tensor_alloc(3, res_shape);

    int out_shape[] = {1, config.n_ctx, config.n_embd * 4};  // saving space for ffn
    buffers->out = tensor_alloc(3, out_shape);

    return buffers->res != NULL && buffers->out != NULL ? 0 : -1;
}

int main() {
    printf("GPT-2 inference\n");

    if (gpt2_initialize(&model, &config) != 0) {
        fprintf(stderr, "Failed to initialize GPT-2 model\n");
        return -1;
    }

    FILE *file = fopen("../models/gpt2-124M-weights.bin", "rb");
    if (file == NULL) {
        fprintf(stderr, "Failed to open weights file: %s\n", "../models/gpt2-124M-weights.bin");
        return -1;
    }

    // Load weights
    if (gpt2_load_weights(&model, file) != 0) {
        fprintf(stderr, "Failed to load GPT-2 weights\n");

        gpt2_free(&model);
        fclose(file);
        return -1;
    }

    printf("Model loaded successfully.\n");

    // sample tokens: "The capital of France is"
    int input_tokens[] = {464, 3139, 286, 4881, 318};  
    int seq_len = 5;
    
    int *d_input_tokens;
    if (prepare_input_tokens(input_tokens, seq_len, &d_input_tokens) != 0) {
        gpt2_free(&model);
        fclose(file);
        return -1;
    }

    // setup inference buffers on GPU
    inference_buffers_t buffers;
    if (setup_inference_buffers(&buffers) != 0) {
        fprintf(stderr, "Failed to setup inference buffers\n");
        gpt2_free(&model);
        fclose(file);
        return -1;
    }

    // embedding layer
    embedding_forward<<<1, 256>>>(buffers.res->data, d_input_tokens, model.emb.wte->data, model.emb.wpe->data, seq_len, config.n_embd, config.n_positions);
    cudaDeviceSynchronize();

    // print res tensor first row for verification
    // float *h_res = (float *) malloc(sizeof(float) * seq_len * config.n_embd);
    // cudaMemcpy(h_res, buffers.res->data, sizeof(float) * seq_len * config.n_embd, cudaMemcpyDeviceToHost);
    // printf("Embedding output (second token):\n");
    // for (int i = 0; i < config.n_embd; i++) {
    //     printf("%f ", h_res[config.n_embd + i]);
    // }
    // printf("\n");
    // free(h_res);

    // layers
    for (int layer_idx = 0; layer_idx < config.n_layer; layer_idx++) {
        block_t *block = &model.h[layer_idx];
        // self-attention
        // (to be implemented)
        // feed-forward
        // (to be implemented)

        layernorm_forward<<<1, 256>>>(buffers.out->data, buffers.res->data, block->ln_1.w->data, block->ln_1.b->data, seq_len, config.n_embd);
        cudaDeviceSynchronize();

        // print embedding of second token after layernorm for verification
        // float *h_out = (float *) malloc(sizeof(float) * seq_len * config.n_embd);
        // cudaMemcpy(h_out, buffers.out->data, sizeof(float) * seq_len * config.n_embd, cudaMemcpyDeviceToHost);
        // printf("After Layer %d LayerNorm (second token):\n", layer_idx + 1);
        // for (int i = 0; i < config.n_embd; i++) {
        //     printf("%f ", h_out[config.n_embd + i]);
        // }
        // printf("\n");
        // free(h_out);

        qkv_projection<<<CEIL_DIV(seq_len * 3 * config.n_embd, 256), 256>>>(buffers.res->data, buffers.out->data, block->attn.qkv_w->data, block->attn.qkv_b->data, 1, seq_len, config.n_embd);
        cudaDeviceSynchronize();

        // print q, k, v of second token for verification
        // float *h_qkv = (float *) malloc(sizeof(float) * seq_len * 3 * config.n_embd);
        // cudaMemcpy(h_qkv, buffers.res->data, sizeof(float) * seq_len * 3 * config.n_embd, cudaMemcpyDeviceToHost);
        // printf("After Layer %d QKV Projection (second token):\n", layer_idx + 1);
        
        // // Token 1 (second token) starts at offset: 1 * 3 * n_embd
        // int token_idx = 1;
        // int token_offset = token_idx * 3 * config.n_embd;
        
        // printf("Q=\n");
        // for (int i = 0; i < config.n_embd; i++) {
        //     printf("%f ", h_qkv[token_offset + i]);
        // }
        // printf("\nK=\n");
        // for (int i = 0; i < config.n_embd; i++) {
        //     printf("%f ", h_qkv[token_offset + config.n_embd + i]);
        // }
        // printf("\nV=\n");
        // for (int i = 0; i < config.n_embd; i++) {
        //     printf("%f ", h_qkv[token_offset + 2 * config.n_embd + i]);
        // }
        // printf("\n");
        // free(h_qkv);

        break;
    }

    cudaFree(d_input_tokens);

    tensor_free(buffers.res);
    tensor_free(buffers.out);

    gpt2_free(&model);
    fclose(file);
    return 0;
}
