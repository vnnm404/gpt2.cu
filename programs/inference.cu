/* GPT-2 inference executable - C implementation */

#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>

#include "gpt2/gpt2.h"
#include "gpt2/layers/embedding.h"
#include "gpt2/layers/layernorm.h"
#include "gpt2/layers/mlp.h"
#include "gpt2/layers/attention.h"
#include "gpt2/layers/residual.h"
#include "gpt2/layers/gelu.h"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

config_t config = {
    .vocab_size = 50257,
    .n_layer = 12,
    .n_head = 12,
    .n_embd = 768,
    .n_positions = 1024,
    .n_ctx = 1024
};

gpt2_t model;

typedef struct {
    tensor_t *buf1; // [batch_size, seq_len, n_embd * 4]
    tensor_t *buf2; // [batch_size, seq_len, n_embd * 4]
    tensor_t *res;  // [batch_size, seq_len, n_embd]
    tensor_t *logits; // [batch_size, seq_len, vocab_size]

    tensor_t *preatt;  // [batch_size, n_head, seq_len, seq_len]
    tensor_t *att;  // [batch_size, n_head, seq_len, seq_len]
} inference_buffers_t;

inference_buffers_t buffers;

int prepare_input_tokens(int *input_tokens, int seq_len, int **d_input_tokens);
void free_inference_buffers(inference_buffers_t *buffers);
int setup_inference_buffers(inference_buffers_t *buffers);
void forward(const int *d_input_tokens, int seq_len);
int extract_greedy_token(int seq_len, tensor_t *logits);

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
    fclose(file);

    printf("Model loaded successfully.\n");

    // sample tokens: "The capital of France is"
    // int input_tokens[] = {464, 3139, 286, 4881, 318, 262, 3139, 286};  
    int input_tokens[] = {14350, 1747, 1244, 1011, 674};  // "Robots might take our"
    int seq_len = 5;
    
    int *d_input_tokens;
    if (prepare_input_tokens(input_tokens, seq_len, &d_input_tokens) != 0) {
        gpt2_free(&model);
        fclose(file);
        return -1;
    }

    // setup inference buffers on GPU
    if (setup_inference_buffers(&buffers) != 0) {
        fprintf(stderr, "Failed to setup inference buffers\n");
        gpt2_free(&model);
        fclose(file);
        return -1;
    }

    // Start timing for forward pass
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Forward pass
    forward(d_input_tokens, seq_len);
    cudaDeviceSynchronize();

    // Stop timing and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("fwd pass time: %.3f ms\n", milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Extract predicted token (greedy)
    int predicted_token = extract_greedy_token(seq_len, buffers.logits);
    printf("Predicted next token ID: %d\n", predicted_token);

    cudaFree(d_input_tokens);
    free_inference_buffers(&buffers);
    gpt2_free(&model);
    return 0;
}

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

void free_inference_buffers(inference_buffers_t *buffers) {
    tensor_free(buffers->buf1);
    tensor_free(buffers->buf2);
    tensor_free(buffers->res);
    tensor_free(buffers->preatt);
    tensor_free(buffers->att);
}

int setup_inference_buffers(inference_buffers_t *buffers) {
    int buf1_shape[] = {1, config.n_ctx, config.n_embd * 4};
    buffers->buf1 = tensor_alloc(3, buf1_shape);
    if (buffers->buf1 == NULL) {
        return -1;
    }

    int buf2_shape[] = {1, config.n_ctx, config.n_embd * 4};
    buffers->buf2 = tensor_alloc(3, buf2_shape);
    if (buffers->buf2 == NULL) {
        free_inference_buffers(buffers);
        return -1;
    }

    int res_shape[] = {1, config.n_ctx, config.n_embd};
    buffers->res = tensor_alloc(3, res_shape);
    if (buffers->res == NULL) {
        free_inference_buffers(buffers);
        return -1;
    }

    int logits_shape[] = {1, config.n_ctx, config.vocab_size};
    buffers->logits = tensor_alloc(3, logits_shape);
    if (buffers->logits == NULL) {
        free_inference_buffers(buffers);
        return -1;
    }

    int preatt_shape[] = {1, config.n_head, config.n_ctx, config.n_ctx};
    buffers->preatt = tensor_alloc(4, preatt_shape);
    if (buffers->preatt == NULL) {
        free_inference_buffers(buffers);
        return -1;
    }

    int att_shape[] = {1, config.n_head, config.n_ctx, config.n_ctx};
    buffers->att = tensor_alloc(4, att_shape);
    if (buffers->att == NULL) {
        free_inference_buffers(buffers);
        return -1;
    }

    return 0;
}

void forward(const int *d_input_tokens, int seq_len) {
    embedding_forward<<<1, 256>>>(buffers.res->data, d_input_tokens, model.emb.wte->data, model.emb.wpe->data, seq_len, config.n_embd, config.vocab_size, config.n_positions);

    // layers
    for (int layer_idx = 0; layer_idx < config.n_layer; layer_idx++) {
        block_t *block = &model.h[layer_idx];

        layernorm_forward<<<1, 256>>>(buffers.buf1->data, buffers.res->data, block->ln_1.w->data, block->ln_1.b->data, NULL, NULL, seq_len, config.n_embd);

        mlp_forward<<<MLP_FORWARD_GRID(config.n_embd * 3, 1, seq_len), MLP_BLOCK_DIM>>>(buffers.buf2->data, buffers.buf1->data, block->attn.qkv_w->data, block->attn.qkv_b->data, 1, seq_len, config.n_embd, config.n_embd * 3);

        attention_forward<<<CEIL_DIV(1 * seq_len * config.n_head, 256), 256>>>(buffers.buf1->data, buffers.preatt->data, buffers.att->data, buffers.buf2->data, 1, seq_len, config.n_head, config.n_embd);

        mlp_forward<<<MLP_FORWARD_GRID(config.n_embd, 1, seq_len), MLP_BLOCK_DIM>>>(buffers.buf2->data, buffers.buf1->data, block->attn.proj_w->data, block->attn.proj_b->data, 1, seq_len, config.n_embd, config.n_embd);

        residual_forward<<<CEIL_DIV(1 * seq_len * config.n_embd, 256), 256>>>(buffers.res->data, buffers.buf2->data, buffers.res->data, 1, seq_len, config.n_embd);

        layernorm_forward<<<1, 256>>>(buffers.buf1->data, buffers.res->data, block->ln_2.w->data, block->ln_2.b->data, NULL, NULL, seq_len, config.n_embd);

        mlp_forward<<<MLP_FORWARD_GRID(config.n_embd * 4, 1, seq_len), MLP_BLOCK_DIM>>>(buffers.buf2->data, buffers.buf1->data, block->mlp.fc_w->data, block->mlp.fc_b->data, 1, seq_len, config.n_embd, config.n_embd * 4);

        gelu_forward<<<CEIL_DIV(1 * seq_len * config.n_embd * 4, 256), 256>>>(buffers.buf1->data, buffers.buf2->data, 1, seq_len, config.n_embd * 4);

        mlp_forward<<<MLP_FORWARD_GRID(config.n_embd, 1, seq_len), MLP_BLOCK_DIM>>>(buffers.buf2->data, buffers.buf1->data, block->mlp.proj_w->data, block->mlp.proj_b->data, 1, seq_len, config.n_embd * 4, config.n_embd);

        residual_forward<<<CEIL_DIV(1 * seq_len * config.n_embd, 256), 256>>>(buffers.res->data, buffers.buf2->data, buffers.res->data, 1, seq_len, config.n_embd);
    }

    layernorm_forward<<<1, 256>>>(buffers.res->data, buffers.res->data, model.ln_f.w->data, model.ln_f.b->data, NULL, NULL, seq_len, config.n_embd);

    mlp_forward<<<MLP_FORWARD_GRID(config.vocab_size, 1, seq_len), MLP_BLOCK_DIM>>>(buffers.logits->data, buffers.res->data, model.emb.wte->data, NULL, 1, seq_len, config.n_embd, config.vocab_size);

    // print first 4 
}

int extract_greedy_token(int seq_len, tensor_t *logits) {
    // logits: [1, seq_len, vocab_size]

    float *h_logits = (float *) malloc(sizeof(float) * config.vocab_size);
    cudaMemcpy(h_logits, logits->data + (seq_len - 1) * logits->shape[2], sizeof(float) * config.vocab_size, cudaMemcpyDeviceToHost);

    int max_token = 0;
    float max_logit = h_logits[0];
    for (int i = 1; i < config.vocab_size; i++) {
        if (h_logits[i] > max_logit) {
            max_logit = h_logits[i];
            max_token = i;
        }
    }
    free(h_logits);

    return max_token;
}