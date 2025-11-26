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

// Include implementations for device function access in megakernel
#include "../src/gpt2.cu"
#include "../src/tensor.cu"
#include "../src/layers/embedding.cu"
#include "../src/layers/layernorm.cu"
#include "../src/layers/mlp.cu"
#include "../src/layers/attention.cu"
#include "../src/layers/residual.cu"
#include "../src/layers/gelu.cu"

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

__device__ int tile = 128;

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

void flatten_gpt2(flat_gpt2_t *out, gpt2_t *m);
__global__ void forward_mk(config_t config, flat_gpt2_t model, inference_buffers_t buffers, const int *d_input_tokens, int seq_len, float *buf1, // [batch_size, seq_len, n_embd * 4]
    float *buf2, // [batch_size, seq_len, n_embd * 4]
    float *res,  // [batch_size, seq_len, n_embd]
    float *logits, // [batch_size, seq_len, vocab_size]

    float *preatt,  // [batch_size, n_head, seq_len, seq_len]
    float *att  // [batch_size, n_head, seq_len, seq_len]
    );




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
    int input_tokens[] = {464, 3139, 286, 4881, 318, 262, 3139, 286};  
    int seq_len = 8;
    
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


    // Get device props
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // device 0
    // printf("%d \n", prop.thread)
    // Start timing for forward pass
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    flat_gpt2_t flat_model;
    flatten_gpt2(&flat_model, &model);

    // Forward pass
    // for(int j = 0; j < 100; j++){
    // forward(d_input_tokens, seq_len);
    
    forward_mk<<<64, 128>>>(config, flat_model, buffers, d_input_tokens, seq_len, 
        buffers.buf1->data, buffers.buf2->data, buffers.res->data, buffers.logits->data, buffers.preatt->data, buffers.att->data    );
    // }
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


#define MAX_SYNC 5000 

__device__ int g_block_counter[MAX_SYNC];  // must be zero-initialized

__device__ void syncblocks(int total_blocks, int stage) {
    // Stage corresponds to g_block_counter[stage]

    // __syncthreads();

    // One thread per block increments stage counter
    // if (threadIdx.x == 0) {
    //     atomicAdd(&g_block_counter[stage], 1);
    //     while (atomicAdd(&g_block_counter[stage], 0) < total_blocks) {
    //         __threadfence();  // ensure visibility across SMs
    //     }
    // }  

    // __syncthreads();
}

static inline float* T(tensor_t* t) {
    return t->data;
}


void flatten_gpt2(flat_gpt2_t *out, gpt2_t *m) {
    // Embeddings
    out->wte = T(m->emb.wte);
    out->wpe = T(m->emb.wpe);

    // Blocks
    for (int i = 0; i < NUM_LAYERS; i++) {
        block_t *b = &m->h[i];

        // ln_1
        out->ln_1_w[i] = T(b->ln_1.w);
        out->ln_1_b[i] = T(b->ln_1.b);

        // attention
        out->qkv_w[i]  = T(b->attn.qkv_w);
        out->qkv_b[i]  = T(b->attn.qkv_b);
        out->proj_w[i] = T(b->attn.proj_w);
        out->proj_b[i] = T(b->attn.proj_b);

        // ln_2
        out->ln_2_w[i] = T(b->ln_2.w);
        out->ln_2_b[i] = T(b->ln_2.b);

        // MLP
        out->fc_w[i]       = T(b->mlp.fc_w);
        out->fc_b[i]       = T(b->mlp.fc_b);
        out->fc_proj_w[i]  = T(b->mlp.proj_w);
        out->fc_proj_b[i]  = T(b->mlp.proj_b);
    }

    // final layernorm
    out->ln_f_w = T(m->ln_f.w);
    out->ln_f_b = T(m->ln_f.b);
}


__global__
void forward_mk(config_t config, flat_gpt2_t flat, const inference_buffers_t buffers,
                const int *d_input_tokens, int seq_len,
                float *buf1,
                float *buf2,
                float *res,
                float *logits,
                float *preatt,
                float *att
) {
    for(int j = 0; j < 1000; j++){
    int phase = 0;
    int num_blocks = gridDim.x;

    // --- embedding ---
    for (int i = blockIdx.x; i < 1; i += num_blocks) {
        embedding_forward_mk(
            res,
            d_input_tokens,
            flat.wte,
            flat.wpe,
            seq_len,
            config.n_embd,
            config.vocab_size,
            config.n_positions,
            i
        );
    }

    syncblocks(gridDim.x, phase++);

    // --- transformer blocks ---
    for (int layer_idx = 0; layer_idx < config.n_layer; layer_idx++) {

        // ln1
        for (int i = blockIdx.x; i < 1; i += num_blocks)
            layernorm_forward_mk(buf1, res,
                                 flat.ln_1_w[layer_idx],
                                 flat.ln_1_b[layer_idx],
                                 seq_len, config.n_embd, i);

        syncblocks(gridDim.x, phase++);

        // qkv matmul
        for (int i = blockIdx.x; i < CEIL_DIV(seq_len * 3 * config.n_embd, tile); i += num_blocks)
            mlp_forward_mk(buf2, buf1,
                           flat.qkv_w[layer_idx],
                           flat.qkv_b[layer_idx],
                           1, seq_len, config.n_embd, config.n_embd * 3, i);

        syncblocks(gridDim.x, phase++);

        // attention softmax
        for (int i = blockIdx.x; i < CEIL_DIV(seq_len * config.n_head, tile); i += num_blocks)
            attention_forward_mk(buf1, preatt, att, buf2,
                                 1, seq_len, config.n_head, config.n_embd, i);

        syncblocks(gridDim.x, phase++);

        // attention proj
        for (int i = blockIdx.x; i < CEIL_DIV(seq_len * config.n_embd, tile); i += num_blocks)
            mlp_forward_mk(buf2, buf1,
                           flat.proj_w[layer_idx],
                           flat.proj_b[layer_idx],
                           1, seq_len, config.n_embd, config.n_embd, i);

        syncblocks(gridDim.x, phase++);

        // residual 1
        for (int i = blockIdx.x; i < CEIL_DIV(seq_len * config.n_embd, tile); i += num_blocks)
            residual_forward_mk(res, buf2, res,
                                1, seq_len, config.n_embd, i);

        syncblocks(gridDim.x, phase++);

        // ln2
        for (int i = blockIdx.x; i < 1; i += num_blocks)
            layernorm_forward_mk(buf1, res,
                                 flat.ln_2_w[layer_idx],
                                 flat.ln_2_b[layer_idx],
                                 seq_len, config.n_embd, i);

        syncblocks(gridDim.x, phase++);

        // mlp fc
        for (int i = blockIdx.x; i < CEIL_DIV(seq_len * 4 * config.n_embd, tile); i += num_blocks)
            mlp_forward_mk(buf2, buf1,
                           flat.fc_w[layer_idx],
                           flat.fc_b[layer_idx],
                           1, seq_len, config.n_embd, config.n_embd * 4, i);

        syncblocks(gridDim.x, phase++);

        // gelu
        for (int i = blockIdx.x; i < CEIL_DIV(seq_len * config.n_embd * 4, tile); i += num_blocks)
            gelu_forward_mk(buf1, buf2,
                            1, seq_len, config.n_embd * 4, i);

        syncblocks(gridDim.x, phase++);

        // mlp proj
        for (int i = blockIdx.x; i < CEIL_DIV(seq_len * config.n_embd, tile); i += num_blocks)
            mlp_forward_mk(buf2, buf1,
                           flat.fc_proj_w[layer_idx],
                           flat.fc_proj_b[layer_idx],
                           1, seq_len, config.n_embd * 4, config.n_embd, i);

        syncblocks(gridDim.x, phase++);

        // residual 2
        for (int i = blockIdx.x; i < CEIL_DIV(seq_len * config.n_embd, tile); i += num_blocks)
            residual_forward_mk(res, buf2, res,
                                1, seq_len, config.n_embd, i);

        syncblocks(gridDim.x, phase++);
    }

    // final ln_f
    for (int i = blockIdx.x; i < 1; i += num_blocks)
        layernorm_forward_mk(res, res,
                             flat.ln_f_w,
                             flat.ln_f_b,
                             seq_len, config.n_embd, i);

    syncblocks(gridDim.x, phase++);

    // logits: matmul with wte
    for (int i = blockIdx.x; i < CEIL_DIV(seq_len * config.vocab_size, tile); i += num_blocks) {
        mlp_forward_mk(logits, res,
                       flat.wte,
                       NULL,
                       1, seq_len, config.n_embd, config.vocab_size, i);
    }
        syncblocks(gridDim.x, phase++);

}
}



void forward(const int *d_input_tokens, int seq_len) {
    embedding_forward<<<1, 256>>>(buffers.res->data, d_input_tokens, model.emb.wte->data, model.emb.wpe->data, seq_len, config.n_embd, config.vocab_size, config.n_positions);

    // layers
    for (int layer_idx = 0; layer_idx < config.n_layer; layer_idx++) {
        block_t *block = &model.h[layer_idx];

        layernorm_forward<<<1, 256>>>(buffers.buf1->data, buffers.res->data, block->ln_1.w->data, block->ln_1.b->data, seq_len, config.n_embd);

        mlp_forward<<<CEIL_DIV(seq_len * 3 * config.n_embd, 256), 256>>>(buffers.buf2->data, buffers.buf1->data, block->attn.qkv_w->data, block->attn.qkv_b->data, 1, seq_len, config.n_embd, config.n_embd * 3);

        attention_forward<<<CEIL_DIV(1 * seq_len * config.n_head, 256), 256>>>(buffers.buf1->data, buffers.preatt->data, buffers.att->data, buffers.buf2->data, 1, seq_len, config.n_head, config.n_embd);

        mlp_forward<<<CEIL_DIV(1 * seq_len * config.n_embd, 256), 256>>>(buffers.buf2->data, buffers.buf1->data, block->attn.proj_w->data, block->attn.proj_b->data, 1, seq_len, config.n_embd, config.n_embd);

        residual_forward<<<CEIL_DIV(1 * seq_len * config.n_embd, 256), 256>>>(buffers.res->data, buffers.buf2->data, buffers.res->data, 1, seq_len, config.n_embd);

        layernorm_forward<<<1, 256>>>(buffers.buf1->data, buffers.res->data, block->ln_2.w->data, block->ln_2.b->data, seq_len, config.n_embd);

        mlp_forward<<<CEIL_DIV(seq_len * 4 * config.n_embd, 256), 256>>>(buffers.buf2->data, buffers.buf1->data, block->mlp.fc_w->data, block->mlp.fc_b->data, 1, seq_len, config.n_embd, config.n_embd * 4);

        gelu_forward<<<CEIL_DIV(1 * seq_len * config.n_embd * 4, 256), 256>>>(buffers.buf1->data, buffers.buf2->data, 1, seq_len, config.n_embd * 4);

        mlp_forward<<<CEIL_DIV(1 * seq_len * config.n_embd, 256), 256>>>(buffers.buf2->data, buffers.buf1->data, block->mlp.proj_w->data, block->mlp.proj_b->data, 1, seq_len, config.n_embd * 4, config.n_embd);

        residual_forward<<<CEIL_DIV(1 * seq_len * config.n_embd, 256), 256>>>(buffers.res->data, buffers.buf2->data, buffers.res->data, 1, seq_len, config.n_embd);
    }

    layernorm_forward<<<1, 256>>>(buffers.res->data, buffers.res->data, model.ln_f.w->data, model.ln_f.b->data, seq_len, config.n_embd);

    mlp_forward<<<CEIL_DIV(1 * seq_len * config.vocab_size, 256), 256>>>(buffers.logits->data, buffers.res->data, model.emb.wte->data, NULL, 1, seq_len, config.n_embd, config.vocab_size);

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