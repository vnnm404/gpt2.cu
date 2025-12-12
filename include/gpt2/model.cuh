#pragma once

#include <stdio.h>

#include "gpt2/tensor.cuh"

#define NUM_PARAMETER_TENSORS 16
#define NUM_ACTIVATION_TENSORS 23

#define NUM_LAYERS 12
#define NUM_HEADS 12
#define EMBEDDING_SIZE 768
#define CONTEXT_SIZE 1024
#define VOCAB_SIZE 50257
#define PADDED_VOCAB_SIZE 50304

typedef struct
{
    int vocab_size;  // vocab size, e.g. 50257
    int batch_size;  // batch size, e.g. 1
    int n_layer;     // number of layers, e.g. 12
    int n_head;      // number of heads in attention, e.g. 12
    int n_embd;      // number of channels, e.g. 768
    int n_positions; // max sequence length, e.g. 1024
    int n_ctx;       // context size, e.g. 1024
} config_t;

typedef struct
{
    tensor_t wte; // (h, V)
    tensor_t wpe; // (maxT, h)
} emb_t;

typedef struct
{
    tensor_t w; // (h)
    tensor_t b; // (h)
} ln_t;

typedef struct
{
    tensor_t qkv_w;  // (h, 3*h)
    tensor_t qkv_b;  // (3*h)
    tensor_t proj_w; // (h, h)
    tensor_t proj_b; // (h)
} attn_t;

typedef struct
{
    tensor_t fc_w;   // (h, 4*h)
    tensor_t fc_b;   // (4*h)
    tensor_t proj_w; // (4*h, h)
    tensor_t proj_b; // (h)
} ffn_t;

typedef struct
{
    ln_t ln_1;
    attn_t attn;
    ln_t ln_2;
    ffn_t mlp;
} block_t;

typedef struct
{
    emb_t emb;
    block_t h[NUM_LAYERS];
    ln_t ln_f;
    float *params_memory; // base pointer for contiguous parameter memory
    size_t num_parameters; // total number of parameters
} gpt2_t;


int gpt2_initialize(gpt2_t *model, const config_t *config) {
    // Calculate total size needed for all parameters
    size_t total_params = 0;
    
    // emb: wte (h, V) + wpe (maxT, h)
    total_params += config->n_embd * config->vocab_size;
    total_params += config->n_positions * config->n_embd;
    
    // per layer: ln_1 (w, b), attn (qkv_w, qkv_b, proj_w, proj_b), ln_2 (w, b), mlp (fc_w, fc_b, proj_w, proj_b)
    for (int i = 0; i < config->n_layer; i++) {
        total_params += config->n_embd;                      // ln_1.w
        total_params += config->n_embd;                      // ln_1.b
        total_params += config->n_embd * 3 * config->n_embd; // attn.qkv_w
        total_params += 3 * config->n_embd;                  // attn.qkv_b
        total_params += config->n_embd * config->n_embd;     // attn.proj_w
        total_params += config->n_embd;                      // attn.proj_b
        total_params += config->n_embd;                      // ln_2.w
        total_params += config->n_embd;                      // ln_2.b
        total_params += config->n_embd * 4 * config->n_embd; // mlp.fc_w
        total_params += 4 * config->n_embd;                  // mlp.fc_b
        total_params += 4 * config->n_embd * config->n_embd; // mlp.proj_w
        total_params += config->n_embd;                      // mlp.proj_b
    }
    
    // final layer norm: ln_f (w, b)
    total_params += config->n_embd; // ln_f.w
    total_params += config->n_embd; // ln_f.b
    
    // Allocate single contiguous block on GPU
    cudaError_t err = cudaMalloc(&model->params_memory, total_params * sizeof(float));
    if (err != cudaSuccess) {
        return -1;
    }
    
    // Now set up tensor structures with pointers into this block
    float *ptr = model->params_memory;
    
    // emb
    int shape_wte[] = {config->n_embd, config->vocab_size};
    model->emb.wte.ndim = 2;
    model->emb.wte.shape[0] = shape_wte[0];
    model->emb.wte.shape[1] = shape_wte[1];
    model->emb.wte.shape[2] = 0;
    model->emb.wte.shape[3] = 0;
    model->emb.wte.data = ptr;
    ptr += config->n_embd * config->vocab_size;
    
    int shape_wpe[] = {config->n_positions, config->n_embd};
    model->emb.wpe.ndim = 2;
    model->emb.wpe.shape[0] = shape_wpe[0];
    model->emb.wpe.shape[1] = shape_wpe[1];
    model->emb.wpe.shape[2] = 0;
    model->emb.wpe.shape[3] = 0;
    model->emb.wpe.data = ptr;
    ptr += config->n_positions * config->n_embd;

    // layers
    for (int i = 0; i < config->n_layer; i++) {
        block_t *block = &model->h[i];

        // ln_1.w
        block->ln_1.w.ndim = 1;
        block->ln_1.w.shape[0] = config->n_embd;
        block->ln_1.w.shape[1] = 0;
        block->ln_1.w.shape[2] = 0;
        block->ln_1.w.shape[3] = 0;
        block->ln_1.w.data = ptr;
        ptr += config->n_embd;
        
        // ln_1.b
        block->ln_1.b.ndim = 1;
        block->ln_1.b.shape[0] = config->n_embd;
        block->ln_1.b.shape[1] = 0;
        block->ln_1.b.shape[2] = 0;
        block->ln_1.b.shape[3] = 0;
        block->ln_1.b.data = ptr;
        ptr += config->n_embd;

        // attn.qkv_w
        block->attn.qkv_w.ndim = 2;
        block->attn.qkv_w.shape[0] = config->n_embd;
        block->attn.qkv_w.shape[1] = 3 * config->n_embd;
        block->attn.qkv_w.shape[2] = 0;
        block->attn.qkv_w.shape[3] = 0;
        block->attn.qkv_w.data = ptr;
        ptr += config->n_embd * 3 * config->n_embd;
        
        // attn.qkv_b
        block->attn.qkv_b.ndim = 1;
        block->attn.qkv_b.shape[0] = 3 * config->n_embd;
        block->attn.qkv_b.shape[1] = 0;
        block->attn.qkv_b.shape[2] = 0;
        block->attn.qkv_b.shape[3] = 0;
        block->attn.qkv_b.data = ptr;
        ptr += 3 * config->n_embd;
        
        // attn.proj_w
        block->attn.proj_w.ndim = 2;
        block->attn.proj_w.shape[0] = config->n_embd;
        block->attn.proj_w.shape[1] = config->n_embd;
        block->attn.proj_w.shape[2] = 0;
        block->attn.proj_w.shape[3] = 0;
        block->attn.proj_w.data = ptr;
        ptr += config->n_embd * config->n_embd;
        
        // attn.proj_b
        block->attn.proj_b.ndim = 1;
        block->attn.proj_b.shape[0] = config->n_embd;
        block->attn.proj_b.shape[1] = 0;
        block->attn.proj_b.shape[2] = 0;
        block->attn.proj_b.shape[3] = 0;
        block->attn.proj_b.data = ptr;
        ptr += config->n_embd;

        // ln_2.w
        block->ln_2.w.ndim = 1;
        block->ln_2.w.shape[0] = config->n_embd;
        block->ln_2.w.shape[1] = 0;
        block->ln_2.w.shape[2] = 0;
        block->ln_2.w.shape[3] = 0;
        block->ln_2.w.data = ptr;
        ptr += config->n_embd;
        
        // ln_2.b
        block->ln_2.b.ndim = 1;
        block->ln_2.b.shape[0] = config->n_embd;
        block->ln_2.b.shape[1] = 0;
        block->ln_2.b.shape[2] = 0;
        block->ln_2.b.shape[3] = 0;
        block->ln_2.b.data = ptr;
        ptr += config->n_embd;

        // mlp.fc_w
        block->mlp.fc_w.ndim = 2;
        block->mlp.fc_w.shape[0] = config->n_embd;
        block->mlp.fc_w.shape[1] = 4 * config->n_embd;
        block->mlp.fc_w.shape[2] = 0;
        block->mlp.fc_w.shape[3] = 0;
        block->mlp.fc_w.data = ptr;
        ptr += config->n_embd * 4 * config->n_embd;
        
        // mlp.fc_b
        block->mlp.fc_b.ndim = 1;
        block->mlp.fc_b.shape[0] = 4 * config->n_embd;
        block->mlp.fc_b.shape[1] = 0;
        block->mlp.fc_b.shape[2] = 0;
        block->mlp.fc_b.shape[3] = 0;
        block->mlp.fc_b.data = ptr;
        ptr += 4 * config->n_embd;
        
        // mlp.proj_w
        block->mlp.proj_w.ndim = 2;
        block->mlp.proj_w.shape[0] = 4 * config->n_embd;
        block->mlp.proj_w.shape[1] = config->n_embd;
        block->mlp.proj_w.shape[2] = 0;
        block->mlp.proj_w.shape[3] = 0;
        block->mlp.proj_w.data = ptr;
        ptr += 4 * config->n_embd * config->n_embd;
        
        // mlp.proj_b
        block->mlp.proj_b.ndim = 1;
        block->mlp.proj_b.shape[0] = config->n_embd;
        block->mlp.proj_b.shape[1] = 0;
        block->mlp.proj_b.shape[2] = 0;
        block->mlp.proj_b.shape[3] = 0;
        block->mlp.proj_b.data = ptr;
        ptr += config->n_embd;
    }

    // final layer norm
    model->ln_f.w.ndim = 1;
    model->ln_f.w.shape[0] = config->n_embd;
    model->ln_f.w.shape[1] = 0;
    model->ln_f.w.shape[2] = 0;
    model->ln_f.w.shape[3] = 0;
    model->ln_f.w.data = ptr;
    ptr += config->n_embd;
    
    model->ln_f.b.ndim = 1;
    model->ln_f.b.shape[0] = config->n_embd;
    model->ln_f.b.shape[1] = 0;
    model->ln_f.b.shape[2] = 0;
    model->ln_f.b.shape[3] = 0;
    model->ln_f.b.data = ptr;
    ptr += config->n_embd;

    // Store total number of parameters
    model->num_parameters = total_params;

    return 0;
}

int gpt2_load_weights(gpt2_t *model, FILE *file) {
    // emb
    if (tensor_load(&model->emb.wte, file) != 0) return -1;
    if (tensor_load(&model->emb.wpe, file) != 0) return -1;

    // layers
    for (int i = 0; i < NUM_LAYERS; i++) {
        block_t *block = &model->h[i];

        // ln_1
        if (tensor_load(&block->ln_1.w, file) != 0) return -1;
        if (tensor_load(&block->ln_1.b, file) != 0) return -1;

        // attn
        if (tensor_load(&block->attn.qkv_w, file) != 0) return -1;
        if (tensor_load(&block->attn.qkv_b, file) != 0) return -1;
        if (tensor_load(&block->attn.proj_w, file) != 0) return -1;
        if (tensor_load(&block->attn.proj_b, file) != 0) return -1;

        // ln_2
        if (tensor_load(&block->ln_2.w, file) != 0) return -1;
        if (tensor_load(&block->ln_2.b, file) != 0) return -1;

        // mlp
        if (tensor_load(&block->mlp.fc_w, file) != 0) return -1;
        if (tensor_load(&block->mlp.fc_b, file) != 0) return -1;
        if (tensor_load(&block->mlp.proj_w, file) != 0) return -1;
        if (tensor_load(&block->mlp.proj_b, file) != 0) return -1;
    }

    // final layer norm
    if (tensor_load(&model->ln_f.w, file) != 0) return -1;
    if (tensor_load(&model->ln_f.b, file) != 0) return -1;

    return 0;
}

void gpt2_free(gpt2_t *model) {
    if (model->params_memory) {
        cudaFree(model->params_memory);
        model->params_memory = NULL;
    }
}
