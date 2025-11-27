/* GPT-2 model implementation - C implementation */

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "gpt2/gpt2.h"
#include "gpt2/tensor.h"

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
    model->emb.wte = (tensor_t *)malloc(sizeof(tensor_t));
    if (model->emb.wte == NULL) return -1;
    model->emb.wte->ndim = 2;
    model->emb.wte->shape = (int *)malloc(2 * sizeof(int));
    if (model->emb.wte->shape == NULL) return -1;
    model->emb.wte->shape[0] = shape_wte[0];
    model->emb.wte->shape[1] = shape_wte[1];
    model->emb.wte->data = ptr;
    ptr += config->n_embd * config->vocab_size;
    
    int shape_wpe[] = {config->n_positions, config->n_embd};
    model->emb.wpe = (tensor_t *)malloc(sizeof(tensor_t));
    if (model->emb.wpe == NULL) return -1;
    model->emb.wpe->ndim = 2;
    model->emb.wpe->shape = (int *)malloc(2 * sizeof(int));
    if (model->emb.wpe->shape == NULL) return -1;
    model->emb.wpe->shape[0] = shape_wpe[0];
    model->emb.wpe->shape[1] = shape_wpe[1];
    model->emb.wpe->data = ptr;
    ptr += config->n_positions * config->n_embd;

    // layers
    for (int i = 0; i < config->n_layer; i++) {
        block_t *block = &model->h[i];

        // ln_1.w
        int shape_ln_w[] = {config->n_embd};
        block->ln_1.w = (tensor_t *)malloc(sizeof(tensor_t));
        if (block->ln_1.w == NULL) return -1;
        block->ln_1.w->ndim = 1;
        block->ln_1.w->shape = (int *)malloc(1 * sizeof(int));
        if (block->ln_1.w->shape == NULL) return -1;
        block->ln_1.w->shape[0] = shape_ln_w[0];
        block->ln_1.w->data = ptr;
        ptr += config->n_embd;
        
        // ln_1.b
        block->ln_1.b = (tensor_t *)malloc(sizeof(tensor_t));
        if (block->ln_1.b == NULL) return -1;
        block->ln_1.b->ndim = 1;
        block->ln_1.b->shape = (int *)malloc(1 * sizeof(int));
        if (block->ln_1.b->shape == NULL) return -1;
        block->ln_1.b->shape[0] = shape_ln_w[0];
        block->ln_1.b->data = ptr;
        ptr += config->n_embd;

        // attn.qkv_w
        int shape_qkv_w[] = {config->n_embd, 3 * config->n_embd};
        block->attn.qkv_w = (tensor_t *)malloc(sizeof(tensor_t));
        if (block->attn.qkv_w == NULL) return -1;
        block->attn.qkv_w->ndim = 2;
        block->attn.qkv_w->shape = (int *)malloc(2 * sizeof(int));
        if (block->attn.qkv_w->shape == NULL) return -1;
        block->attn.qkv_w->shape[0] = shape_qkv_w[0];
        block->attn.qkv_w->shape[1] = shape_qkv_w[1];
        block->attn.qkv_w->data = ptr;
        ptr += config->n_embd * 3 * config->n_embd;
        
        // attn.qkv_b
        int shape_qkv_b[] = {3 * config->n_embd};
        block->attn.qkv_b = (tensor_t *)malloc(sizeof(tensor_t));
        if (block->attn.qkv_b == NULL) return -1;
        block->attn.qkv_b->ndim = 1;
        block->attn.qkv_b->shape = (int *)malloc(1 * sizeof(int));
        if (block->attn.qkv_b->shape == NULL) return -1;
        block->attn.qkv_b->shape[0] = shape_qkv_b[0];
        block->attn.qkv_b->data = ptr;
        ptr += 3 * config->n_embd;
        
        // attn.proj_w
        int shape_proj_w[] = {config->n_embd, config->n_embd};
        block->attn.proj_w = (tensor_t *)malloc(sizeof(tensor_t));
        if (block->attn.proj_w == NULL) return -1;
        block->attn.proj_w->ndim = 2;
        block->attn.proj_w->shape = (int *)malloc(2 * sizeof(int));
        if (block->attn.proj_w->shape == NULL) return -1;
        block->attn.proj_w->shape[0] = shape_proj_w[0];
        block->attn.proj_w->shape[1] = shape_proj_w[1];
        block->attn.proj_w->data = ptr;
        ptr += config->n_embd * config->n_embd;
        
        // attn.proj_b
        int shape_proj_b[] = {config->n_embd};
        block->attn.proj_b = (tensor_t *)malloc(sizeof(tensor_t));
        if (block->attn.proj_b == NULL) return -1;
        block->attn.proj_b->ndim = 1;
        block->attn.proj_b->shape = (int *)malloc(1 * sizeof(int));
        if (block->attn.proj_b->shape == NULL) return -1;
        block->attn.proj_b->shape[0] = shape_proj_b[0];
        block->attn.proj_b->data = ptr;
        ptr += config->n_embd;

        // ln_2.w
        block->ln_2.w = (tensor_t *)malloc(sizeof(tensor_t));
        if (block->ln_2.w == NULL) return -1;
        block->ln_2.w->ndim = 1;
        block->ln_2.w->shape = (int *)malloc(1 * sizeof(int));
        if (block->ln_2.w->shape == NULL) return -1;
        block->ln_2.w->shape[0] = shape_ln_w[0];
        block->ln_2.w->data = ptr;
        ptr += config->n_embd;
        
        // ln_2.b
        block->ln_2.b = (tensor_t *)malloc(sizeof(tensor_t));
        if (block->ln_2.b == NULL) return -1;
        block->ln_2.b->ndim = 1;
        block->ln_2.b->shape = (int *)malloc(1 * sizeof(int));
        if (block->ln_2.b->shape == NULL) return -1;
        block->ln_2.b->shape[0] = shape_ln_w[0];
        block->ln_2.b->data = ptr;
        ptr += config->n_embd;

        // mlp.fc_w
        int shape_fc_w[] = {config->n_embd, 4 * config->n_embd};
        block->mlp.fc_w = (tensor_t *)malloc(sizeof(tensor_t));
        if (block->mlp.fc_w == NULL) return -1;
        block->mlp.fc_w->ndim = 2;
        block->mlp.fc_w->shape = (int *)malloc(2 * sizeof(int));
        if (block->mlp.fc_w->shape == NULL) return -1;
        block->mlp.fc_w->shape[0] = shape_fc_w[0];
        block->mlp.fc_w->shape[1] = shape_fc_w[1];
        block->mlp.fc_w->data = ptr;
        ptr += config->n_embd * 4 * config->n_embd;
        
        // mlp.fc_b
        int shape_fc_b[] = {4 * config->n_embd};
        block->mlp.fc_b = (tensor_t *)malloc(sizeof(tensor_t));
        if (block->mlp.fc_b == NULL) return -1;
        block->mlp.fc_b->ndim = 1;
        block->mlp.fc_b->shape = (int *)malloc(1 * sizeof(int));
        if (block->mlp.fc_b->shape == NULL) return -1;
        block->mlp.fc_b->shape[0] = shape_fc_b[0];
        block->mlp.fc_b->data = ptr;
        ptr += 4 * config->n_embd;
        
        // mlp.proj_w
        int shape_mlp_proj_w[] = {4 * config->n_embd, config->n_embd};
        block->mlp.proj_w = (tensor_t *)malloc(sizeof(tensor_t));
        if (block->mlp.proj_w == NULL) return -1;
        block->mlp.proj_w->ndim = 2;
        block->mlp.proj_w->shape = (int *)malloc(2 * sizeof(int));
        if (block->mlp.proj_w->shape == NULL) return -1;
        block->mlp.proj_w->shape[0] = shape_mlp_proj_w[0];
        block->mlp.proj_w->shape[1] = shape_mlp_proj_w[1];
        block->mlp.proj_w->data = ptr;
        ptr += 4 * config->n_embd * config->n_embd;
        
        // mlp.proj_b
        int shape_mlp_proj_b[] = {config->n_embd};
        block->mlp.proj_b = (tensor_t *)malloc(sizeof(tensor_t));
        if (block->mlp.proj_b == NULL) return -1;
        block->mlp.proj_b->ndim = 1;
        block->mlp.proj_b->shape = (int *)malloc(1 * sizeof(int));
        if (block->mlp.proj_b->shape == NULL) return -1;
        block->mlp.proj_b->shape[0] = shape_mlp_proj_b[0];
        block->mlp.proj_b->data = ptr;
        ptr += config->n_embd;
    }

    // final layer norm
    int shape_ln_f_w[] = {config->n_embd};
    model->ln_f.w = (tensor_t *)malloc(sizeof(tensor_t));
    if (model->ln_f.w == NULL) return -1;
    model->ln_f.w->ndim = 1;
    model->ln_f.w->shape = (int *)malloc(1 * sizeof(int));
    if (model->ln_f.w->shape == NULL) return -1;
    model->ln_f.w->shape[0] = shape_ln_f_w[0];
    model->ln_f.w->data = ptr;
    ptr += config->n_embd;
    
    model->ln_f.b = (tensor_t *)malloc(sizeof(tensor_t));
    if (model->ln_f.b == NULL) return -1;
    model->ln_f.b->ndim = 1;
    model->ln_f.b->shape = (int *)malloc(1 * sizeof(int));
    if (model->ln_f.b->shape == NULL) return -1;
    model->ln_f.b->shape[0] = shape_ln_f_w[0];
    model->ln_f.b->data = ptr;
    ptr += config->n_embd;

    // Store total number of parameters
    model->num_parameters = total_params;

    return 0;
}

int gpt2_load_weights(gpt2_t *model, FILE *file) {
    // emb
    if (tensor_load(model->emb.wte, file) != 0) return -1;
    if (tensor_load(model->emb.wpe, file) != 0) return -1;

    // layers
    for (int i = 0; i < NUM_LAYERS; i++) {
        block_t *block = &model->h[i];

        // ln_1
        if (tensor_load(block->ln_1.w, file) != 0) return -1;
        if (tensor_load(block->ln_1.b, file) != 0) return -1;

        // attn
        if (tensor_load(block->attn.qkv_w, file) != 0) return -1;
        if (tensor_load(block->attn.qkv_b, file) != 0) return -1;
        if (tensor_load(block->attn.proj_w, file) != 0) return -1;
        if (tensor_load(block->attn.proj_b, file) != 0) return -1;

        // ln_2
        if (tensor_load(block->ln_2.w, file) != 0) return -1;
        if (tensor_load(block->ln_2.b, file) != 0) return -1;

        // mlp
        if (tensor_load(block->mlp.fc_w, file) != 0) return -1;
        if (tensor_load(block->mlp.fc_b, file) != 0) return -1;
        if (tensor_load(block->mlp.proj_w, file) != 0) return -1;
        if (tensor_load(block->mlp.proj_b, file) != 0) return -1;
    }

    // final layer norm
    if (tensor_load(model->ln_f.w, file) != 0) return -1;
    if (tensor_load(model->ln_f.b, file) != 0) return -1;

    return 0;
}

void gpt2_free(gpt2_t *model) {
    // Free the single contiguous parameter memory block
    if (model->params_memory) {
        cudaFree(model->params_memory);
    }
    
    // Free tensor structures (but not their data pointers, since they point into params_memory)
    // Free emb
    if (model->emb.wte) {
        if (model->emb.wte->shape) free(model->emb.wte->shape);
        free(model->emb.wte);
    }
    if (model->emb.wpe) {
        if (model->emb.wpe->shape) free(model->emb.wpe->shape);
        free(model->emb.wpe);
    }

    // Free layers
    for (int i = 0; i < NUM_LAYERS; i++) {
        block_t *block = &model->h[i];

        // ln_1
        if (block->ln_1.w) {
            if (block->ln_1.w->shape) free(block->ln_1.w->shape);
            free(block->ln_1.w);
        }
        if (block->ln_1.b) {
            if (block->ln_1.b->shape) free(block->ln_1.b->shape);
            free(block->ln_1.b);
        }

        // attn
        if (block->attn.qkv_w) {
            if (block->attn.qkv_w->shape) free(block->attn.qkv_w->shape);
            free(block->attn.qkv_w);
        }
        if (block->attn.qkv_b) {
            if (block->attn.qkv_b->shape) free(block->attn.qkv_b->shape);
            free(block->attn.qkv_b);
        }
        if (block->attn.proj_w) {
            if (block->attn.proj_w->shape) free(block->attn.proj_w->shape);
            free(block->attn.proj_w);
        }
        if (block->attn.proj_b) {
            if (block->attn.proj_b->shape) free(block->attn.proj_b->shape);
            free(block->attn.proj_b);
        }

        // ln_2
        if (block->ln_2.w) {
            if (block->ln_2.w->shape) free(block->ln_2.w->shape);
            free(block->ln_2.w);
        }
        if (block->ln_2.b) {
            if (block->ln_2.b->shape) free(block->ln_2.b->shape);
            free(block->ln_2.b);
        }

        // mlp
        if (block->mlp.fc_w) {
            if (block->mlp.fc_w->shape) free(block->mlp.fc_w->shape);
            free(block->mlp.fc_w);
        }
        if (block->mlp.fc_b) {
            if (block->mlp.fc_b->shape) free(block->mlp.fc_b->shape);
            free(block->mlp.fc_b);
        }
        if (block->mlp.proj_w) {
            if (block->mlp.proj_w->shape) free(block->mlp.proj_w->shape);
            free(block->mlp.proj_w);
        }
        if (block->mlp.proj_b) {
            if (block->mlp.proj_b->shape) free(block->mlp.proj_b->shape);
            free(block->mlp.proj_b);
        }
    }

    // final layer norm
    if (model->ln_f.w) {
        if (model->ln_f.w->shape) free(model->ln_f.w->shape);
        free(model->ln_f.w);
    }
    if (model->ln_f.b) {
        if (model->ln_f.b->shape) free(model->ln_f.b->shape);
        free(model->ln_f.b);
    }
}
