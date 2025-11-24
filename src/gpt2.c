/* GPT-2 model implementation - C implementation */

#include <stdlib.h>
#include <stdio.h>

#include "gpt2/gpt2.h"
#include "gpt2/tensor.h"

int gpt2_initialize(gpt2_t *model, const config_t *config) {
    // emb
    int shape_wte[] = {config->padded_vocab_size, config->n_embd};
    model->emb.wte = tensor_alloc(2, shape_wte);
    if (model->emb.wte == NULL) return -1;
    int shape_wpe[] = {config->n_positions, config->n_embd};
    model->emb.wpe = tensor_alloc(2, shape_wpe);
    if (model->emb.wpe == NULL) return -1;

    // layers
    for (int i = 0; i < config->n_layer; i++) {
        block_t *block = &model->h[i];

        // ln_1
        int shape_ln_w[] = {config->n_embd};
        block->ln_1.w = tensor_alloc(1, shape_ln_w);
        if (block->ln_1.w == NULL) return -1;
        block->ln_1.b = tensor_alloc(1, shape_ln_w);
        if (block->ln_1.b == NULL) return -1;

        // attn
        int shape_qkv_w[] = {3 * config->n_embd, config->n_embd};
        block->attn.qkv_w = tensor_alloc(2, shape_qkv_w);
        if (block->attn.qkv_w == NULL) return -1;
        int shape_qkv_b[] = {3 * config->n_embd};
        block->attn.qkv_b = tensor_alloc(1, shape_qkv_b);
        if (block->attn.qkv_b == NULL) return -1;
        int shape_proj_w[] = {config->n_embd, config->n_embd};
        block->attn.proj_w = tensor_alloc(2, shape_proj_w);
        if (block->attn.proj_w == NULL) return -1;
        int shape_proj_b[] = {config->n_embd};
        block->attn.proj_b = tensor_alloc(1, shape_proj_b);
        if (block->attn.proj_b == NULL) return -1;

        // ln_2
        block->ln_2.w = tensor_alloc(1, shape_ln_w);
        if (block->ln_2.w == NULL) return -1;
        block->ln_2.b = tensor_alloc(1, shape_ln_w);
        if (block->ln_2.b == NULL) return -1;

        // mlp
        int shape_fc_w[] = {4 * config->n_embd, config->n_embd};
        block->mlp.fc_w = tensor_alloc(2, shape_fc_w);
        if (block->mlp.fc_w == NULL) return -1;
        int shape_fc_b[] = {4 * config->n_embd};
        block->mlp.fc_b = tensor_alloc(1, shape_fc_b);
        if (block->mlp.fc_b == NULL) return -1;
        int shape_mlp_proj_w[] = {config->n_embd, 4 * config->n_embd};
        block->mlp.proj_w = tensor_alloc(2, shape_mlp_proj_w);
        if (block->mlp.proj_w == NULL) return -1;
        int shape_mlp_proj_b[] = {config->n_embd};
        block->mlp.proj_b = tensor_alloc(1, shape_mlp_proj_b);
        if (block->mlp.proj_b == NULL) return -1;
    }

    // final layer norm
    int shape_ln_f_w[] = {config->n_embd};
    model->ln_f.w = tensor_alloc(1, shape_ln_f_w);
    if (model->ln_f.w == NULL) return -1;
    model->ln_f.b = tensor_alloc(1, shape_ln_f_w);
    if (model->ln_f.b == NULL) return -1;

    return 0;
}

void gpt2_free(gpt2_t *model) {
    // Free emb
    tensor_free(model->emb.wte);
    tensor_free(model->emb.wpe);

    // Free layers
    for (int i = 0; i < NUM_LAYERS; i++) {
        block_t *block = &model->h[i];

        // ln_1
        tensor_free(block->ln_1.w);
        tensor_free(block->ln_1.b);

        // attn
        tensor_free(block->attn.qkv_w);
        tensor_free(block->attn.qkv_b);
        tensor_free(block->attn.proj_w);
        tensor_free(block->attn.proj_b);

        // ln_2
        tensor_free(block->ln_2.w);
        tensor_free(block->ln_2.b);

        // mlp
        tensor_free(block->mlp.fc_w);
        tensor_free(block->mlp.fc_b);
        tensor_free(block->mlp.proj_w);
        tensor_free(block->mlp.proj_b);
    }

    // final layer norm
    tensor_free(model->ln_f.w);
    tensor_free(model->ln_f.b);
}
