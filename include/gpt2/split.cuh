#pragma once

#include "gpt2/global/embedding.cuh"
#include "gpt2/global/layernorm.cuh"
#include "gpt2/global/mlp.cuh"
#include "gpt2/global/attention.cuh"
#include "gpt2/global/residual.cuh"
#include "gpt2/global/gelu.cuh"
#include "gpt2/global/softmax.cuh"
#include "gpt2/global/cross_entropy.cuh"
#include "gpt2/global/adamw.cuh"

#include "gpt2/tensor.cuh"
#include "gpt2/model.cuh"
#include "gpt2/train.cuh"
#include "gpt2/utils.cuh"


void forward(config_t config, gpt2_t model, train_buffers_t buffers, const int *d_input_tokens, int seq_len)
{
    int L = config.n_layer;
    int B = config.batch_size;
    int S = seq_len;
    int h = config.n_embd;
    int n_head = config.n_head;
    int V = config.vocab_size;

    int thr = 1024;

    embedding_forward<<<B, thr>>>(buffers.encoded.data, d_input_tokens, model.emb.wte.data, model.emb.wpe.data, S, h, V, config.n_positions);

    for (int layer_idx = 0; layer_idx < config.n_layer; layer_idx++)
    {
        block_t *block = &model.h[layer_idx];
        layer_buffers_t *layer_bufs = &buffers.blocks[layer_idx];

        tensor_t res = (layer_idx == 0) ? buffers.encoded : buffers.blocks[layer_idx - 1].res_3;

        layernorm_forward<<<B, thr>>>(layer_bufs->ln_1.data, res.data, block->ln_1.w.data, block->ln_1.b.data, layer_bufs->ln_1_mean.data, layer_bufs->ln_1_rstd.data, S, h);

        mlp_forward<<<MLP_FORWARD_GRID(h * 3, B, S), MLP_BLOCK_DIM>>>(layer_bufs->qkv.data, layer_bufs->ln_1.data, block->attn.qkv_w.data, block->attn.qkv_b.data, B, S, h, h * 3);

        attention_forward<<<CEIL_DIV(B * S * n_head, thr), thr>>>(layer_bufs->atty.data, layer_bufs->preatt.data, layer_bufs->att.data, layer_bufs->qkv.data, B, S, n_head, h);

        mlp_forward<<<MLP_FORWARD_GRID(h, B, S), MLP_BLOCK_DIM>>>(layer_bufs->att_proj.data, layer_bufs->atty.data, block->attn.proj_w.data, block->attn.proj_b.data, B, S, h, h);
        residual_forward<<<CEIL_DIV(B * S * h, thr), thr>>>(layer_bufs->res_2.data, layer_bufs->att_proj.data, res.data, B, S, h);

        layernorm_forward<<<B, thr>>>(layer_bufs->ln_2.data, layer_bufs->res_2.data, block->ln_2.w.data, block->ln_2.b.data, layer_bufs->ln_2_mean.data, layer_bufs->ln_2_rstd.data, S, h);

        mlp_forward<<<MLP_FORWARD_GRID(h * 4, B, S), MLP_BLOCK_DIM>>>(layer_bufs->mlp_fc.data, layer_bufs->ln_2.data, block->mlp.fc_w.data, block->mlp.fc_b.data, B, S, h, h * 4);

        gelu_forward<<<CEIL_DIV(B * S * 4 * h, thr), thr>>>(layer_bufs->mlp_fc_gelu.data, layer_bufs->mlp_fc.data, B, S, h * 4);

        mlp_forward<<<MLP_FORWARD_GRID(h, B, S), MLP_BLOCK_DIM>>>(layer_bufs->mlp_proj.data, layer_bufs->mlp_fc_gelu.data, block->mlp.proj_w.data, block->mlp.proj_b.data, B, S, h * 4, h);
        residual_forward<<<CEIL_DIV(B * S * h, thr), thr>>>(layer_bufs->res_3.data, layer_bufs->mlp_proj.data, layer_bufs->res_2.data, B, S, h);
    }

    tensor_t res = buffers.blocks[L - 1].res_3;
    layernorm_forward<<<B, thr>>>(buffers.ln_f.data, res.data, model.ln_f.w.data, model.ln_f.b.data, buffers.ln_f_mean.data, buffers.ln_f_rstd.data, S, h);

    mlp_forward<<<MLP_FORWARD_GRID(V, B, S), MLP_BLOCK_DIM>>>(buffers.logits.data, buffers.ln_f.data, model.emb.wte.data, NULL, B, S, h, V);

    softmax_forward<<<CEIL_DIV(B * S * V, thr), thr>>>(buffers.probs.data, buffers.logits.data, B, S, V);
}

void cross_entropy(config_t config, train_buffers_t buffers, const int *d_target_tokens, int seq_len) {
    int B = config.batch_size;
    int S = seq_len;
    int V = config.vocab_size;

    int thr = 1024;
    cross_entropy_forward<<<CEIL_DIV(B * S, thr), thr>>>(buffers.losses.data, buffers.probs.data, d_target_tokens, B, S, V);
}

void backward(config_t config, gpt2_t model, train_buffers_t buffers, gpt2_t g_model, train_buffers_t g_buffers, const int *d_input_tokens, const int *d_target_tokens, int seq_len) {
    int L = config.n_layer;
    int B = config.batch_size;
    int S = seq_len;
    int h = config.n_embd;
    int n_head = config.n_head;
    int V = config.vocab_size;

    int thr = 1024;

    cross_entropy_backward<<<CEIL_DIV(B * S, thr), thr>>>(g_buffers.logits.data, buffers.probs.data, d_target_tokens, B, S, V);

    mlp_backward_input<<<MLP_BACKWARD_INPUT_GRID(h, B, S), MLP_BLOCK_DIM>>>(g_buffers.ln_f.data, g_buffers.logits.data, model.emb.wte.data, B, S, h, V);
    mlp_backward_weight<<<MLP_BACKWARD_WEIGHT_GRID(V, h), MLP_BLOCK_DIM>>>(g_model.emb.wte.data, NULL, g_buffers.logits.data, buffers.ln_f.data, B, S, h, V);

    tensor_t res = buffers.blocks[L - 1].res_3;
    tensor_t g_res = g_buffers.blocks[L - 1].res_3;

    layernorm_backward<<<B, thr>>>(g_res.data, g_model.ln_f.w.data, g_model.ln_f.b.data, g_buffers.ln_f.data, res.data, model.ln_f.w.data, buffers.ln_f_mean.data, buffers.ln_f_rstd.data, B, S, h);

    for (int layer_idx = L - 1; layer_idx >= 0; layer_idx--)
    {
        block_t *block = &model.h[layer_idx];
        block_t *g_block = &g_model.h[layer_idx];
        layer_buffers_t *layer_bufs = &buffers.blocks[layer_idx];
        layer_buffers_t *g_layer_bufs = &g_buffers.blocks[layer_idx];

        tensor_t res = (layer_idx == 0) ? buffers.encoded : buffers.blocks[layer_idx - 1].res_3;
        tensor_t g_res = (layer_idx == 0) ? g_buffers.encoded : g_buffers.blocks[layer_idx - 1].res_3;

        residual_backward<<<CEIL_DIV(B * S * h, thr), thr>>>(g_layer_bufs->res_2.data, g_layer_bufs->mlp_proj.data, g_layer_bufs->res_3.data, B * S * h);

        mlp_backward_input<<<MLP_BACKWARD_INPUT_GRID(h * 4, B, S), MLP_BLOCK_DIM>>>(g_layer_bufs->mlp_fc_gelu.data, g_layer_bufs->mlp_proj.data, block->mlp.proj_w.data, B, S, h * 4, h);
        mlp_backward_weight<<<MLP_BACKWARD_WEIGHT_GRID(h, h * 4), MLP_BLOCK_DIM>>>(g_block->mlp.proj_w.data, g_block->mlp.proj_b.data, g_layer_bufs->mlp_proj.data, layer_bufs->mlp_fc_gelu.data, B, S, h * 4, h);

        gelu_backward<<<CEIL_DIV(B * S * 4 * h, thr), thr>>>(g_layer_bufs->mlp_fc.data, layer_bufs->mlp_fc.data, g_layer_bufs->mlp_fc_gelu.data, B * S * 4 * h);

        mlp_backward_input<<<MLP_BACKWARD_INPUT_GRID(h, B, S), MLP_BLOCK_DIM>>>(g_layer_bufs->ln_2.data, g_layer_bufs->mlp_fc.data, block->mlp.fc_w.data, B, S, h, h * 4);
        mlp_backward_weight<<<MLP_BACKWARD_WEIGHT_GRID(h * 4, h), MLP_BLOCK_DIM>>>(g_block->mlp.fc_w.data, g_block->mlp.fc_b.data, g_layer_bufs->mlp_fc.data, layer_bufs->ln_2.data, B, S, h, h * 4);

        layernorm_backward<<<B, thr>>>(g_layer_bufs->res_2.data, g_block->ln_2.w.data, g_block->ln_2.b.data, g_layer_bufs->ln_2.data, layer_bufs->res_2.data, block->ln_2.w.data, layer_bufs->ln_2_mean.data, layer_bufs->ln_2_rstd.data, B, S, h);

        residual_backward<<<CEIL_DIV(B * S * h, thr), thr>>>(g_res.data, g_layer_bufs->att_proj.data, g_layer_bufs->res_2.data, B * S * h);

        mlp_backward_input<<<MLP_BACKWARD_INPUT_GRID(h, B, S), MLP_BLOCK_DIM>>>(g_layer_bufs->atty.data, g_layer_bufs->att_proj.data, block->attn.proj_w.data, B, S, h, h);
        mlp_backward_weight<<<MLP_BACKWARD_WEIGHT_GRID(h, h), MLP_BLOCK_DIM>>>(g_block->attn.proj_w.data, g_block->attn.proj_b.data, g_layer_bufs->att_proj.data, layer_bufs->atty.data, B, S, h, h);

        attention_backward<<<CEIL_DIV(B * S * n_head, thr), thr>>>(g_layer_bufs->qkv.data, g_layer_bufs->preatt.data, g_layer_bufs->att.data, g_layer_bufs->atty.data, layer_bufs->qkv.data, layer_bufs->att.data, B, S, h, n_head);

        mlp_backward_input<<<MLP_BACKWARD_INPUT_GRID(h, B, S), MLP_BLOCK_DIM>>>(g_layer_bufs->ln_1.data, g_layer_bufs->qkv.data, block->attn.qkv_w.data, B, S, h, h * 3);
        mlp_backward_weight<<<MLP_BACKWARD_WEIGHT_GRID(h * 3, h), MLP_BLOCK_DIM>>>(g_block->attn.qkv_w.data, g_block->attn.qkv_b.data, g_layer_bufs->qkv.data, layer_bufs->ln_1.data, B, S, h, h * 3);

        layernorm_backward<<<B, thr>>>(g_res.data, g_block->ln_1.w.data, g_block->ln_1.b.data, g_layer_bufs->ln_1.data, res.data, block->ln_1.w.data, layer_bufs->ln_1_mean.data, layer_bufs->ln_1_rstd.data, B, S, h);
    }

    embedding_backward<<<CEIL_DIV(B * S, thr), thr>>>(g_model.emb.wte.data, g_model.emb.wpe.data, g_buffers.encoded.data, d_input_tokens, B, S, h);
}

void gpt2_update(gpt2_t *model, gpt2_t *grads, adamw_state_t *opt) {
    if (opt->m_memory == NULL) {
        size_t num_params = model->num_parameters;
        gpuErrchk(cudaMalloc(&opt->m_memory, num_params * sizeof(float)));
        gpuErrchk(cudaMalloc(&opt->v_memory, num_params * sizeof(float)));
        
        gpuErrchk(cudaMemset(opt->m_memory, 0, num_params * sizeof(float)));
        gpuErrchk(cudaMemset(opt->v_memory, 0, num_params * sizeof(float)));
    }
    
    opt->t++;
    
    int thr = 1024;
    int num_blocks = CEIL_DIV(model->num_parameters, thr);

    adamw_kernel<<<num_blocks, thr>>>(
        model->params_memory,
        grads->params_memory,
        opt->m_memory,
        opt->v_memory,
        model->num_parameters,
        opt->learning_rate,
        opt->beta1,
        opt->beta2,
        opt->eps,
        opt->weight_decay,
        opt->t
    );
    
    gpuErrchk(cudaGetLastError());
}
