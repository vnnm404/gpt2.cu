#include <stdio.h>

#include "gpt2/gpt2.h"
#include "gpt2/train.h"
#include "gpt2/tensor.h"

#include "gpt2/layers/embedding.h"
#include "gpt2/layers/layernorm.h"
#include "gpt2/layers/mlp.h"
#include "gpt2/layers/attention.h"
#include "gpt2/layers/residual.h"
#include "gpt2/layers/gelu.h"
#include "gpt2/layers/softmax.h"
#include "gpt2/layers/cross_entropy.h"
#include "gpt2/layers/adamw.h"

#define THREADS 256

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// Helper: compute mean loss (used by test main)
float compute_mean_loss(tensor_t *losses, int B, int S)
{
    float *cpu_losses = (float *)malloc(B * S * sizeof(float));
    cudaMemcpy(cpu_losses, losses->data, B * S * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < B * S; i++)
    {
        sum += cpu_losses[i];
    }

    free(cpu_losses);
    return sum / (B * S);
}

// Implementation of setup/free/forward/backward/update functions copied from test
int setup_train_buffers(config_t config, train_buffers_t *buffers, int seq_len)
{
    int B = config.batch_size;
    int S = seq_len;
    int h = config.n_embd;
    int n_head = config.n_head;
    int V = config.vocab_size;
    int four_h = 4 * h;

    // Calculate total size needed for all activations
    size_t total_size = 0;

    // encoded: B * S * h
    total_size += B * S * h;

    // Per layer activations
    for (int i = 0; i < config.n_layer; i++)
    {
        total_size += B * S * h;          // ln_1
        total_size += B * S;              // ln_1_mean
        total_size += B * S;              // ln_1_rstd
        total_size += B * S * 3 * h;      // qkv
        total_size += B * S * h;          // atty
        total_size += B * n_head * S * S; // preatt
        total_size += B * n_head * S * S; // att
        total_size += B * S * h;          // att_proj
        total_size += B * S * h;          // res_2
        total_size += B * S * h;          // ln_2
        total_size += B * S;              // ln_2_mean
        total_size += B * S;              // ln_2_rstd
        total_size += B * S * four_h;     // mlp_fc
        total_size += B * S * four_h;     // mlp_fc_gelu
        total_size += B * S * h;          // mlp_proj
        total_size += B * S * h;          // res_3
    }

    // Final layers
    total_size += B * S * h; // ln_f
    total_size += B * S;     // ln_f_mean
    total_size += B * S;     // ln_f_rstd
    total_size += B * S * V; // logits
    total_size += B * S * V; // probs
    total_size += B * S;     // losses

    // Allocate single contiguous block on GPU
    cudaError_t err = cudaMalloc(&buffers->activations_memory, total_size * sizeof(float));
    if (err != cudaSuccess)
    {
        return -1;
    }

    // Now set up tensor structures with pointers into this block
    float *ptr = buffers->activations_memory;

    // encoded
    int encoded_shape[3] = {B, S, h};
    buffers->encoded.ndim = 3;
    buffers->encoded.shape[0] = encoded_shape[0];
    buffers->encoded.shape[1] = encoded_shape[1];
    buffers->encoded.shape[2] = encoded_shape[2];
    buffers->encoded.shape[3] = 0;
    buffers->encoded.data = ptr;
    ptr += B * S * h;

    // Per layer activations
    for (int i = 0; i < config.n_layer; i++)
    {
        layer_buffers_t *layer_bufs = &buffers->blocks[i];

        // ln_1
        layer_bufs->ln_1.ndim = 3;
        layer_bufs->ln_1.shape[0] = B;
        layer_bufs->ln_1.shape[1] = S;
        layer_bufs->ln_1.shape[2] = h;
        layer_bufs->ln_1.shape[3] = 0;
        layer_bufs->ln_1.data = ptr;
        ptr += B * S * h;

        // ln_1_mean
        layer_bufs->ln_1_mean.ndim = 2;
        layer_bufs->ln_1_mean.shape[0] = B;
        layer_bufs->ln_1_mean.shape[1] = S;
        layer_bufs->ln_1_mean.shape[2] = 0;
        layer_bufs->ln_1_mean.shape[3] = 0;
        layer_bufs->ln_1_mean.data = ptr;
        ptr += B * S;

        // ln_1_rstd
        layer_bufs->ln_1_rstd.ndim = 2;
        layer_bufs->ln_1_rstd.shape[0] = B;
        layer_bufs->ln_1_rstd.shape[1] = S;
        layer_bufs->ln_1_rstd.shape[2] = 0;
        layer_bufs->ln_1_rstd.shape[3] = 0;
        layer_bufs->ln_1_rstd.data = ptr;
        ptr += B * S;

        // qkv
        layer_bufs->qkv.ndim = 3;
        layer_bufs->qkv.shape[0] = B;
        layer_bufs->qkv.shape[1] = S;
        layer_bufs->qkv.shape[2] = 3 * h;
        layer_bufs->qkv.shape[3] = 0;
        layer_bufs->qkv.data = ptr;
        ptr += B * S * 3 * h;

        // atty
        layer_bufs->atty.ndim = 3;
        layer_bufs->atty.shape[0] = B;
        layer_bufs->atty.shape[1] = S;
        layer_bufs->atty.shape[2] = h;
        layer_bufs->atty.shape[3] = 0;
        layer_bufs->atty.data = ptr;
        ptr += B * S * h;

        // preatt
        layer_bufs->preatt.ndim = 4;
        layer_bufs->preatt.shape[0] = B;
        layer_bufs->preatt.shape[1] = n_head;
        layer_bufs->preatt.shape[2] = S;
        layer_bufs->preatt.shape[3] = S;
        layer_bufs->preatt.data = ptr;
        ptr += B * n_head * S * S;

        // att
        layer_bufs->att.ndim = 4;
        layer_bufs->att.shape[0] = B;
        layer_bufs->att.shape[1] = n_head;
        layer_bufs->att.shape[2] = S;
        layer_bufs->att.shape[3] = S;
        layer_bufs->att.data = ptr;
        ptr += B * n_head * S * S;

        // att_proj
        layer_bufs->att_proj.ndim = 3;
        layer_bufs->att_proj.shape[0] = B;
        layer_bufs->att_proj.shape[1] = S;
        layer_bufs->att_proj.shape[2] = h;
        layer_bufs->att_proj.shape[3] = 0;
        layer_bufs->att_proj.data = ptr;
        ptr += B * S * h;

        // res_2
        layer_bufs->res_2.ndim = 3;
        layer_bufs->res_2.shape[0] = B;
        layer_bufs->res_2.shape[1] = S;
        layer_bufs->res_2.shape[2] = h;
        layer_bufs->res_2.shape[3] = 0;
        layer_bufs->res_2.data = ptr;
        ptr += B * S * h;

        // ln_2
        layer_bufs->ln_2.ndim = 3;
        layer_bufs->ln_2.shape[0] = B;
        layer_bufs->ln_2.shape[1] = S;
        layer_bufs->ln_2.shape[2] = h;
        layer_bufs->ln_2.shape[3] = 0;
        layer_bufs->ln_2.data = ptr;
        ptr += B * S * h;

        // ln_2_mean
        layer_bufs->ln_2_mean.ndim = 2;
        layer_bufs->ln_2_mean.shape[0] = B;
        layer_bufs->ln_2_mean.shape[1] = S;
        layer_bufs->ln_2_mean.shape[2] = 0;
        layer_bufs->ln_2_mean.shape[3] = 0;
        layer_bufs->ln_2_mean.data = ptr;
        ptr += B * S;

        // ln_2_rstd
        layer_bufs->ln_2_rstd.ndim = 2;
        layer_bufs->ln_2_rstd.shape[0] = B;
        layer_bufs->ln_2_rstd.shape[1] = S;
        layer_bufs->ln_2_rstd.shape[2] = 0;
        layer_bufs->ln_2_rstd.shape[3] = 0;
        layer_bufs->ln_2_rstd.data = ptr;
        ptr += B * S;

        // mlp_fc
        layer_bufs->mlp_fc.ndim = 3;
        layer_bufs->mlp_fc.shape[0] = B;
        layer_bufs->mlp_fc.shape[1] = S;
        layer_bufs->mlp_fc.shape[2] = four_h;
        layer_bufs->mlp_fc.shape[3] = 0;
        layer_bufs->mlp_fc.data = ptr;
        ptr += B * S * four_h;

        // mlp_fc_gelu
        layer_bufs->mlp_fc_gelu.ndim = 3;
        layer_bufs->mlp_fc_gelu.shape[0] = B;
        layer_bufs->mlp_fc_gelu.shape[1] = S;
        layer_bufs->mlp_fc_gelu.shape[2] = four_h;
        layer_bufs->mlp_fc_gelu.shape[3] = 0;
        layer_bufs->mlp_fc_gelu.data = ptr;
        ptr += B * S * four_h;

        // mlp_proj
        layer_bufs->mlp_proj.ndim = 3;
        layer_bufs->mlp_proj.shape[0] = B;
        layer_bufs->mlp_proj.shape[1] = S;
        layer_bufs->mlp_proj.shape[2] = h;
        layer_bufs->mlp_proj.shape[3] = 0;
        layer_bufs->mlp_proj.data = ptr;
        ptr += B * S * h;

        // res_3
        layer_bufs->res_3.ndim = 3;
        layer_bufs->res_3.shape[0] = B;
        layer_bufs->res_3.shape[1] = S;
        layer_bufs->res_3.shape[2] = h;
        layer_bufs->res_3.shape[3] = 0;
        layer_bufs->res_3.data = ptr;
        ptr += B * S * h;
    }

    // Final layers
    // ln_f
    buffers->ln_f.ndim = 3;
    buffers->ln_f.shape[0] = B;
    buffers->ln_f.shape[1] = S;
    buffers->ln_f.shape[2] = h;
    buffers->ln_f.shape[3] = 0;
    buffers->ln_f.data = ptr;
    ptr += B * S * h;

    // ln_f_mean
    buffers->ln_f_mean.ndim = 2;
    buffers->ln_f_mean.shape[0] = B;
    buffers->ln_f_mean.shape[1] = S;
    buffers->ln_f_mean.shape[2] = 0;
    buffers->ln_f_mean.shape[3] = 0;
    buffers->ln_f_mean.data = ptr;
    ptr += B * S;

    // ln_f_rstd
    buffers->ln_f_rstd.ndim = 2;
    buffers->ln_f_rstd.shape[0] = B;
    buffers->ln_f_rstd.shape[1] = S;
    buffers->ln_f_rstd.shape[2] = 0;
    buffers->ln_f_rstd.shape[3] = 0;
    buffers->ln_f_rstd.data = ptr;
    ptr += B * S;

    // logits
    buffers->logits.ndim = 3;
    buffers->logits.shape[0] = B;
    buffers->logits.shape[1] = S;
    buffers->logits.shape[2] = V;
    buffers->logits.shape[3] = 0;
    buffers->logits.data = ptr;
    ptr += B * S * V;

    // probs
    buffers->probs.ndim = 3;
    buffers->probs.shape[0] = B;
    buffers->probs.shape[1] = S;
    buffers->probs.shape[2] = V;
    buffers->probs.shape[3] = 0;
    buffers->probs.data = ptr;
    ptr += B * S * V;

    // losses
    buffers->losses.ndim = 2;
    buffers->losses.shape[0] = B;
    buffers->losses.shape[1] = S;
    buffers->losses.shape[2] = 0;
    buffers->losses.shape[3] = 0;
    buffers->losses.data = ptr;
    ptr += B * S;

    return 0;
}

void free_train_buffers(train_buffers_t *buffers)
{
    // Free the single contiguous activation memory block
    if (buffers->activations_memory)
    {
        cudaFree(buffers->activations_memory);
        buffers->activations_memory = NULL;
    }
}

void forward(config_t config, gpt2_t model, train_buffers_t buffers, const int *d_input_tokens, int seq_len)
{
    int L = config.n_layer;
    int B = config.batch_size;
    int S = seq_len;
    int h = config.n_embd;
    int n_head = config.n_head;
    int V = config.vocab_size;

    int thr = THREADS;

    embedding_forward<<<B, thr>>>(buffers.encoded.data, d_input_tokens, model.emb.wte.data, model.emb.wpe.data, S, h, V, config.n_positions);

    for (int layer_idx = 0; layer_idx < config.n_layer; layer_idx++)
    {
        block_t *block = &model.h[layer_idx];
        layer_buffers_t *layer_bufs = &buffers.blocks[layer_idx];

        tensor_t res = (layer_idx == 0) ? buffers.encoded : buffers.blocks[layer_idx - 1].res_3;

        layernorm_forward<<<B * S, LN_BLOCK_SIZE, LN_SMEM_SIZE>>>(layer_bufs->ln_1.data, res.data, block->ln_1.w.data, block->ln_1.b.data, layer_bufs->ln_1_mean.data, layer_bufs->ln_1_rstd.data, S, h);

        mlp_forward<TILE_SIZE><<<MLP_FORWARD_GRID(h * 3, B, S), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(layer_bufs->qkv.data, layer_bufs->ln_1.data, block->attn.qkv_w.data, block->attn.qkv_b.data, B, S, h, h * 3);

        attention_forward<<<ATTN_FWD_GRID(B, S, n_head), ATTN_FWD_BLOCK(h / n_head), ATTN_FWD_SMEM(S, h / n_head)>>>(layer_bufs->atty.data, layer_bufs->preatt.data, layer_bufs->att.data, layer_bufs->qkv.data, B, S, n_head, h);

        mlp_forward<TILE_SIZE><<<MLP_FORWARD_GRID(h, B, S), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(layer_bufs->att_proj.data, layer_bufs->atty.data, block->attn.proj_w.data, block->attn.proj_b.data, B, S, h, h);
        residual_forward<<<CEIL_DIV(B * S * h, thr), thr>>>(layer_bufs->res_2.data, layer_bufs->att_proj.data, res.data, B, S, h);

        layernorm_forward<<<B * S, LN_BLOCK_SIZE, LN_SMEM_SIZE>>>(layer_bufs->ln_2.data, layer_bufs->res_2.data, block->ln_2.w.data, block->ln_2.b.data, layer_bufs->ln_2_mean.data, layer_bufs->ln_2_rstd.data, S, h);

        mlp_forward<TILE_SIZE><<<MLP_FORWARD_GRID(h * 4, B, S), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(layer_bufs->mlp_fc.data, layer_bufs->ln_2.data, block->mlp.fc_w.data, block->mlp.fc_b.data, B, S, h, h * 4);

        gelu_forward<<<CEIL_DIV(B * S * 4 * h, thr), thr>>>(layer_bufs->mlp_fc_gelu.data, layer_bufs->mlp_fc.data, B, S, h * 4);

        mlp_forward<TILE_SIZE><<<MLP_FORWARD_GRID(h, B, S), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(layer_bufs->mlp_proj.data, layer_bufs->mlp_fc_gelu.data, block->mlp.proj_w.data, block->mlp.proj_b.data, B, S, h * 4, h);
        residual_forward<<<CEIL_DIV(B * S * h, thr), thr>>>(layer_bufs->res_3.data, layer_bufs->mlp_proj.data, layer_bufs->res_2.data, B, S, h);
    }

    tensor_t res = buffers.blocks[L - 1].res_3;
    layernorm_forward<<<B * S, LN_BLOCK_SIZE, LN_SMEM_SIZE>>>(buffers.ln_f.data, res.data, model.ln_f.w.data, model.ln_f.b.data, buffers.ln_f_mean.data, buffers.ln_f_rstd.data, S, h);

    mlp_forward<TILE_SIZE><<<MLP_FORWARD_GRID(V, B, S), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(buffers.logits.data, buffers.ln_f.data, model.emb.wte.data, NULL, B, S, h, V);

    softmax_forward<<<CEIL_DIV(B * S * V, thr), thr>>>(buffers.probs.data, buffers.logits.data, B, S, V);
}

void cross_entropy(config_t config, train_buffers_t buffers, const int *d_target_tokens, int seq_len)
{
    int B = config.batch_size;
    int S = seq_len;
    int V = config.vocab_size;

    int thr = THREADS;
    cross_entropy_forward<<<CEIL_DIV(B * S, thr), thr>>>(buffers.losses.data, buffers.probs.data, d_target_tokens, B, S, V);
}

void backward(config_t config, gpt2_t model, train_buffers_t buffers, gpt2_t g_model, train_buffers_t g_buffers, const int *d_input_tokens, const int *d_target_tokens, int seq_len)
{
    int L = config.n_layer;
    int B = config.batch_size;
    int S = seq_len;
    int h = config.n_embd;
    int n_head = config.n_head;
    int V = config.vocab_size;

    int thr = THREADS;

    cross_entropy_backward<<<CEIL_DIV(B * S, thr), thr>>>(g_buffers.logits.data, buffers.probs.data, d_target_tokens, B, S, V);

    mlp_backward_input<TILE_SIZE><<<MLP_BACKWARD_INPUT_GRID(h, B, S), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(g_buffers.ln_f.data, g_buffers.logits.data, model.emb.wte.data, B, S, h, V);
    mlp_backward_weight<TILE_SIZE><<<MLP_BACKWARD_WEIGHT_GRID(V, h), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(g_model.emb.wte.data, NULL, g_buffers.logits.data, buffers.ln_f.data, B, S, h, V);

    tensor_t res = buffers.blocks[L - 1].res_3;
    tensor_t g_res = g_buffers.blocks[L - 1].res_3;

    layernorm_backward<<<B * S, LN_BLOCK_SIZE, LN_SMEM_SIZE>>>(g_res.data, g_model.ln_f.w.data, g_model.ln_f.b.data, g_buffers.ln_f.data, res.data, model.ln_f.w.data, buffers.ln_f_mean.data, buffers.ln_f_rstd.data, B, S, h);

    for (int layer_idx = L - 1; layer_idx >= 0; layer_idx--)
    {
        block_t *block = &model.h[layer_idx];
        block_t *g_block = &g_model.h[layer_idx];
        layer_buffers_t *layer_bufs = &buffers.blocks[layer_idx];
        layer_buffers_t *g_layer_bufs = &g_buffers.blocks[layer_idx];

        tensor_t res = (layer_idx == 0) ? buffers.encoded : buffers.blocks[layer_idx - 1].res_3;
        tensor_t g_res = (layer_idx == 0) ? g_buffers.encoded : g_buffers.blocks[layer_idx - 1].res_3;

        residual_backward<<<CEIL_DIV(B * S * h, thr), thr>>>(g_layer_bufs->res_2.data, g_layer_bufs->mlp_proj.data, g_layer_bufs->res_3.data, B * S * h);

        mlp_backward_input<TILE_SIZE><<<MLP_BACKWARD_INPUT_GRID(h * 4, B, S), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(g_layer_bufs->mlp_fc_gelu.data, g_layer_bufs->mlp_proj.data, block->mlp.proj_w.data, B, S, h * 4, h);
        mlp_backward_weight<TILE_SIZE><<<MLP_BACKWARD_WEIGHT_GRID(h, h * 4), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(g_block->mlp.proj_w.data, g_block->mlp.proj_b.data, g_layer_bufs->mlp_proj.data, layer_bufs->mlp_fc_gelu.data, B, S, h * 4, h);

        gelu_backward<<<CEIL_DIV(B * S * 4 * h, thr), thr>>>(g_layer_bufs->mlp_fc.data, layer_bufs->mlp_fc.data, g_layer_bufs->mlp_fc_gelu.data, B * S * 4 * h);

        mlp_backward_input<TILE_SIZE><<<MLP_BACKWARD_INPUT_GRID(h, B, S), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(g_layer_bufs->ln_2.data, g_layer_bufs->mlp_fc.data, block->mlp.fc_w.data, B, S, h, h * 4);
        mlp_backward_weight<TILE_SIZE><<<MLP_BACKWARD_WEIGHT_GRID(h * 4, h), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(g_block->mlp.fc_w.data, g_block->mlp.fc_b.data, g_layer_bufs->mlp_fc.data, layer_bufs->ln_2.data, B, S, h, h * 4);

        layernorm_backward<<<B * S, LN_BLOCK_SIZE, LN_SMEM_SIZE>>>(g_layer_bufs->res_2.data, g_block->ln_2.w.data, g_block->ln_2.b.data, g_layer_bufs->ln_2.data, layer_bufs->res_2.data, block->ln_2.w.data, layer_bufs->ln_2_mean.data, layer_bufs->ln_2_rstd.data, B, S, h);

        residual_backward<<<CEIL_DIV(B * S * h, thr), thr>>>(g_res.data, g_layer_bufs->att_proj.data, g_layer_bufs->res_2.data, B * S * h);

        mlp_backward_input<TILE_SIZE><<<MLP_BACKWARD_INPUT_GRID(h, B, S), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(g_layer_bufs->atty.data, g_layer_bufs->att_proj.data, block->attn.proj_w.data, B, S, h, h);
        mlp_backward_weight<TILE_SIZE><<<MLP_BACKWARD_WEIGHT_GRID(h, h), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(g_block->attn.proj_w.data, g_block->attn.proj_b.data, g_layer_bufs->att_proj.data, layer_bufs->atty.data, B, S, h, h);

        attention_backward<<<ATTN_BWD_GRID(B, S, n_head), ATTN_BWD_BLOCK(h / n_head), ATTN_BWD_SMEM(S, h / n_head)>>>(g_layer_bufs->qkv.data, g_layer_bufs->preatt.data, g_layer_bufs->att.data, g_layer_bufs->atty.data, layer_bufs->qkv.data, layer_bufs->att.data, B, S, h, n_head);

        mlp_backward_input<TILE_SIZE><<<MLP_BACKWARD_INPUT_GRID(h, B, S), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(g_layer_bufs->ln_1.data, g_layer_bufs->qkv.data, block->attn.qkv_w.data, B, S, h, h * 3);
        mlp_backward_weight<TILE_SIZE><<<MLP_BACKWARD_WEIGHT_GRID(h * 3, h), MLP_BLOCK_DIM, MLP_SHARED_MEM_SIZE>>>(g_block->attn.qkv_w.data, g_block->attn.qkv_b.data, g_layer_bufs->qkv.data, layer_bufs->ln_1.data, B, S, h, h * 3);

        layernorm_backward<<<B * S, LN_BLOCK_SIZE, LN_SMEM_SIZE>>>(g_res.data, g_block->ln_1.w.data, g_block->ln_1.b.data, g_layer_bufs->ln_1.data, res.data, block->ln_1.w.data, layer_bufs->ln_1_mean.data, layer_bufs->ln_1_rstd.data, B, S, h);
    }

    embedding_backward<<<CEIL_DIV(B * S, thr), thr>>>(g_model.emb.wte.data, g_model.emb.wpe.data, g_buffers.encoded.data, d_input_tokens, B, S, h);
}

void gpt2_update(gpt2_t *model, gpt2_t *grads, adamw_state_t *opt)
{
    if (opt->m_memory == NULL)
    {
        size_t num_params = model->num_parameters;
        gpuErrchk(cudaMalloc(&opt->m_memory, num_params * sizeof(float)));
        gpuErrchk(cudaMalloc(&opt->v_memory, num_params * sizeof(float)));

        gpuErrchk(cudaMemset(opt->m_memory, 0, num_params * sizeof(float)));
        gpuErrchk(cudaMemset(opt->v_memory, 0, num_params * sizeof(float)));
    }

    opt->t++;

    int thr = THREADS;
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
        opt->t);

    gpuErrchk(cudaGetLastError());
}

void gpt2_zero_grad(gpt2_t *grads)
{
    cudaMemset(grads->params_memory, 0, grads->num_parameters * sizeof(float));
}

void zero_activation_grads(config_t config, train_buffers_t *g_buffers)
{
    cudaMemset(g_buffers->encoded.data, 0, tensor_size(g_buffers->encoded) * sizeof(float));

    for (int i = 0; i < config.n_layer; i++)
    {
        layer_buffers_t *g_layer = &g_buffers->blocks[i];

        cudaMemset(g_layer->ln_1.data, 0, tensor_size(g_layer->ln_1) * sizeof(float));
        cudaMemset(g_layer->ln_1_mean.data, 0, tensor_size(g_layer->ln_1_mean) * sizeof(float));
        cudaMemset(g_layer->ln_1_rstd.data, 0, tensor_size(g_layer->ln_1_rstd) * sizeof(float));
        cudaMemset(g_layer->qkv.data, 0, tensor_size(g_layer->qkv) * sizeof(float));
        cudaMemset(g_layer->atty.data, 0, tensor_size(g_layer->atty) * sizeof(float));
        cudaMemset(g_layer->preatt.data, 0, tensor_size(g_layer->preatt) * sizeof(float));
        cudaMemset(g_layer->att.data, 0, tensor_size(g_layer->att) * sizeof(float));
        cudaMemset(g_layer->att_proj.data, 0, tensor_size(g_layer->att_proj) * sizeof(float));
        cudaMemset(g_layer->res_2.data, 0, tensor_size(g_layer->res_2) * sizeof(float));
        cudaMemset(g_layer->ln_2.data, 0, tensor_size(g_layer->ln_2) * sizeof(float));
        cudaMemset(g_layer->ln_2_mean.data, 0, tensor_size(g_layer->ln_2_mean) * sizeof(float));
        cudaMemset(g_layer->ln_2_rstd.data, 0, tensor_size(g_layer->ln_2_rstd) * sizeof(float));
        cudaMemset(g_layer->mlp_fc.data, 0, tensor_size(g_layer->mlp_fc) * sizeof(float));
        cudaMemset(g_layer->mlp_fc_gelu.data, 0, tensor_size(g_layer->mlp_fc_gelu) * sizeof(float));
        cudaMemset(g_layer->mlp_proj.data, 0, tensor_size(g_layer->mlp_proj) * sizeof(float));
        cudaMemset(g_layer->res_3.data, 0, tensor_size(g_layer->res_3) * sizeof(float));
    }

    cudaMemset(g_buffers->ln_f.data, 0, tensor_size(g_buffers->ln_f) * sizeof(float));
    cudaMemset(g_buffers->ln_f_mean.data, 0, tensor_size(g_buffers->ln_f_mean) * sizeof(float));
    cudaMemset(g_buffers->ln_f_rstd.data, 0, tensor_size(g_buffers->ln_f_rstd) * sizeof(float));
    cudaMemset(g_buffers->logits.data, 0, tensor_size(g_buffers->logits) * sizeof(float));
    cudaMemset(g_buffers->probs.data, 0, tensor_size(g_buffers->probs) * sizeof(float));
    cudaMemset(g_buffers->losses.data, 0, tensor_size(g_buffers->losses) * sizeof(float));
}