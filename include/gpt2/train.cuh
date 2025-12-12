#pragma once

#include <cuda_runtime.h>

#include "gpt2/model.cuh"
#include "gpt2/tensor.cuh"

typedef struct {
    tensor_t ln_1;
    tensor_t ln_1_mean;
    tensor_t ln_1_rstd;
    tensor_t qkv;
    tensor_t atty;
    tensor_t preatt;
    tensor_t att;
    tensor_t att_proj;
    tensor_t res_2;
    tensor_t ln_2;
    tensor_t ln_2_mean;
    tensor_t ln_2_rstd;
    tensor_t mlp_fc;
    tensor_t mlp_fc_gelu;
    tensor_t mlp_proj;
    tensor_t res_3;
} layer_buffers_t;

typedef struct {
    float *activations_memory;
    tensor_t encoded;
    layer_buffers_t blocks[NUM_LAYERS];
    tensor_t ln_f;
    tensor_t ln_f_mean;
    tensor_t ln_f_rstd;
    tensor_t logits;
    tensor_t probs;
    tensor_t losses;
} train_buffers_t;

typedef struct {
    float *m_memory;
    float *v_memory;
    float learning_rate;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int t;
} adamw_state_t;

float compute_mean_loss(tensor_t *losses, int B, int S) {
    float *cpu_losses = (float *)malloc(B * S * sizeof(float));
    cudaMemcpy(cpu_losses, losses->data, B * S * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = 0.0f;
    for (int i = 0; i < B * S; i++) {
        sum += cpu_losses[i];
    }
    
    free(cpu_losses);
    return sum / (B * S);
}

int setup_train_buffers(config_t config, train_buffers_t *buffers, int seq_len)
{
    int B = config.batch_size;
    int S = seq_len;
    int h = config.n_embd;
    int n_head = config.n_head;
    int V = config.vocab_size;
    int four_h = 4 * h;

    size_t total_size = 0;
    
    total_size += B * S * h;
    
    for (int i = 0; i < config.n_layer; i++) {
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
    
    total_size += B * S * h;  // ln_f
    total_size += B * S;      // ln_f_mean
    total_size += B * S;      // ln_f_rstd
    total_size += B * S * V;  // logits
    total_size += B * S * V;  // probs
    total_size += B * S;      // losses
    
    cudaError_t err = cudaMalloc(&buffers->activations_memory, total_size * sizeof(float));
    if (err != cudaSuccess) {
        return -1;
    }
    
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
    if (buffers->activations_memory) {
        cudaFree(buffers->activations_memory);
        buffers->activations_memory = NULL;
    }
}

void gpt2_zero_grad(gpt2_t *grads) {
    cudaMemset(grads->params_memory, 0, grads->num_parameters * sizeof(float));
}

void zero_activation_grads(config_t config, train_buffers_t *g_buffers) {
    cudaMemset(g_buffers->encoded.data, 0, tensor_size(g_buffers->encoded) * sizeof(float));
    
    for (int i = 0; i < config.n_layer; i++) {
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
