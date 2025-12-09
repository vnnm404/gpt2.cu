#pragma once

#include <cuda_runtime.h>
#include "gpt2/gpt2.h"
#include "gpt2/tensor.h"


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

float compute_mean_loss(tensor_t *losses, int B, int S);
int setup_train_buffers(config_t config, train_buffers_t *buffers, int seq_len);
void free_train_buffers(train_buffers_t *buffers);
void forward(config_t config, gpt2_t model, train_buffers_t buffers, const int *d_input_tokens, int seq_len);
void cross_entropy(config_t config, train_buffers_t buffers, const int *d_target_tokens, int seq_len);
void backward(config_t config, gpt2_t model, train_buffers_t buffers, gpt2_t g_model, train_buffers_t g_buffers, const int *d_input_tokens, const int *d_target_tokens, int seq_len);
void gpt2_update(gpt2_t *model, gpt2_t *grads, adamw_state_t *opt);
void gpt2_zero_grad(gpt2_t *grads);
void zero_activation_grads(config_t config, train_buffers_t *g_buffers);
