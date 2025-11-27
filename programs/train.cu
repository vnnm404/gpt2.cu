/* GPT-2 training executable - C implementation */

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
#include "gpt2/layers/softmax.h"
#include "gpt2/layers/cross_entropy.h"

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

config_t config = {
    .vocab_size = 50257,
    .batch_size = 8,
    .n_layer = 12,
    .n_head = 12,
    .n_embd = 768,
    .n_positions = 1024,
    .n_ctx = 1024};

gpt2_t model;

typedef struct
{
    tensor_t *ln_1;        // [B, S, h]
    tensor_t *ln_1_mean;   // [B, h]
    tensor_t *ln_1_rstd;   // [B, h]
    tensor_t *qkv;         // [B, S, 3h]
    tensor_t *atty;        // [B, S, h]
    tensor_t *preatt;      // [B, n_head, S, S]
    tensor_t *att;         // [B, n_head, S, S]
    tensor_t *att_proj;    // [B, S, h]
    tensor_t *res_2;       // [B, S, h]
    tensor_t *ln_2;        // [B, S, h]
    tensor_t *ln_2_mean;   // [B, h]
    tensor_t *ln_2_rstd;   // [B, h]
    tensor_t *mlp_fc;      // [B, S, 4h]
    tensor_t *mlp_fc_gelu; // [B, S, 4h]
    tensor_t *mlp_proj;    // [B, S, h]
    tensor_t *res_3;       // [B, S, h]
} layer_buffers_t;

typedef struct
{
    tensor_t *encoded;                  // [B, S, h]
    layer_buffers_t blocks[NUM_LAYERS]; // [L, ...] per-layer buffers
    tensor_t *ln_f;                     // [B, S, h]
    tensor_t *ln_f_mean;                // [B, h]
    tensor_t *ln_f_rstd;                // [B, h]
    tensor_t *logits;                   // [B, S, V]
    tensor_t *probs;                    // [B, S, V]
    tensor_t *losses;                   // [B, S]
} train_buffers_t;

train_buffers_t buffers;
train_buffers_t grad_buffers;

int prepare_input_tokens(int *input_tokens, int seq_len, int **d_input_tokens, int **d_target_tokens);
int setup_train_buffers(train_buffers_t *buffers, int seq_len);
void free_train_buffers(train_buffers_t *buffers);
void forward(const int *d_input_tokens, int seq_len);
void cross_entropy(const int *d_target_tokens, int seq_len);
void backward();
int extract_greedy_token(int seq_len, tensor_t *logits);

int main()
{
    printf("GPT-2 training\n");

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

    int input_tokens[] = {
        464, 3139, 286, 4881, 318, 464, 3139, 286, 4881, 318,
        464, 3139, 286, 4881, 318, 464, 3139, 286, 4881, 318,
        464, 3139, 286, 4881, 318, 464, 3139, 286, 4881, 318,
        464, 3139, 286, 4881, 318, 464, 3139, 286, 4881, 318,
        464, 3139, 286, 4881, 318, 464, 3139, 286, 4881, 318,
        464, 3139, 286, 4881, 318, 464, 3139, 286, 4881, 318,
        464, 3139, 286, 4881, 318, 464, 3139, 286, 4881, 318,
        464, 3139, 286, 4881, 318, 464, 3139, 286, 4881, 318,
    };
    int seq_len = 10;
    int *d_input_tokens, *d_target_tokens;
    prepare_input_tokens(input_tokens, config.batch_size * seq_len, &d_input_tokens, &d_target_tokens);
    setup_train_buffers(&buffers, seq_len);
    seq_len -= 1; // because target is shifted by 1

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    forward(d_input_tokens, seq_len);
    cross_entropy(d_target_tokens, seq_len);
    backward();

    gpuErrchk(cudaDeviceSynchronize());

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

    gpuErrchk(cudaFree(d_input_tokens));
    free_train_buffers(&buffers);
    gpt2_free(&model);
    return 0;
}

int prepare_input_tokens(int *input_tokens, int seq_len, int **d_input_tokens, int **d_target_tokens)
{
    // input_tokens: [B, S]
    // prepare input as input_tokens[:, :seq_len-1] and target as input_tokens[:, 1:seq_len]
    int size = sizeof(int) * config.batch_size * seq_len;
    int *h_input = (int *) malloc(size);
    int *h_target = (int *) malloc(size);
    for (int b = 0; b < config.batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            h_input[b * seq_len + s] = input_tokens[b * seq_len + s];
            if (s > 0) {
                h_target[b * seq_len + (s - 1)] = input_tokens[b * seq_len + s];
            }
        }
    }

    gpuErrchk(cudaMalloc((void **)d_input_tokens, (seq_len - 1) * sizeof(int)));
    gpuErrchk(cudaMemcpy(*d_input_tokens, h_input, (seq_len - 1) * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void **)d_target_tokens, (seq_len - 1) * sizeof(int)));
    gpuErrchk(cudaMemcpy(*d_target_tokens, h_target, (seq_len - 1) * sizeof(int), cudaMemcpyHostToDevice));

    free(h_input);
    free(h_target);

    return 0;
}

int setup_train_buffers(train_buffers_t *buffers, int seq_len)
{
    int B = config.batch_size;
    int S = seq_len;
    int h = config.n_embd;
    int n_head = config.n_head;
    int V = config.vocab_size;
    int four_h = 4 * h;

    // Allocate all necessary tensors
    int encoded_shape[3] = {B, S, h};
    buffers->encoded = tensor_alloc(3, encoded_shape);

    // layers
    for (int i = 0; i < config.n_layer; i++)
    {
        layer_buffers_t *layer_bufs = &buffers->blocks[i];

        int ln1_shape[3] = {B, S, h};
        int ln1_mean_shape[2] = {B, h};
        int qkv_shape[3] = {B, S, 3 * h};
        int atty_shape[3] = {B, S, h};
        int preatt_shape[4] = {B, n_head, S, S};
        int att_shape[4] = {B, n_head, S, S};
        int att_proj_shape[3] = {B, S, h};
        int res_shape[3] = {B, S, h};
        int ln2_shape[3] = {B, S, h};
        int ln2_mean_shape[2] = {B, h};
        int mlp_fc_shape[3] = {B, S, four_h};
        int mlp_fc_gelu_shape[3] = {B, S, four_h};
        int mlp_proj_shape[3] = {B, S, h};

        layer_bufs->ln_1 = tensor_alloc(3, ln1_shape);
        layer_bufs->ln_1_mean = tensor_alloc(2, ln1_mean_shape);
        layer_bufs->ln_1_rstd = tensor_alloc(2, ln1_mean_shape);
        layer_bufs->qkv = tensor_alloc(3, qkv_shape);
        layer_bufs->atty = tensor_alloc(3, atty_shape);
        layer_bufs->preatt = tensor_alloc(4, preatt_shape);
        layer_bufs->att = tensor_alloc(4, att_shape);
        layer_bufs->att_proj = tensor_alloc(3, att_proj_shape);
        layer_bufs->res_2 = tensor_alloc(3, res_shape);
        layer_bufs->ln_2 = tensor_alloc(3, ln2_shape);
        layer_bufs->ln_2_mean = tensor_alloc(2, ln2_mean_shape);
        layer_bufs->ln_2_rstd = tensor_alloc(2, ln2_mean_shape);
        layer_bufs->mlp_fc = tensor_alloc(3, mlp_fc_shape);
        layer_bufs->mlp_fc_gelu = tensor_alloc(3, mlp_fc_gelu_shape);
        layer_bufs->mlp_proj = tensor_alloc(3, mlp_proj_shape);
        layer_bufs->res_3 = tensor_alloc(3, res_shape);
    }

    int ln_f_shape[3] = {B, S, h};
    int ln_f_mean_shape[2] = {B, h};
    int logits_shape[3] = {B, S, V};
    int probs_shape[3] = {B, S, V};
    int losses_shape[2] = {B, S};

    buffers->ln_f = tensor_alloc(3, ln_f_shape);
    buffers->ln_f_mean = tensor_alloc(2, ln_f_mean_shape);
    buffers->ln_f_rstd = tensor_alloc(2, ln_f_mean_shape);
    buffers->logits = tensor_alloc(3, logits_shape);
    buffers->probs = tensor_alloc(3, probs_shape);
    buffers->losses = tensor_alloc(2, losses_shape);

    return 0;
}

void free_train_buffers(train_buffers_t *buffers)
{
    tensor_free(buffers->encoded);
    for (int i = 0; i < config.n_layer; i++) {
        layer_buffers_t *layer_bufs = &buffers->blocks[i];

        tensor_free(layer_bufs->ln_1);
        tensor_free(layer_bufs->ln_1_mean);
        tensor_free(layer_bufs->ln_1_rstd);
        tensor_free(layer_bufs->qkv);
        tensor_free(layer_bufs->atty);
        tensor_free(layer_bufs->preatt);
        tensor_free(layer_bufs->att);
        tensor_free(layer_bufs->att_proj);
        tensor_free(layer_bufs->res_2);
        tensor_free(layer_bufs->ln_2);
        tensor_free(layer_bufs->ln_2_mean);
        tensor_free(layer_bufs->ln_2_rstd);
        tensor_free(layer_bufs->mlp_fc);
        tensor_free(layer_bufs->mlp_fc_gelu);
        tensor_free(layer_bufs->mlp_proj);
        tensor_free(layer_bufs->res_3);
    }
    tensor_free(buffers->ln_f);
    tensor_free(buffers->ln_f_mean);
    tensor_free(buffers->ln_f_rstd);
    tensor_free(buffers->logits);
    tensor_free(buffers->probs);
    tensor_free(buffers->losses);
}

void forward(const int *d_input_tokens, int seq_len)
{
    int L = config.n_layer;
    int B = config.batch_size;
    int S = seq_len;
    int h = config.n_embd;
    int n_head = config.n_head;
    int V = config.vocab_size;

    int thr = 256;

    embedding_forward<<<B, thr>>>(buffers.encoded->data, d_input_tokens, model.emb.wte->data, model.emb.wpe->data, S, h, V, config.n_positions);

    // layers
    for (int layer_idx = 0; layer_idx < config.n_layer; layer_idx++)
    {
        block_t *block = &model.h[layer_idx];
        layer_buffers_t *layer_bufs = &buffers.blocks[layer_idx];

        tensor_t *res = (layer_idx == 0) ? buffers.encoded : buffers.blocks[layer_idx - 1].res_3;

        layernorm_forward<<<B, thr>>>(layer_bufs->ln_1->data, res->data, block->ln_1.w->data, block->ln_1.b->data, layer_bufs->ln_1_mean->data, layer_bufs->ln_1_rstd->data, S, h);

        mlp_forward<<<CEIL_DIV(B * S * 3 * h, thr), thr>>>(layer_bufs->qkv->data, layer_bufs->ln_1->data, block->attn.qkv_w->data, block->attn.qkv_b->data, B, S, h, h * 3);
        gpuErrchk(cudaGetLastError());

        attention_forward<<<CEIL_DIV(B * S * n_head, thr), thr>>>(layer_bufs->atty->data, layer_bufs->preatt->data, layer_bufs->att->data, layer_bufs->qkv->data, B, S, n_head, h);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        mlp_forward<<<CEIL_DIV(B * S * h, thr), thr>>>(layer_bufs->att_proj->data, layer_bufs->atty->data, block->attn.proj_w->data, block->attn.proj_b->data, B, S, h, h);
        residual_forward<<<CEIL_DIV(B * S * h, thr), thr>>>(layer_bufs->res_2->data, layer_bufs->att_proj->data, res->data, B, S, h);

        layernorm_forward<<<B, thr>>>(layer_bufs->ln_2->data, layer_bufs->res_2->data, block->ln_2.w->data, block->ln_2.b->data, layer_bufs->ln_2_mean->data, layer_bufs->ln_2_rstd->data, S, h);

        mlp_forward<<<CEIL_DIV(B * S * 4 * h, thr), thr>>>(layer_bufs->mlp_fc->data, layer_bufs->ln_2->data, block->mlp.fc_w->data, block->mlp.fc_b->data, B, S, h, h * 4);

        gelu_forward<<<CEIL_DIV(B * S * 4 * h, thr), thr>>>(layer_bufs->mlp_fc_gelu->data, layer_bufs->mlp_fc->data, B, S, h * 4);

        mlp_forward<<<CEIL_DIV(B * S * h, thr), thr>>>(layer_bufs->mlp_proj->data, layer_bufs->mlp_fc_gelu->data, block->mlp.proj_w->data, block->mlp.proj_b->data, B, S, h * 4, h);
        residual_forward<<<CEIL_DIV(B * S * h, thr), thr>>>(layer_bufs->res_3->data, layer_bufs->mlp_proj->data, layer_bufs->res_2->data, B, S, h);
    }

    tensor_t *res = buffers.blocks[L - 1].res_3;
    layernorm_forward<<<B, thr>>>(buffers.ln_f->data, res->data, model.ln_f.w->data, model.ln_f.b->data, buffers.ln_f_mean->data, buffers.ln_f_rstd->data, S, h);

    mlp_forward<<<CEIL_DIV(B * S * V, thr), thr>>>(buffers.logits->data, buffers.ln_f->data, model.emb.wte->data, NULL, B, S, h, V);

    softmax_forward<<<CEIL_DIV(B * S * V, thr), thr>>>(buffers.probs->data, buffers.logits->data, B, S, V);
}

void cross_entropy(const int *d_target_tokens, int seq_len) {
    int B = config.batch_size;
    int S = seq_len;
    int V = config.vocab_size;

    int thr = 256;
    cross_entropy_forward<<<CEIL_DIV(B * S, thr), thr>>>(buffers.losses->data, buffers.probs->data, d_target_tokens, B, S, V);
}

void backward() {
    // TODO: store means and rstd in the layernorm fwd
}

int extract_greedy_token(int seq_len, tensor_t *logits) {
    // logits: [B, S, V]

    int size = sizeof(float) * config.vocab_size;
    float *h_logits = (float *) malloc(size);
    // int skip = seq_len * config.vocab_size;
    cudaMemcpy(h_logits, logits->data + (seq_len - 1) * logits->shape[2], size, cudaMemcpyDeviceToHost);

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