/* GPT-2 training test - validates forward pass, backward pass, and optimizer updates */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
// C++ containers used for top-k computation
#include <vector>
#include <queue>
#include <functional>
#include <algorithm>

#include "gpt2/gpt2.h"
#include "gpt2/layers/embedding.h"
#include "gpt2/layers/layernorm.h"
#include "gpt2/layers/mlp.h"
#include "gpt2/layers/attention.h"
#include "gpt2/layers/residual.h"
#include "gpt2/layers/gelu.h"
#include "gpt2/layers/softmax.h"
#include "gpt2/layers/cross_entropy.h"
#include "gpt2/layers/adamw.h"

#include "../src/layers/embedding.cu"
#include "../src/layers/layernorm.cu"
#include "../src/layers/mlp.cu"
#include "../src/layers/attention.cu"
#include "../src/layers/residual.cu"
#include "../src/layers/gelu.cu"
#include "../src/layers/softmax.cu"
#include "../src/layers/cross_entropy.cu"
#include "../src/layers/adamw.cu"


#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define NUM_SM 28

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

// Configuration matching the test expectations
config_t config = {
    .vocab_size = 50257,
    .batch_size = 4,  // B = 4 from test
    .n_layer = 12,
    .n_head = 12,
    .n_embd = 768,
    .n_positions = 1024,
    .n_ctx = 1024
};

// Model structures
gpt2_t model;
gpt2_t g_model;  // model weight gradients

// Device copies for kernel
gpt2_t *d_model;
gpt2_t *d_g_model;

// Training buffer structures (same as train.cu)
typedef struct
{
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

typedef struct
{
    tensor_t encoded;
    layer_buffers_t blocks[NUM_LAYERS];
    tensor_t ln_f;
    tensor_t ln_f_mean;
    tensor_t ln_f_rstd;
    tensor_t logits;
    tensor_t probs;
    tensor_t losses;
} train_buffers_t;

train_buffers_t buffers;
train_buffers_t g_buffers;

// Device copies for kernel
train_buffers_t *d_buffers;
train_buffers_t *d_g_buffers;

// AdamW optimizer state
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

adamw_state_t opt_state;

// megakernel, gpu memory
typedef struct {
    int op;
    int prev_op;
    int layer;
    int b_x, b_y;

    int bar_idx;
    int expected;
    bool inc;
} instruction_t;

typedef struct {
    int n;
    instruction_t *instructions;
} stream_t;

int *bar;  // [B, 1 + (L * 10 + 3) + 1 + (5 + L * 13 + 1)] global atomics
stream_t *streams[NUM_SM];  // Host streams
stream_t **d_streams;  // Device streams array

// Function prototypes
int setup_train_buffers(train_buffers_t *buffers, int seq_len);
void free_train_buffers(train_buffers_t *buffers);
void forward(const int *d_input_tokens, int seq_len);
void cross_entropy(const int *d_target_tokens, int seq_len);
void backward(const int *d_input_tokens, const int *d_target_tokens, int seq_len);
void gpt2_update(gpt2_t *model, gpt2_t *grads, adamw_state_t *opt);
void gpt2_zero_grad(gpt2_t *grads);
void zero_activation_grads(train_buffers_t *g_buffers);
stream_t** schedule_instructions(int seq_len);
void free_schedule(stream_t **d_streams);

__global__ void megakernel(
    config_t config,

    gpt2_t *model,
    gpt2_t *g_model,

    train_buffers_t *buffers,
    train_buffers_t *g_buffers,

    adamw_state_t opt_state,

    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,

    int *bar,
    stream_t **streams
);

__device__ void execute_stream(
    config_t config,

    gpt2_t *model,
    gpt2_t *g_model,

    train_buffers_t *buffers,
    train_buffers_t *g_buffers,

    adamw_state_t opt_state,

    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,

    int *bar,
    stream_t *stream
);

__device__ void execute_instruction(
    config_t config,

    gpt2_t *model,
    gpt2_t *g_model,

    train_buffers_t *buffers,
    train_buffers_t *g_buffers,

    adamw_state_t opt_state,

    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,

    int *bar,
    instruction_t instr
);

// Helper: compute mean loss
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

int main(int argc, char *argv[]) {
    printf("GPT-2 MK Training Test\n");

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s\nmaxThreadsPerBlock=%d\nsharedMemPerBlock=%zu\nregsPerBlock=%d\nmultiProcessorCount=%d\n",
        p.name, p.maxThreadsPerBlock, p.sharedMemPerBlock, p.regsPerBlock, p.multiProcessorCount);
    
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, megakernel);
    printf("megakernel: numRegs=%d, sharedSizeBytes=%zu, maxThreadsPerBlock=%d\n",
       attr.numRegs, attr.sharedSizeBytes, attr.maxThreadsPerBlock);
    
    // Initialize models
    if (gpt2_initialize(&model, &config) != 0) {
        fprintf(stderr, "Failed to initialize GPT-2 model\n");
        return 1;
    }
    
    if (gpt2_initialize(&g_model, &config) != 0) {
        fprintf(stderr, "Failed to initialize GPT-2 gradient model\n");
        gpt2_free(&model);
        return 1;
    }
    
    // Load model weights
    FILE *model_file = fopen("../models/gpt2-124M-weights.bin", "rb");
    if (model_file == NULL) {
        fprintf(stderr, "Error opening model file\n");
        gpt2_free(&model);
        gpt2_free(&g_model);
        return 1;
    }
    
    if (gpt2_load_weights(&model, model_file) != 0) {
        fprintf(stderr, "Failed to load GPT-2 weights\n");
        fclose(model_file);
        gpt2_free(&model);
        gpt2_free(&g_model);
        return 1;
    }
    fclose(model_file);
    
    printf("Model loaded successfully.\n");
    
    // Load debug state file
    FILE *state_file = fopen("../models/gpt2_124M_debug_state.bin", "rb");
    if (state_file == NULL) {
        fprintf(stderr, "Error opening state file\n");
        gpt2_free(&model);
        gpt2_free(&g_model);
        return 1;
    }
    
    int state_header[256];
    size_t items_read = fread(state_header, sizeof(int), 256, state_file);
    if (items_read != 256) {
        fprintf(stderr, "Failed to read state header\n");
        fclose(state_file);
        gpt2_free(&model);
        gpt2_free(&g_model);
        return 1;
    }
    if (state_header[0] != 20240327) {
        fprintf(stderr, "Bad magic state file\n");
        fclose(state_file);
        gpt2_free(&model);
        gpt2_free(&g_model);
        return 1;
    }
    // if (state_header[1] != 1) {
    //     fprintf(stderr, "Bad version in state file\n");
    //     fclose(state_file);
    //     gpt2_free(&model);
    //     gpt2_free(&g_model);
    //     return 1;
    // }
    printf("Version: %d\n", state_header[1]);
    
    int B = state_header[2];
    int T = state_header[3];
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);
    
    // Verify batch size matches
    if (B != config.batch_size) {
        fprintf(stderr, "Batch size mismatch: config=%d, state=%d\n", config.batch_size, B);
        fclose(state_file);
        gpt2_free(&model);
        gpt2_free(&g_model);
        return 1;
    }
    
    int V = config.vocab_size;
    
    // Allocate CPU memory for inputs
    int *x = (int *)malloc(B * T * sizeof(int));
    int *y = (int *)malloc(B * T * sizeof(int));
    
    // Read inputs from state file (skip logits, loss, and gradients)
    items_read = fread(x, sizeof(int), B * T, state_file);
    if (items_read != B * T) {
        fprintf(stderr, "Failed to read input x\n");
        fclose(state_file);
        free(x);
        free(y);
        gpt2_free(&model);
        gpt2_free(&g_model);
        return 1;
    }
    items_read = fread(y, sizeof(int), B * T, state_file);
    if (items_read != B * T) {
        fprintf(stderr, "Failed to read input y\n");
        fclose(state_file);
        free(x);
        free(y);
        gpt2_free(&model);
        gpt2_free(&g_model);
        return 1;
    }
    
    // Skip expected_logits, expected_loss, and expected_grads
    fseek(state_file, B * T * V * sizeof(float), SEEK_CUR);  // skip logits
    fseek(state_file, sizeof(float), SEEK_CUR);              // skip loss
    fseek(state_file, model.num_parameters * sizeof(float), SEEK_CUR);  // skip grads
    
    fclose(state_file);
    
    // Copy inputs to GPU
    int *d_input_tokens, *d_target_tokens;
    gpuErrchk(cudaMalloc(&d_input_tokens, B * T * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_target_tokens, B * T * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_input_tokens, x, B * T * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_target_tokens, y, B * T * sizeof(int), cudaMemcpyHostToDevice));
    
    // Setup training buffers
    d_streams = schedule_instructions(T);
    setup_train_buffers(&buffers, T);
    setup_train_buffers(&g_buffers, T);

    // Allocate device memory for structs
    gpuErrchk(cudaMalloc(&d_model, sizeof(gpt2_t)));
    gpuErrchk(cudaMalloc(&d_g_model, sizeof(gpt2_t)));
    gpuErrchk(cudaMalloc(&d_buffers, sizeof(train_buffers_t)));
    gpuErrchk(cudaMalloc(&d_g_buffers, sizeof(train_buffers_t)));
    
    // Copy structs to device
    gpuErrchk(cudaMemcpy(d_model, &model, sizeof(gpt2_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_g_model, &g_model, sizeof(gpt2_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_buffers, &buffers, sizeof(train_buffers_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_g_buffers, &g_buffers, sizeof(train_buffers_t), cudaMemcpyHostToDevice));

    // allocate global bar
    int bar_size = config.batch_size * (1 + (config.n_layer * 10 + 3) + 1 + (5 + config.n_layer * 14 + 1));
    gpuErrchk(cudaMalloc(&bar, bar_size * sizeof(int)));

    int shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    int threads_per_block = 1024;
    printf("Shared memory size per block: %d bytes\n", shared_mem_size);
    
    // Initialize optimizer state
    opt_state.learning_rate = 1e-4f;
    opt_state.beta1 = 0.9f;
    opt_state.beta2 = 0.999f;
    opt_state.eps = 1e-8f;
    opt_state.weight_decay = 0.01f;
    opt_state.t = 0;
    opt_state.m_memory = NULL;
    opt_state.v_memory = NULL;
    
    float losses[10];
    
    // Run 10 training iterations
    for (int step = 0; step < 10; step++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Zero gradients
        gpt2_zero_grad(&g_model);
        zero_activation_grads(&g_buffers);
        gpuErrchk(cudaMemset(bar, 0, bar_size * sizeof(int)));
        
        // forward(d_input_tokens, T);
        // cross_entropy(d_target_tokens, T);
        // backward(d_input_tokens, d_target_tokens, T);

        megakernel<<<NUM_SM, threads_per_block, shared_mem_size>>>(
            config,
            d_model,
            d_g_model,
            d_buffers,
            d_g_buffers,
            opt_state,
            T,
            d_input_tokens,
            d_target_tokens,
            bar,
            d_streams
        );
        gpuErrchk(cudaGetLastError());
        // gpuErrchk(cudaDeviceSynchronize());

        float mean_loss = compute_mean_loss(&buffers.losses, B, T);
        
        // Update parameters
        gpt2_update(&model, &g_model, &opt_state);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        printf("step %d: loss %f (took %f ms)\n", step, mean_loss, time_elapsed_s * 1000);
        losses[step] = mean_loss;
    }
    
    // Check expected losses
    float expected_losses[10] = {
        5.270007133483887,
        4.059706687927246,
        3.3751230239868164,
        2.8007826805114746,
        2.315382242202759,
        1.8490285873413086,
        1.3946564197540283,
        0.9991465210914612,
        0.6240804195404053,
        0.37651097774505615
    };
    
    int allok = 1;
    for (int i = 0; i < 10; i++) {
        if (fabs(losses[i] - expected_losses[i]) >= 1e-2) {
            printf("LOSS MISMATCH AT STEP %d: %f %f\n", i, losses[i], expected_losses[i]);
            allok = 0;
        } else {
            printf("loss ok at step %d: %f %f\n", i, losses[i], expected_losses[i]);
        }
    }
    
    printf("overall okay: %d\n", allok);
    
    // Cleanup
    free_schedule(d_streams);
    free(x);
    free(y);
    cudaFree(d_input_tokens);
    cudaFree(d_target_tokens);
    free_train_buffers(&buffers);
    free_train_buffers(&g_buffers);
    if (opt_state.m_memory) cudaFree(opt_state.m_memory);
    if (opt_state.v_memory) cudaFree(opt_state.v_memory);
    gpt2_free(&model);
    gpt2_free(&g_model);
    
    return allok ? 0 : 1;
}

// Implementation of helper functions from train.cu

int setup_train_buffers(train_buffers_t *buffers, int seq_len)
{
    int B = config.batch_size;
    int S = seq_len;
    int h = config.n_embd;
    int n_head = config.n_head;
    int V = config.vocab_size;
    int four_h = 4 * h;

    int encoded_shape[3] = {B, S, h};
    buffers->encoded = tensor_alloc(3, encoded_shape);

    for (int i = 0; i < config.n_layer; i++)
    {
        layer_buffers_t *layer_bufs = &buffers->blocks[i];

        int ln1_shape[3] = {B, S, h};
        int ln1_mean_shape[2] = {B, S};
        int qkv_shape[3] = {B, S, 3 * h};
        int atty_shape[3] = {B, S, h};
        int preatt_shape[4] = {B, n_head, S, S};
        int att_shape[4] = {B, n_head, S, S};
        int att_proj_shape[3] = {B, S, h};
        int res_shape[3] = {B, S, h};
        int ln2_shape[3] = {B, S, h};
        int ln2_mean_shape[2] = {B, S};
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
    int ln_f_mean_shape[2] = {B, S};
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
    tensor_free(&buffers->encoded);
    for (int i = 0; i < config.n_layer; i++) {
        layer_buffers_t *layer_bufs = &buffers->blocks[i];

        tensor_free(&layer_bufs->ln_1);
        tensor_free(&layer_bufs->ln_1_mean);
        tensor_free(&layer_bufs->ln_1_rstd);
        tensor_free(&layer_bufs->qkv);
        tensor_free(&layer_bufs->atty);
        tensor_free(&layer_bufs->preatt);
        tensor_free(&layer_bufs->att);
        tensor_free(&layer_bufs->att_proj);
        tensor_free(&layer_bufs->res_2);
        tensor_free(&layer_bufs->ln_2);
        tensor_free(&layer_bufs->ln_2_mean);
        tensor_free(&layer_bufs->ln_2_rstd);
        tensor_free(&layer_bufs->mlp_fc);
        tensor_free(&layer_bufs->mlp_fc_gelu);
        tensor_free(&layer_bufs->mlp_proj);
        tensor_free(&layer_bufs->res_3);
    }
    tensor_free(&buffers->ln_f);
    tensor_free(&buffers->ln_f_mean);
    tensor_free(&buffers->ln_f_rstd);
    tensor_free(&buffers->logits);
    tensor_free(&buffers->probs);
    tensor_free(&buffers->losses);
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

void cross_entropy(const int *d_target_tokens, int seq_len) {
    int B = config.batch_size;
    int S = seq_len;
    int V = config.vocab_size;

    int thr = 256;
    cross_entropy_forward<<<CEIL_DIV(B * S, thr), thr>>>(buffers.losses.data, buffers.probs.data, d_target_tokens, B, S, V);
}

void backward(const int *d_input_tokens, const int *d_target_tokens, int seq_len) {
    int L = config.n_layer;
    int B = config.batch_size;
    int S = seq_len;
    int h = config.n_embd;
    int n_head = config.n_head;
    int V = config.vocab_size;

    int thr = 256;

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

    // printf("Updating parameters with AdamW: num_parameters=%d, num_blocks=%d, threads_per_block=%d\n", model->num_parameters, num_blocks, thr);
    
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

void gpt2_zero_grad(gpt2_t *grads) {
    cudaMemset(grads->params_memory, 0, grads->num_parameters * sizeof(float));
}

void zero_activation_grads(train_buffers_t *g_buffers) {
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

stream_t** schedule_instructions(int seq_len) {
    int L = config.n_layer;
    int B = config.batch_size;
    int S = seq_len;
    int h = config.n_embd;
    int n_head = config.n_head;
    int V = config.vocab_size;
    int thr = 1024;
 
    // Initialize streams for each SM
    for (int sm = 0; sm < NUM_SM; sm++) {
        streams[sm] = (stream_t *)malloc(sizeof(stream_t));
        streams[sm]->n = 0;
        streams[sm]->instructions = NULL;
    }
 
    // Temporary storage for all instructions before distributing to SMs
    int max_instructions = 400000; // Conservative estimate
    instruction_t *all_instructions = (instruction_t *)malloc(max_instructions * sizeof(instruction_t));
    int instruction_count = 0;
 
    int prev_op = 0;
 
    // OP 1: Embedding forward - [B blocks, 1D grid]
    {
        int op = 1;
        int layer = -1;
        int bar_idx = -1;
        int expected = 0;
        int num_blocks = B;
 
        for (int block = 0; block < num_blocks; block++) {
            all_instructions[instruction_count++] = (instruction_t){
                .op = op,
                .prev_op = prev_op,
                .layer = layer,
                .b_x = block,
                .b_y = -1,
                .bar_idx = bar_idx,
                .expected = expected
            };
        }
        prev_op = op;
    }
 
    // Forward pass through layers
    for (int layer_idx = 0; layer_idx < L; layer_idx++) {
        // OP 2: LayerNorm 1 - [B blocks, 1D grid]
        {
            int op = 2;
            int bar_idx = (layer_idx == 0) ? 0 : (1 + (layer_idx - 1) * 10 + 9);
            int expected = (layer_idx == 0) ? B : CEIL_DIV(B * S * h, thr);
            int num_blocks = B;
 
            for (int block = 0; block < num_blocks; block++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = layer_idx,
                    .b_x = block,
                    .b_y = -1,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
            prev_op = op;
        }
 
        // OP 3: QKV - [MLP_FORWARD_GRID(h * 3, B, S) blocks, 2D grid]
        {
            int op = 3;
            int bar_idx = 1 + layer_idx * 10;
            int expected = B;
            dim3 grid = MLP_FORWARD_GRID(h * 3, B, S);
            int num_blocks_x = grid.x;
            int num_blocks_y = grid.y;
 
            for (int by = 0; by < num_blocks_y; by++) {
                for (int bx = 0; bx < num_blocks_x; bx++) {
                    all_instructions[instruction_count++] = (instruction_t){
                        .op = op,
                        .prev_op = prev_op,
                        .layer = layer_idx,
                        .b_x = bx,
                        .b_y = by,
                        .bar_idx = bar_idx,
                        .expected = expected
                    };
                }
            }
            prev_op = op;
        }
 
        // OP 4: Attention - [CEIL_DIV(B * S * n_head, thr) blocks, 1D grid]
        {
            int op = 4;
            int bar_idx = 1 + layer_idx * 10 + 1;
            dim3 grid = MLP_FORWARD_GRID(h * 3, B, S);
            int expected = grid.x * grid.y;
            int num_blocks = CEIL_DIV(B * S * n_head, thr);
 
            for (int block = 0; block < num_blocks; block++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = layer_idx,
                    .b_x = block,
                    .b_y = -1,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
            prev_op = op;
        }
 
        // OP 5: Attention projection - [MLP_FORWARD_GRID(h, B, S) blocks, 2D grid]
        {
            int op = 5;
            int bar_idx = 1 + layer_idx * 10 + 2;
            int expected = CEIL_DIV(B * S * n_head, thr);
            dim3 grid = MLP_FORWARD_GRID(h, B, S);
            int num_blocks_x = grid.x;
            int num_blocks_y = grid.y;
 
            for (int by = 0; by < num_blocks_y; by++) {
                for (int bx = 0; bx < num_blocks_x; bx++) {
                    all_instructions[instruction_count++] = (instruction_t){
                        .op = op,
                        .prev_op = prev_op,
                        .layer = layer_idx,
                        .b_x = bx,
                        .b_y = by,
                        .bar_idx = bar_idx,
                        .expected = expected
                    };
                }
            }
            prev_op = op;
        }
 
        // OP 6: Residual 2 - [CEIL_DIV(B * S * h, thr) blocks, 1D grid]
        {
            int op = 6;
            int bar_idx = 1 + layer_idx * 10 + 3;
            dim3 grid = MLP_FORWARD_GRID(h, B, S);
            int expected = grid.x * grid.y;
            int num_blocks = CEIL_DIV(B * S * h, thr);
 
            for (int block = 0; block < num_blocks; block++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = layer_idx,
                    .b_x = block,
                    .b_y = -1,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
            prev_op = op;
        }
 
        // OP 7: LayerNorm 2 - [B blocks, 1D grid]
        {
            int op = 7;
            int bar_idx = 1 + layer_idx * 10 + 4;
            int expected = CEIL_DIV(B * S * h, thr);
            int num_blocks = B;
 
            for (int block = 0; block < num_blocks; block++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = layer_idx,
                    .b_x = block,
                    .b_y = -1,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
            prev_op = op;
        }
 
        // OP 8: MLP FC - [MLP_FORWARD_GRID(h * 4, B, S) blocks, 2D grid]
        {
            int op = 8;
            int bar_idx = 1 + layer_idx * 10 + 5;
            int expected = B;
            dim3 grid = MLP_FORWARD_GRID(h * 4, B, S);
            int num_blocks_x = grid.x;
            int num_blocks_y = grid.y;
 
            for (int by = 0; by < num_blocks_y; by++) {
                for (int bx = 0; bx < num_blocks_x; bx++) {
                    all_instructions[instruction_count++] = (instruction_t){
                        .op = op,
                        .prev_op = prev_op,
                        .layer = layer_idx,
                        .b_x = bx,
                        .b_y = by,
                        .bar_idx = bar_idx,
                        .expected = expected
                    };
                }
            }
            prev_op = op;
        }
 
        // OP 9: GELU - [CEIL_DIV(B * S * 4 * h, thr) blocks, 1D grid]
        {
            int op = 9;
            int bar_idx = 1 + layer_idx * 10 + 6;
            dim3 grid = MLP_FORWARD_GRID(h * 4, B, S);
            int expected = grid.x * grid.y;
            int num_blocks = CEIL_DIV(B * S * 4 * h, thr);
 
            for (int block = 0; block < num_blocks; block++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = layer_idx,
                    .b_x = block,
                    .b_y = -1,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
            prev_op = op;
        }
 
        // OP 10: MLP projection - [MLP_FORWARD_GRID(h, B, S) blocks, 2D grid]
        {
            int op = 10;
            int bar_idx = 1 + layer_idx * 10 + 7;
            int expected = CEIL_DIV(B * S * 4 * h, thr);
            dim3 grid = MLP_FORWARD_GRID(h, B, S);
            int num_blocks_x = grid.x;
            int num_blocks_y = grid.y;
 
            for (int by = 0; by < num_blocks_y; by++) {
                for (int bx = 0; bx < num_blocks_x; bx++) {
                    all_instructions[instruction_count++] = (instruction_t){
                        .op = op,
                        .prev_op = prev_op,
                        .layer = layer_idx,
                        .b_x = bx,
                        .b_y = by,
                        .bar_idx = bar_idx,
                        .expected = expected
                    };
                }
            }
            prev_op = op;
        }
 
        // OP 11: Residual 3 - [CEIL_DIV(B * S * h, thr) blocks, 1D grid]
        {
            int op = 11;
            int bar_idx = 1 + layer_idx * 10 + 8;
            dim3 grid = MLP_FORWARD_GRID(h, B, S);
            int expected = grid.x * grid.y;
            int num_blocks = CEIL_DIV(B * S * h, thr);
 
            for (int block = 0; block < num_blocks; block++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = layer_idx,
                    .b_x = block,
                    .b_y = -1,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
            prev_op = op;
        }
    }
 
    // OP 12: Final LayerNorm - [B blocks, 1D grid]
    {
        int op = 12;
        int bar_idx = 1 + (L - 1) * 10 + 9;
        int expected = CEIL_DIV(B * S * h, thr);
        int num_blocks = B;
 
        for (int block = 0; block < num_blocks; block++) {
            all_instructions[instruction_count++] = (instruction_t){
                .op = op,
                .prev_op = prev_op,
                .layer = -1,
                .b_x = block,
                .b_y = -1,
                .bar_idx = bar_idx,
                .expected = expected
            };
        }
        prev_op = op;
    }
 
    // OP 13: Logits - [MLP_FORWARD_GRID(V, B, S) blocks, 2D grid]
    {
        int op = 13;
        int bar_idx = 1 + (L * 10) + 0;
        int expected = B;
        dim3 grid = MLP_FORWARD_GRID(V, B, S);
        int num_blocks_x = grid.x;
        int num_blocks_y = grid.y;
 
        for (int by = 0; by < num_blocks_y; by++) {
            for (int bx = 0; bx < num_blocks_x; bx++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = -1,
                    .b_x = bx,
                    .b_y = by,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
        }
        prev_op = op;
    }
 
    // OP 14: Softmax - [CEIL_DIV(B * S * V, thr) blocks, 1D grid]
    {
        int op = 14;
        int bar_idx = 1 + (L * 10) + 1;
        dim3 grid = MLP_FORWARD_GRID(V, B, S);
        int expected = grid.x * grid.y;
        int num_blocks = CEIL_DIV(B * S * V, thr);
 
        for (int block = 0; block < num_blocks; block++) {
            all_instructions[instruction_count++] = (instruction_t){
                .op = op,
                .prev_op = prev_op,
                .layer = -1,
                .b_x = block,
                .b_y = -1,
                .bar_idx = bar_idx,
                .expected = expected
            };
        }
        prev_op = op;
    }
 
    // OP 15: Cross-entropy - [CEIL_DIV(B * S, thr) blocks, 1D grid]
    {
        int op = 15;
        int bar_idx = 1 + (L * 10) + 2;
        int expected = CEIL_DIV(B * S * V, thr);
        int num_blocks = CEIL_DIV(B * S, thr);
 
        for (int block = 0; block < num_blocks; block++) {
            all_instructions[instruction_count++] = (instruction_t){
                .op = op,
                .prev_op = prev_op,
                .layer = -1,
                .b_x = block,
                .b_y = -1,
                .bar_idx = bar_idx,
                .expected = expected
            };
        }
        prev_op = op;
    }
 
    printf("Forward pass instructions scheduled: %d\n", instruction_count);
 
    // OP 16: Cross-entropy backward - [CEIL_DIV(B * S, thr) blocks, 1D grid]
    {
        int op = 16;
        int bar_idx = 1 + (L * 10) + 3;
        int expected = CEIL_DIV(B * S, thr);
        int num_blocks = CEIL_DIV(B * S, thr);
 
        for (int block = 0; block < num_blocks; block++) {
            all_instructions[instruction_count++] = (instruction_t){
                .op = op,
                .prev_op = prev_op,
                .layer = -1,
                .b_x = block,
                .b_y = -1,
                .bar_idx = bar_idx,
                .expected = expected
            };
        }
        prev_op = op;
    }
 
    // OP 17: Logits backward - [MLP_BACKWARD_INPUT_GRID(h, B, S) blocks, 2D grid]
    {
        int op = 17;
        int bar_idx = 1 + (L * 10) + 3 + 1;
        int expected = CEIL_DIV(B * S, thr);
        dim3 grid = MLP_BACKWARD_INPUT_GRID(h, B, S);
        int num_blocks_x = grid.x;
        int num_blocks_y = grid.y;
 
        for (int by = 0; by < num_blocks_y; by++) {
            for (int bx = 0; bx < num_blocks_x; bx++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = -1,
                    .b_x = bx,
                    .b_y = by,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
        }
        prev_op = op;
    }
 
    // OP 18: Embedding weight gradient - [MLP_BACKWARD_WEIGHT_GRID(V, h) blocks, 2D grid]
    {
        int op = 18;
        int bar_idx = 1 + (L * 10) + 3 + 2;
        dim3 grid_prev = MLP_BACKWARD_INPUT_GRID(h, B, S);
        int expected = grid_prev.x * grid_prev.y;
        dim3 grid = MLP_BACKWARD_WEIGHT_GRID(V, h);
        int num_blocks_x = grid.x;
        int num_blocks_y = grid.y;
 
        for (int by = 0; by < num_blocks_y; by++) {
            for (int bx = 0; bx < num_blocks_x; bx++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = -1,
                    .b_x = bx,
                    .b_y = by,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
        }
        prev_op = op;
    }
 
    // OP 19: Final LayerNorm backward - [B blocks, 1D grid]
    {
        int op = 19;
        int bar_idx = 1 + (L * 10) + 3 + 3;
        dim3 grid_prev = MLP_BACKWARD_WEIGHT_GRID(V, h);
        int expected = grid_prev.x * grid_prev.y;
        int num_blocks = B;
 
        for (int block = 0; block < num_blocks; block++) {
            all_instructions[instruction_count++] = (instruction_t){
                .op = op,
                .prev_op = prev_op,
                .layer = -1,
                .b_x = block,
                .b_y = -1,
                .bar_idx = bar_idx,
                .expected = expected
            };
        }
        prev_op = op;
    }
 
    // Backward pass through layers
    for (int layer_idx = L - 1; layer_idx >=0; layer_idx--) {
        // OP 20: Residual backward (res_3) - [CEIL_DIV(B * S * h, thr) blocks, 1D grid]
        {
            int op = 20;
            int bar_idx = (layer_idx == L - 1) ? (1 + (L * 10) + 3 + 4) : (1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) - 1) * 14 + 13);
            int expected = B;
            int num_blocks = CEIL_DIV(B * S * h, thr);
 
            for (int block = 0; block < num_blocks; block++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = layer_idx,
                    .b_x = block,
                    .b_y = -1,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
            prev_op = op;
        }
 
        // OP 21: MLP projection backward input - [MLP_BACKWARD_INPUT_GRID(h * 4, B, S) blocks, 2D grid]
        {
            int op = 21;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14);
            int expected = CEIL_DIV(B * S * h, thr);
            dim3 grid = MLP_BACKWARD_INPUT_GRID(h * 4, B, S);
            int num_blocks_x = grid.x;
            int num_blocks_y = grid.y;
 
            for (int by = 0; by < num_blocks_y; by++) {
                for (int bx = 0; bx < num_blocks_x; bx++) {
                    all_instructions[instruction_count++] = (instruction_t){
                        .op = op,
                        .prev_op = prev_op,
                        .layer = layer_idx,
                        .b_x = bx,
                        .b_y = by,
                        .bar_idx = bar_idx,
                        .expected = expected
                    };
                }
            }
            prev_op = op;
        }
 
 
        // OP 22: MLP projection backward weight - [MLP_BACKWARD_WEIGHT_GRID(h, h * 4) blocks, 2D grid]
        {
            int op = 22;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 1;
            dim3 grid_prev = MLP_BACKWARD_INPUT_GRID(h * 4, B, S);
            int expected = grid_prev.x * grid_prev.y;
            dim3 grid = MLP_BACKWARD_WEIGHT_GRID(h, h * 4);
            int num_blocks_x = grid.x;
            int num_blocks_y = grid.y;
 
            for (int by = 0; by < num_blocks_y; by++) {
                for (int bx = 0; bx < num_blocks_x; bx++) {
                    all_instructions[instruction_count++] = (instruction_t){
                        .op = op,
                        .prev_op = prev_op,
                        .layer = layer_idx,
                        .b_x = bx,
                        .b_y = by,
                        .bar_idx = bar_idx,
                        .expected = expected
                    };
                }
            }
            prev_op = op;
        }
// if(layer_idx == L-2)break;
 
 
        // OP 23: GELU backward - [CEIL_DIV(B * S * 4 * h, thr) blocks, 1D grid]
        {
            int op = 23;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 2;
            dim3 grid_prev = MLP_BACKWARD_WEIGHT_GRID(h, h * 4);
            int expected = grid_prev.x * grid_prev.y;
            int num_blocks = CEIL_DIV(B * S * 4 * h, thr);
 
            for (int block = 0; block < num_blocks; block++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = layer_idx,
                    .b_x = block,
                    .b_y = -1,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
            prev_op = op;
        }
 
        // OP 24: MLP FC backward input - [MLP_BACKWARD_INPUT_GRID(h, B, S) blocks, 2D grid]
        {
            int op = 24;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 3;
            int expected = CEIL_DIV(B * S * 4 * h, thr);
            dim3 grid = MLP_BACKWARD_INPUT_GRID(h, B, S);
            int num_blocks_x = grid.x;
            int num_blocks_y = grid.y;
 
            for (int by = 0; by < num_blocks_y; by++) {
                for (int bx = 0; bx < num_blocks_x; bx++) {
                    all_instructions[instruction_count++] = (instruction_t){
                        .op = op,
                        .prev_op = prev_op,
                        .layer = layer_idx,
                        .b_x = bx,
                        .b_y = by,
                        .bar_idx = bar_idx,
                        .expected = expected
                    };
                }
            }
            prev_op = op;
        }
 
        // OP 25: MLP FC backward weight - [MLP_BACKWARD_WEIGHT_GRID(h * 4, h) blocks, 2D grid]
        {
            int op = 25;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 4;
            dim3 grid_prev = MLP_BACKWARD_INPUT_GRID(h, B, S);
            int expected = grid_prev.x * grid_prev.y;
            dim3 grid = MLP_BACKWARD_WEIGHT_GRID(h * 4, h);
            int num_blocks_x = grid.x;
            int num_blocks_y = grid.y;
 
            for (int by = 0; by < num_blocks_y; by++) {
                for (int bx = 0; bx < num_blocks_x; bx++) {
                    all_instructions[instruction_count++] = (instruction_t){
                        .op = op,
                        .prev_op = prev_op,
                        .layer = layer_idx,
                        .b_x = bx,
                        .b_y = by,
                        .bar_idx = bar_idx,
                        .expected = expected
                    };
                }
            }
            prev_op = op;
        }
 
        // OP 26: LayerNorm 2 backward - [B blocks, 1D grid]
        {
            int op = 26;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 5;
            dim3 grid_prev = MLP_BACKWARD_WEIGHT_GRID(h * 4, h);
            int expected = grid_prev.x * grid_prev.y;
            int num_blocks = B;
 
            for (int block = 0; block < num_blocks; block++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = layer_idx,
                    .b_x = block,
                    .b_y = -1,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
            prev_op = op;
        }
 
 
        // OP 27: Residual backward (res_2) - [CEIL_DIV(B * S * h, thr) blocks, 1D grid]
        {
            int op = 27;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 6;
            int expected = B;
            int num_blocks = CEIL_DIV(B * S * h, thr);
 
            for (int block = 0; block < num_blocks; block++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = layer_idx,
                    .b_x = block,
                    .b_y = -1,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
            prev_op = op;
        }
 
        // OP 28: Attention projection backward input - [MLP_BACKWARD_INPUT_GRID(h, B, S) blocks, 2D grid]
        {
            int op = 28;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 7;
            int expected = CEIL_DIV(B * S * h, thr);
            dim3 grid = MLP_BACKWARD_INPUT_GRID(h, B, S);
            int num_blocks_x = grid.x;
            int num_blocks_y = grid.y;
 
            for (int by = 0; by < num_blocks_y; by++) {
                for (int bx = 0; bx < num_blocks_x; bx++) {
                    all_instructions[instruction_count++] = (instruction_t){
                        .op = op,
                        .prev_op = prev_op,
                        .layer = layer_idx,
                        .b_x = bx,
                        .b_y = by,
                        .bar_idx = bar_idx,
                        .expected = expected
                    };
                }
            }
            prev_op = op;
        }
 
        // OP 29: Attention projection backward weight - [MLP_BACKWARD_WEIGHT_GRID(h, h) blocks, 2D grid]
        {
            int op = 29;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 8;
            dim3 grid_prev = MLP_BACKWARD_INPUT_GRID(h, B, S);
            int expected = grid_prev.x * grid_prev.y;
            dim3 grid = MLP_BACKWARD_WEIGHT_GRID(h, h);
            int num_blocks_x = grid.x;
            int num_blocks_y = grid.y;
 
            for (int by = 0; by < num_blocks_y; by++) {
                for (int bx = 0; bx < num_blocks_x; bx++) {
                    all_instructions[instruction_count++] = (instruction_t){
                        .op = op,
                        .prev_op = prev_op,
                        .layer = layer_idx,
                        .b_x = bx,
                        .b_y = by,
                        .bar_idx = bar_idx,
                        .expected = expected
                    };
                }
            }
            prev_op = op;
        }
 
        // OP 30: Attention backward - [CEIL_DIV(B * S * n_head, thr) blocks, 1D grid]
        {
            int op = 30;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 9;
            dim3 grid_prev = MLP_BACKWARD_WEIGHT_GRID(h, h);
            int expected = grid_prev.x * grid_prev.y;
            int num_blocks = CEIL_DIV(B * S * n_head, thr);
 
            for (int block = 0; block < num_blocks; block++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = layer_idx,
                    .b_x = block,
                    .b_y = -1,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
            prev_op = op;
        }
 
        // OP 31: QKV backward input - [MLP_BACKWARD_INPUT_GRID(h, B, S) blocks, 2D grid]
        {
            int op = 31;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 10;
            int expected = CEIL_DIV(B * S * n_head, thr);
            dim3 grid = MLP_BACKWARD_INPUT_GRID(h, B, S);
            int num_blocks_x = grid.x;
            int num_blocks_y = grid.y;
 
            for (int by = 0; by < num_blocks_y; by++) {
                for (int bx = 0; bx < num_blocks_x; bx++) {
                    all_instructions[instruction_count++] = (instruction_t){
                        .op = op,
                        .prev_op = prev_op,
                        .layer = layer_idx,
                        .b_x = bx,
                        .b_y = by,
                        .bar_idx = bar_idx,
                        .expected = expected
                    };
                }
            }
            prev_op = op;
        }
 
        // OP 32: QKV backward weight - [MLP_BACKWARD_WEIGHT_GRID(h * 3, h) blocks, 2D grid]
        {
            int op = 32;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 11;
            dim3 grid_prev = MLP_BACKWARD_INPUT_GRID(h, B, S);
            int expected = grid_prev.x * grid_prev.y;
            dim3 grid = MLP_BACKWARD_WEIGHT_GRID(h * 3, h);
            int num_blocks_x = grid.x;
            int num_blocks_y = grid.y;
 
            for (int by = 0; by < num_blocks_y; by++) {
                for (int bx = 0; bx < num_blocks_x; bx++) {
                    all_instructions[instruction_count++] = (instruction_t){
                        .op = op,
                        .prev_op = prev_op,
                        .layer = layer_idx,
                        .b_x = bx,
                        .b_y = by,
                        .bar_idx = bar_idx,
                        .expected = expected
                    };
                }
            }
            prev_op = op;
        }
 
        // OP 33: LayerNorm 1 backward - [B blocks, 1D grid]
        {
            int op = 33;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 12;
            dim3 grid_prev = MLP_BACKWARD_WEIGHT_GRID(h * 3, h);
            int expected = grid_prev.x * grid_prev.y;
            int num_blocks = B;
 
            for (int block = 0; block < num_blocks; block++) {
                all_instructions[instruction_count++] = (instruction_t){
                    .op = op,
                    .prev_op = prev_op,
                    .layer = layer_idx,
                    .b_x = block,
                    .b_y = -1,
                    .bar_idx = bar_idx,
                    .expected = expected
                };
            }
            prev_op = op;
        }
    }
 
    // OP 34: Embedding backward - [CEIL_DIV(B * S, thr) blocks, 1D grid]
    {
        int op = 34;
        int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1) * 14) + 13;
        int expected = B;
        int num_blocks = CEIL_DIV(B * S, thr);
 
        for (int block = 0; block < num_blocks; block++) {
            all_instructions[instruction_count++] = (instruction_t){
                .op = op,
                .prev_op = prev_op,
                .layer = -1,
                .b_x = block,
                .b_y = -1,
                .bar_idx = bar_idx,
                .expected = expected
            };
        }
        prev_op = op;
    }
 
    printf("Total instructions scheduled: %d\n", instruction_count);

    // Populate the `inc` flag: only the last `min(run_length, NUM_SM)`
    // instructions of each contiguous run of the same op should perform
    // the atomic increment. First, initialize all to false.
    for (int i = 0; i < instruction_count; i++) {
        all_instructions[i].inc = false;
    }

    // Walk through instructions and detect contiguous runs of the same op.
    // For each run, mark the last `min(run_len, NUM_SM)` entries as `inc = true`.
    if (instruction_count > 0) {
        int run_start = 0;
        int run_op = all_instructions[0].op;

        for (int i = 1; i <= instruction_count; i++) {
            // Treat end of array as run boundary
            int cur_op = (i < instruction_count) ? all_instructions[i].op : -9999;
            if (i == instruction_count || cur_op != run_op) {
                int run_end = i - 1;
                int run_len = run_end - run_start + 1;

                // Only mark runs with non-negative ops and skip ops that
                // should not perform increments (e.g. op 34).
                if (run_op >= 0 && run_op != 34) {
                    int to_mark = (run_len < NUM_SM) ? run_len : NUM_SM;
                    int start_idx = run_end - to_mark + 1;
                    for (int j = start_idx; j <= run_end; j++) {
                        if (j >= 0 && j < instruction_count) {
                            all_instructions[j].inc = true;
                        }
                    }
                }

                // start a new run if not at the end
                if (i < instruction_count) {
                    run_start = i;
                    run_op = cur_op;
                }
            }
        }
    }

    // Distribute instructions to SMs in round-robin fashion
    int *sm_counts = (int *)calloc(NUM_SM, sizeof(int));
 
    // First pass: count instructions per SM
    for (int i = 0; i < instruction_count; i++) {
        int sm_id = i % NUM_SM;
        sm_counts[sm_id]++;
    }
 
    // Allocate instruction arrays for each SM
    for (int sm = 0; sm < NUM_SM; sm++) {
        streams[sm]->n = sm_counts[sm];
        if (sm_counts[sm] > 0) {
            streams[sm]->instructions = (instruction_t *)malloc(sm_counts[sm] * sizeof(instruction_t));
        }
    }
 
    // Second pass: distribute instructions
    int *sm_indices = (int *)calloc(NUM_SM, sizeof(int));
    for (int i = 0; i < instruction_count; i++) {
        int sm_id = i % NUM_SM;
        streams[sm_id]->instructions[sm_indices[sm_id]++] = all_instructions[i];
    }
 
    // Cleanup
    free(all_instructions);
    free(sm_counts);
    free(sm_indices);
 
    printf("Scheduled %d instructions across %d SMs\n", instruction_count, NUM_SM);
    for (int sm = 0; sm < NUM_SM; sm++) {
        printf("SM %d: %d instructions\n", sm, streams[sm]->n);
    }
 
    // Allocate device memory for streams and instructions
    stream_t **d_streams_ptr;
    stream_t *d_stream_structs[NUM_SM];
 
    // Allocate device memory for the streams array pointer
    gpuErrchk(cudaMalloc(&d_streams_ptr, NUM_SM * sizeof(stream_t*)));
 
    // For each SM, allocate device memory for stream struct and instructions
    for (int sm = 0; sm < NUM_SM; sm++) {
        stream_t *d_stream;
        instruction_t *d_instructions = NULL;
 
        // Allocate device memory for stream struct
        gpuErrchk(cudaMalloc(&d_stream, sizeof(stream_t)));
 
        // Allocate and copy instructions if any
        if (streams[sm]->n > 0) {
            gpuErrchk(cudaMalloc(&d_instructions, streams[sm]->n * sizeof(instruction_t)));
            gpuErrchk(cudaMemcpy(d_instructions, streams[sm]->instructions, 
                                streams[sm]->n * sizeof(instruction_t), cudaMemcpyHostToDevice));
        }
 
        // Create temporary stream struct with device instruction pointer
        stream_t temp_stream = {streams[sm]->n, d_instructions};
 
        // Copy stream struct to device
        gpuErrchk(cudaMemcpy(d_stream, &temp_stream, sizeof(stream_t), cudaMemcpyHostToDevice));
 
        d_stream_structs[sm] = d_stream;
    }
 
    // Copy array of device stream pointers to device
    gpuErrchk(cudaMemcpy(d_streams_ptr, d_stream_structs, NUM_SM * sizeof(stream_t*), cudaMemcpyHostToDevice));
 
    return d_streams_ptr;
}
 
void free_schedule(stream_t **d_streams_ptr) {
    // First, copy device stream pointers back to host to free instruction arrays
    stream_t *d_stream_structs[NUM_SM];
    gpuErrchk(cudaMemcpy(d_stream_structs, d_streams_ptr, NUM_SM * sizeof(stream_t*), cudaMemcpyDeviceToHost));
 
    // Free device memory for each stream
    for (int sm = 0; sm < NUM_SM; sm++) {
        // Copy stream struct back to get instruction pointer
        stream_t temp_stream;
        gpuErrchk(cudaMemcpy(&temp_stream, d_stream_structs[sm], sizeof(stream_t), cudaMemcpyDeviceToHost));
 
        // Free device instructions if any
        if (temp_stream.instructions != NULL) {
            gpuErrchk(cudaFree(temp_stream.instructions));
        }
 
        // Free device stream struct
        gpuErrchk(cudaFree(d_stream_structs[sm]));
    }
 
    // Free device streams array
    gpuErrchk(cudaFree(d_streams_ptr));
 
    // Free host memory
    for (int sm = 0; sm < NUM_SM; sm++) {
        if (streams[sm] != NULL) {
            if (streams[sm]->instructions != NULL) {
                free(streams[sm]->instructions);
            }
            free(streams[sm]);
            streams[sm] = NULL;
        }
    }
}
 
__global__ void megakernel(
    config_t config,
 
    gpt2_t *model,
    gpt2_t *g_model,
 
    train_buffers_t *buffers,
    train_buffers_t *g_buffers,
 
    adamw_state_t opt_state,
 
    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,
 
    int *bar,
    stream_t **streams
) {
    int sm_id = blockIdx.x;  // Each SM gets its own block
    stream_t *stream = streams[sm_id];
    execute_stream(config, model, g_model, buffers, g_buffers, opt_state, seq_len, d_input_tokens, d_target_tokens, bar, stream);
}
 
 
__device__ void execute_stream(
    config_t config,
 
    gpt2_t *model,
    gpt2_t *g_model,
 
    train_buffers_t *buffers,
    train_buffers_t *g_buffers,
 
    adamw_state_t opt_state,
 
    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,
 
    int *bar,
    stream_t *stream
) {
    // if (threadIdx.x == 0) {
    //     printf("SM %d starting execution of %d instructions\n", blockIdx.x, stream->n);
    // }
 
    for (int i = 0; i < stream->n; i++) {
        instruction_t instr = stream->instructions[i];
        execute_instruction(config, model, g_model, buffers, g_buffers, opt_state, seq_len, d_input_tokens, d_target_tokens, bar, instr);
    }
}
 
 
__device__ void execute_instruction(
    config_t config,
 
    gpt2_t *model,
    gpt2_t *g_model,
 
    train_buffers_t *buffers,
    train_buffers_t *g_buffers,
 
    adamw_state_t opt_state,
 
    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,
 
    int *bar,
    instruction_t instr
) {
    int L = config.n_layer;
    int B = config.batch_size;
    int S = seq_len;
    int h = config.n_embd;
    int n_head = config.n_head;
    int V = config.vocab_size;
 
    volatile int *vbar = (volatile int *)bar;
 
    // Shared memory for MLP operations (2 * TILE_SIZE * TILE_SIZE floats)
    extern __shared__ float shared_mem[];
 
    if (instr.op != 1) {
        if (threadIdx.x == 0) {
            // printf("SM %d waiting at barrier %d for op %d (expecting %d, actual %d)\n", blockIdx.x, instr.bar_idx, instr.op, min(NUM_SM, instr.expected), vbar[instr.bar_idx]);
            while (vbar[instr.bar_idx] < min(NUM_SM, instr.expected)) {
                // printf("[SPIN] SM %d waiting at barrier %d for op %d (expecting %d, actual %d)\n", blockIdx.x, instr.bar_idx, instr.op, min(NUM_SM, instr.expected), vbar[instr.bar_idx]);
                // __nanosleep(10);
            }
        }
 
        __syncthreads();
    }
 
    switch (instr.op) {
        case 1: {
            // OP 1: Embedding forward
            embedding_forward_device(buffers->encoded.data, d_input_tokens, model->emb.wte.data, model->emb.wpe.data, S, h, V, config.n_positions, instr.b_x);
            break;
        }
 
        case 2: {
            // OP 2: LayerNorm 1
 
 
            float *res = (instr.layer == 0) ? buffers->encoded.data : buffers->blocks[instr.layer - 1].res_3.data;
            layernorm_forward_device(buffers->blocks[instr.layer].ln_1.data, res, model->h[instr.layer].ln_1.w.data, model->h[instr.layer].ln_1.b.data, buffers->blocks[instr.layer].ln_1_mean.data, buffers->blocks[instr.layer].ln_1_rstd.data, S, h, instr.b_x);
            break;
        }
 
        case 3: {
            // OP 3: QKV projection
 
            mlp_forward_device(buffers->blocks[instr.layer].qkv.data, buffers->blocks[instr.layer].ln_1.data, model->h[instr.layer].attn.qkv_w.data, model->h[instr.layer].attn.qkv_b.data, B, S, h, h * 3, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 4: {
            // OP 4: Attention
 
            attention_forward_device(buffers->blocks[instr.layer].atty.data, buffers->blocks[instr.layer].preatt.data, buffers->blocks[instr.layer].att.data, buffers->blocks[instr.layer].qkv.data, B, S, n_head, h, instr.b_x);
            break;
        }
 
        case 5: {
            // OP 5: Attention projection
 
            mlp_forward_device(buffers->blocks[instr.layer].att_proj.data, buffers->blocks[instr.layer].atty.data, model->h[instr.layer].attn.proj_w.data, model->h[instr.layer].attn.proj_b.data, B, S, h, h, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 6: {
            // OP 6: Residual 2
 
            float *res = (instr.layer == 0) ? buffers->encoded.data : buffers->blocks[instr.layer - 1].res_3.data;
            residual_forward_device(buffers->blocks[instr.layer].res_2.data, buffers->blocks[instr.layer].att_proj.data, res, B, S, h, instr.b_x);
            break;
        }
 
        case 7: {
            // OP 7: LayerNorm 2
 
            layernorm_forward_device(buffers->blocks[instr.layer].ln_2.data, buffers->blocks[instr.layer].res_2.data, model->h[instr.layer].ln_2.w.data, model->h[instr.layer].ln_2.b.data, buffers->blocks[instr.layer].ln_2_mean.data, buffers->blocks[instr.layer].ln_2_rstd.data, S, h, instr.b_x);
            break;
        }
 
        case 8: {
            // OP 8: MLP FC
 
            mlp_forward_device(buffers->blocks[instr.layer].mlp_fc.data, buffers->blocks[instr.layer].ln_2.data, model->h[instr.layer].mlp.fc_w.data, model->h[instr.layer].mlp.fc_b.data, B, S, h, h * 4, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 9: {
            // OP 9: GELU
 
            gelu_forward_device(buffers->blocks[instr.layer].mlp_fc_gelu.data, buffers->blocks[instr.layer].mlp_fc.data, B, S, h * 4, instr.b_x);
            break;
        }
 
        case 10: {
            // OP 10: MLP projection
 
            mlp_forward_device(buffers->blocks[instr.layer].mlp_proj.data, buffers->blocks[instr.layer].mlp_fc_gelu.data, model->h[instr.layer].mlp.proj_w.data, model->h[instr.layer].mlp.proj_b.data, B, S, h * 4, h, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 11: {
            // OP 11: Residual 3
 
 
            residual_forward_device(buffers->blocks[instr.layer].res_3.data, buffers->blocks[instr.layer].mlp_proj.data, buffers->blocks[instr.layer].res_2.data, B, S, h, instr.b_x);
            break;
        }
 
        case 12: {
            // OP 12: Final LayerNorm
 
 
            layernorm_forward_device(buffers->ln_f.data, buffers->blocks[L - 1].res_3.data, model->ln_f.w.data, model->ln_f.b.data, buffers->ln_f_mean.data, buffers->ln_f_rstd.data, S, h, instr.b_x);
            break;
        }
 
        case 13: {
            // OP 13: Logits
 
            mlp_forward_device(buffers->logits.data, buffers->ln_f.data, model->emb.wte.data, NULL, B, S, h, V, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 14: {
            // OP 14: Softmax
 
            softmax_forward_device(buffers->probs.data, buffers->logits.data, B, S, V, instr.b_x, shared_mem);
            break;
        }
 
        case 15: {
            // OP 15: Cross-entropy forward
 
            cross_entropy_forward_device(buffers->losses.data, buffers->probs.data, d_target_tokens, B, S, V, instr.b_x);
            break;
        }
 
        case 16: {
            // OP 16: Cross-entropy backward
 
            cross_entropy_backward_device(g_buffers->logits.data, buffers->probs.data, d_target_tokens, B, S, V, instr.b_x);
            break;
        }
 
        case 17: {
            // OP 17: Logits backward (input gradient)
 
            mlp_backward_input_device(g_buffers->ln_f.data, g_buffers->logits.data, model->emb.wte.data, B, S, h, V, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 18: {
            // OP 18: Embedding weight gradient
 
            mlp_backward_weight_device(g_model->emb.wte.data, NULL, g_buffers->logits.data, buffers->ln_f.data, B, S, h, V, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 19: {
            // OP 19: Final LayerNorm backward
 
            layernorm_backward_device(g_buffers->blocks[L - 1].res_3.data, g_model->ln_f.w.data, g_model->ln_f.b.data, g_buffers->ln_f.data, buffers->blocks[L - 1].res_3.data, model->ln_f.w.data, buffers->ln_f_mean.data, buffers->ln_f_rstd.data, B, S, h, instr.b_x);
            break;
        }
 
        case 20: {
            // OP 20: Residual backward (res_3)
            residual_backward_device(g_buffers->blocks[instr.layer].res_2.data, g_buffers->blocks[instr.layer].mlp_proj.data, g_buffers->blocks[instr.layer].res_3.data, B * S * h, instr.b_x);
            break;
        }
 
        case 21: {
            // OP 21: MLP projection backward input
 
            mlp_backward_input_device(g_buffers->blocks[instr.layer].mlp_fc_gelu.data, g_buffers->blocks[instr.layer].mlp_proj.data, model->h[instr.layer].mlp.proj_w.data, B, S, h * 4, h, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 22: {
            // OP 22: MLP projection backward weight
            mlp_backward_weight_device(g_model->h[instr.layer].mlp.proj_w.data, g_model->h[instr.layer].mlp.proj_b.data, g_buffers->blocks[instr.layer].mlp_proj.data, buffers->blocks[instr.layer].mlp_fc_gelu.data, B, S, h * 4, h, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 23: {
            // OP 23: GELU backward
 
            gelu_backward_device(g_buffers->blocks[instr.layer].mlp_fc.data, buffers->blocks[instr.layer].mlp_fc.data, g_buffers->blocks[instr.layer].mlp_fc_gelu.data, B * S * 4 * h, instr.b_x);
            break;
        }
 
        case 24: {
            // OP 24: MLP FC backward input
            mlp_backward_input_device(g_buffers->blocks[instr.layer].ln_2.data, g_buffers->blocks[instr.layer].mlp_fc.data, model->h[instr.layer].mlp.fc_w.data, B, S, h, h * 4, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 25: {
            // OP 25: MLP FC backward weight
 
            mlp_backward_weight_device(g_model->h[instr.layer].mlp.fc_w.data, g_model->h[instr.layer].mlp.fc_b.data, g_buffers->blocks[instr.layer].mlp_fc.data, buffers->blocks[instr.layer].ln_2.data, B, S, h, h * 4, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 26: {
            // OP 26: LayerNorm 2 backward
 
            layernorm_backward_device(g_buffers->blocks[instr.layer].res_2.data, g_model->h[instr.layer].ln_2.w.data, g_model->h[instr.layer].ln_2.b.data, g_buffers->blocks[instr.layer].ln_2.data, buffers->blocks[instr.layer].res_2.data, model->h[instr.layer].ln_2.w.data, buffers->blocks[instr.layer].ln_2_mean.data, buffers->blocks[instr.layer].ln_2_rstd.data, B, S, h, instr.b_x);
            break;
        }
 
        case 27: {
            // OP 27: Residual backward (res_2)
 
            float *g_res = (instr.layer == 0) ? g_buffers->encoded.data : g_buffers->blocks[instr.layer - 1].res_3.data;
            residual_backward_device(g_res, g_buffers->blocks[instr.layer].att_proj.data, g_buffers->blocks[instr.layer].res_2.data, B * S * h, instr.b_x);
            break;
        }
 
        case 28: {
            // OP 28: Attention projection backward input
 
            mlp_backward_input_device(g_buffers->blocks[instr.layer].atty.data, g_buffers->blocks[instr.layer].att_proj.data, model->h[instr.layer].attn.proj_w.data, B, S, h, h, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 29: {
            // OP 29: Attention projection backward weight
 
            mlp_backward_weight_device(g_model->h[instr.layer].attn.proj_w.data, g_model->h[instr.layer].attn.proj_b.data, g_buffers->blocks[instr.layer].att_proj.data, buffers->blocks[instr.layer].atty.data, B, S, h, h, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 30: {
            // OP 30: Attention backward
 
            attention_backward_device(g_buffers->blocks[instr.layer].qkv.data, g_buffers->blocks[instr.layer].preatt.data, g_buffers->blocks[instr.layer].att.data, g_buffers->blocks[instr.layer].atty.data, buffers->blocks[instr.layer].qkv.data, buffers->blocks[instr.layer].att.data, B, S, h, n_head, instr.b_x);
            break;
        }
 
        case 31: {
            // OP 31: QKV backward input
 
            mlp_backward_input_device(g_buffers->blocks[instr.layer].ln_1.data, g_buffers->blocks[instr.layer].qkv.data, model->h[instr.layer].attn.qkv_w.data, B, S, h, h * 3, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 32: {
            // OP 32: QKV backward weight
 
            mlp_backward_weight_device(g_model->h[instr.layer].attn.qkv_w.data, g_model->h[instr.layer].attn.qkv_b.data, g_buffers->blocks[instr.layer].qkv.data, buffers->blocks[instr.layer].ln_1.data, B, S, h, h * 3, instr.b_x, instr.b_y, shared_mem);
            break;
        }
 
        case 33: {
            // OP 33: LayerNorm 1 backward
 
            float *g_res = (instr.layer == 0) ? g_buffers->encoded.data : g_buffers->blocks[instr.layer - 1].res_3.data;
            float *res = (instr.layer == 0) ? buffers->encoded.data : buffers->blocks[instr.layer - 1].res_3.data;
 
            layernorm_backward_device(g_res, g_model->h[instr.layer].ln_1.w.data, g_model->h[instr.layer].ln_1.b.data, g_buffers->blocks[instr.layer].ln_1.data, res, model->h[instr.layer].ln_1.w.data, buffers->blocks[instr.layer].ln_1_mean.data, buffers->blocks[instr.layer].ln_1_rstd.data, B, S, h, instr.b_x);
            break;
        }
 
        case 34: {
            // OP 34: Embedding backward
 
            embedding_backward_device(g_model->emb.wte.data, g_model->emb.wpe.data, g_buffers->encoded.data, d_input_tokens, B, S, h, instr.b_x);
            break;
        }
 
        default:
            break;
    }
 
    __syncthreads();

    // Only the instructions marked with `inc` perform the atomic increment.
    if (instr.inc && threadIdx.x == 0) {
        // printf("[INC]\n");
        atomicAdd(&bar[instr.bar_idx + 1], 1);
    }
}
