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

// ============================================================================
// Compile-time constants for megakernel pointer arithmetic
// ============================================================================
#define MK_BATCH_SIZE 4
#define MK_SEQ_LEN 64
#define MK_VOCAB_SIZE 50257
#define MK_N_LAYER 12
#define MK_N_HEAD 12
#define MK_N_EMBD 768
#define MK_N_POSITIONS 1024
#define MK_FOUR_H (4 * MK_N_EMBD)

// ============================================================================
// Model parameter offsets (from params_memory base pointer)
// ============================================================================
// emb.wte: (h, V) => h * V
#define MK_WTE_OFFSET 0
#define MK_WTE_SIZE (MK_N_EMBD * MK_VOCAB_SIZE)

// emb.wpe: (maxT, h) => n_positions * h
#define MK_WPE_OFFSET (MK_WTE_OFFSET + MK_WTE_SIZE)
#define MK_WPE_SIZE (MK_N_POSITIONS * MK_N_EMBD)

// Per-layer parameter sizes
#define MK_LN_W_SIZE MK_N_EMBD
#define MK_LN_B_SIZE MK_N_EMBD
#define MK_QKV_W_SIZE (MK_N_EMBD * 3 * MK_N_EMBD)
#define MK_QKV_B_SIZE (3 * MK_N_EMBD)
#define MK_ATTN_PROJ_W_SIZE (MK_N_EMBD * MK_N_EMBD)
#define MK_ATTN_PROJ_B_SIZE MK_N_EMBD
#define MK_MLP_FC_W_SIZE (MK_N_EMBD * MK_FOUR_H)
#define MK_MLP_FC_B_SIZE MK_FOUR_H
#define MK_MLP_PROJ_W_SIZE (MK_FOUR_H * MK_N_EMBD)
#define MK_MLP_PROJ_B_SIZE MK_N_EMBD

// Size of one transformer block's parameters
#define MK_BLOCK_PARAMS_SIZE ( \
    MK_LN_W_SIZE + MK_LN_B_SIZE + \
    MK_QKV_W_SIZE + MK_QKV_B_SIZE + MK_ATTN_PROJ_W_SIZE + MK_ATTN_PROJ_B_SIZE + \
    MK_LN_W_SIZE + MK_LN_B_SIZE + \
    MK_MLP_FC_W_SIZE + MK_MLP_FC_B_SIZE + MK_MLP_PROJ_W_SIZE + MK_MLP_PROJ_B_SIZE \
)

// Start of block parameters
#define MK_BLOCKS_OFFSET (MK_WPE_OFFSET + MK_WPE_SIZE)

// Offsets within a single block (relative to block start)
#define MK_BLK_LN1_W_OFF 0
#define MK_BLK_LN1_B_OFF (MK_BLK_LN1_W_OFF + MK_LN_W_SIZE)
#define MK_BLK_QKV_W_OFF (MK_BLK_LN1_B_OFF + MK_LN_B_SIZE)
#define MK_BLK_QKV_B_OFF (MK_BLK_QKV_W_OFF + MK_QKV_W_SIZE)
#define MK_BLK_ATTN_PROJ_W_OFF (MK_BLK_QKV_B_OFF + MK_QKV_B_SIZE)
#define MK_BLK_ATTN_PROJ_B_OFF (MK_BLK_ATTN_PROJ_W_OFF + MK_ATTN_PROJ_W_SIZE)
#define MK_BLK_LN2_W_OFF (MK_BLK_ATTN_PROJ_B_OFF + MK_ATTN_PROJ_B_SIZE)
#define MK_BLK_LN2_B_OFF (MK_BLK_LN2_W_OFF + MK_LN_W_SIZE)
#define MK_BLK_MLP_FC_W_OFF (MK_BLK_LN2_B_OFF + MK_LN_B_SIZE)
#define MK_BLK_MLP_FC_B_OFF (MK_BLK_MLP_FC_W_OFF + MK_MLP_FC_W_SIZE)
#define MK_BLK_MLP_PROJ_W_OFF (MK_BLK_MLP_FC_B_OFF + MK_MLP_FC_B_SIZE)
#define MK_BLK_MLP_PROJ_B_OFF (MK_BLK_MLP_PROJ_W_OFF + MK_MLP_PROJ_W_SIZE)

// Final layernorm
#define MK_LN_F_W_OFFSET (MK_BLOCKS_OFFSET + MK_N_LAYER * MK_BLOCK_PARAMS_SIZE)
#define MK_LN_F_B_OFFSET (MK_LN_F_W_OFFSET + MK_N_EMBD)

// Macros to get layer-specific parameter pointers
#define MK_BLOCK_OFFSET(layer) (MK_BLOCKS_OFFSET + (layer) * MK_BLOCK_PARAMS_SIZE)
#define MK_LN1_W(base, layer) ((base) + MK_BLOCK_OFFSET(layer) + MK_BLK_LN1_W_OFF)
#define MK_LN1_B(base, layer) ((base) + MK_BLOCK_OFFSET(layer) + MK_BLK_LN1_B_OFF)
#define MK_QKV_W(base, layer) ((base) + MK_BLOCK_OFFSET(layer) + MK_BLK_QKV_W_OFF)
#define MK_QKV_B(base, layer) ((base) + MK_BLOCK_OFFSET(layer) + MK_BLK_QKV_B_OFF)
#define MK_ATTN_PROJ_W(base, layer) ((base) + MK_BLOCK_OFFSET(layer) + MK_BLK_ATTN_PROJ_W_OFF)
#define MK_ATTN_PROJ_B(base, layer) ((base) + MK_BLOCK_OFFSET(layer) + MK_BLK_ATTN_PROJ_B_OFF)
#define MK_LN2_W(base, layer) ((base) + MK_BLOCK_OFFSET(layer) + MK_BLK_LN2_W_OFF)
#define MK_LN2_B(base, layer) ((base) + MK_BLOCK_OFFSET(layer) + MK_BLK_LN2_B_OFF)
#define MK_MLP_FC_W(base, layer) ((base) + MK_BLOCK_OFFSET(layer) + MK_BLK_MLP_FC_W_OFF)
#define MK_MLP_FC_B(base, layer) ((base) + MK_BLOCK_OFFSET(layer) + MK_BLK_MLP_FC_B_OFF)
#define MK_MLP_PROJ_W(base, layer) ((base) + MK_BLOCK_OFFSET(layer) + MK_BLK_MLP_PROJ_W_OFF)
#define MK_MLP_PROJ_B(base, layer) ((base) + MK_BLOCK_OFFSET(layer) + MK_BLK_MLP_PROJ_B_OFF)
#define MK_WTE(base) ((base) + MK_WTE_OFFSET)
#define MK_WPE(base) ((base) + MK_WPE_OFFSET)
#define MK_LN_F_W(base) ((base) + MK_LN_F_W_OFFSET)
#define MK_LN_F_B(base) ((base) + MK_LN_F_B_OFFSET)

// ============================================================================
// Activation buffer offsets (from activations_memory base pointer)
// ============================================================================
// encoded: B * S * h
#define MK_ACT_ENCODED_OFFSET 0
#define MK_ACT_ENCODED_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * MK_N_EMBD)

// Per-layer activation sizes
#define MK_ACT_LN1_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * MK_N_EMBD)
#define MK_ACT_LN1_MEAN_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN)
#define MK_ACT_LN1_RSTD_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN)
#define MK_ACT_QKV_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * 3 * MK_N_EMBD)
#define MK_ACT_ATTY_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * MK_N_EMBD)
#define MK_ACT_PREATT_SIZE (MK_BATCH_SIZE * MK_N_HEAD * MK_SEQ_LEN * MK_SEQ_LEN)
#define MK_ACT_ATT_SIZE (MK_BATCH_SIZE * MK_N_HEAD * MK_SEQ_LEN * MK_SEQ_LEN)
#define MK_ACT_ATT_PROJ_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * MK_N_EMBD)
#define MK_ACT_RES2_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * MK_N_EMBD)
#define MK_ACT_LN2_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * MK_N_EMBD)
#define MK_ACT_LN2_MEAN_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN)
#define MK_ACT_LN2_RSTD_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN)
#define MK_ACT_MLP_FC_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * MK_FOUR_H)
#define MK_ACT_MLP_FC_GELU_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * MK_FOUR_H)
#define MK_ACT_MLP_PROJ_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * MK_N_EMBD)
#define MK_ACT_RES3_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * MK_N_EMBD)

// Size of one transformer block's activations
#define MK_ACT_BLOCK_SIZE ( \
    MK_ACT_LN1_SIZE + MK_ACT_LN1_MEAN_SIZE + MK_ACT_LN1_RSTD_SIZE + \
    MK_ACT_QKV_SIZE + MK_ACT_ATTY_SIZE + MK_ACT_PREATT_SIZE + MK_ACT_ATT_SIZE + \
    MK_ACT_ATT_PROJ_SIZE + MK_ACT_RES2_SIZE + \
    MK_ACT_LN2_SIZE + MK_ACT_LN2_MEAN_SIZE + MK_ACT_LN2_RSTD_SIZE + \
    MK_ACT_MLP_FC_SIZE + MK_ACT_MLP_FC_GELU_SIZE + MK_ACT_MLP_PROJ_SIZE + MK_ACT_RES3_SIZE \
)

// Start of block activations
#define MK_ACT_BLOCKS_OFFSET (MK_ACT_ENCODED_OFFSET + MK_ACT_ENCODED_SIZE)

// Offsets within a single block's activations (relative to block start)
#define MK_ACT_BLK_LN1_OFF 0
#define MK_ACT_BLK_LN1_MEAN_OFF (MK_ACT_BLK_LN1_OFF + MK_ACT_LN1_SIZE)
#define MK_ACT_BLK_LN1_RSTD_OFF (MK_ACT_BLK_LN1_MEAN_OFF + MK_ACT_LN1_MEAN_SIZE)
#define MK_ACT_BLK_QKV_OFF (MK_ACT_BLK_LN1_RSTD_OFF + MK_ACT_LN1_RSTD_SIZE)
#define MK_ACT_BLK_ATTY_OFF (MK_ACT_BLK_QKV_OFF + MK_ACT_QKV_SIZE)
#define MK_ACT_BLK_PREATT_OFF (MK_ACT_BLK_ATTY_OFF + MK_ACT_ATTY_SIZE)
#define MK_ACT_BLK_ATT_OFF (MK_ACT_BLK_PREATT_OFF + MK_ACT_PREATT_SIZE)
#define MK_ACT_BLK_ATT_PROJ_OFF (MK_ACT_BLK_ATT_OFF + MK_ACT_ATT_SIZE)
#define MK_ACT_BLK_RES2_OFF (MK_ACT_BLK_ATT_PROJ_OFF + MK_ACT_ATT_PROJ_SIZE)
#define MK_ACT_BLK_LN2_OFF (MK_ACT_BLK_RES2_OFF + MK_ACT_RES2_SIZE)
#define MK_ACT_BLK_LN2_MEAN_OFF (MK_ACT_BLK_LN2_OFF + MK_ACT_LN2_SIZE)
#define MK_ACT_BLK_LN2_RSTD_OFF (MK_ACT_BLK_LN2_MEAN_OFF + MK_ACT_LN2_MEAN_SIZE)
#define MK_ACT_BLK_MLP_FC_OFF (MK_ACT_BLK_LN2_RSTD_OFF + MK_ACT_LN2_RSTD_SIZE)
#define MK_ACT_BLK_MLP_FC_GELU_OFF (MK_ACT_BLK_MLP_FC_OFF + MK_ACT_MLP_FC_SIZE)
#define MK_ACT_BLK_MLP_PROJ_OFF (MK_ACT_BLK_MLP_FC_GELU_OFF + MK_ACT_MLP_FC_GELU_SIZE)
#define MK_ACT_BLK_RES3_OFF (MK_ACT_BLK_MLP_PROJ_OFF + MK_ACT_MLP_PROJ_SIZE)

// Final layer activations
#define MK_ACT_LN_F_OFFSET (MK_ACT_BLOCKS_OFFSET + MK_N_LAYER * MK_ACT_BLOCK_SIZE)
#define MK_ACT_LN_F_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * MK_N_EMBD)
#define MK_ACT_LN_F_MEAN_OFFSET (MK_ACT_LN_F_OFFSET + MK_ACT_LN_F_SIZE)
#define MK_ACT_LN_F_MEAN_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN)
#define MK_ACT_LN_F_RSTD_OFFSET (MK_ACT_LN_F_MEAN_OFFSET + MK_ACT_LN_F_MEAN_SIZE)
#define MK_ACT_LN_F_RSTD_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN)
#define MK_ACT_LOGITS_OFFSET (MK_ACT_LN_F_RSTD_OFFSET + MK_ACT_LN_F_RSTD_SIZE)
#define MK_ACT_LOGITS_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * MK_VOCAB_SIZE)
#define MK_ACT_PROBS_OFFSET (MK_ACT_LOGITS_OFFSET + MK_ACT_LOGITS_SIZE)
#define MK_ACT_PROBS_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN * MK_VOCAB_SIZE)
#define MK_ACT_LOSSES_OFFSET (MK_ACT_PROBS_OFFSET + MK_ACT_PROBS_SIZE)
#define MK_ACT_LOSSES_SIZE (MK_BATCH_SIZE * MK_SEQ_LEN)

// Macros to get layer-specific activation pointers
#define MK_ACT_BLOCK_OFFSET(layer) (MK_ACT_BLOCKS_OFFSET + (layer) * MK_ACT_BLOCK_SIZE)
#define MK_ACT_ENCODED(base) ((base) + MK_ACT_ENCODED_OFFSET)
#define MK_ACT_LN1(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_LN1_OFF)
#define MK_ACT_LN1_MEAN(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_LN1_MEAN_OFF)
#define MK_ACT_LN1_RSTD(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_LN1_RSTD_OFF)
#define MK_ACT_QKV(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_QKV_OFF)
#define MK_ACT_ATTY(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_ATTY_OFF)
#define MK_ACT_PREATT(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_PREATT_OFF)
#define MK_ACT_ATT(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_ATT_OFF)
#define MK_ACT_ATT_PROJ(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_ATT_PROJ_OFF)
#define MK_ACT_RES2(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_RES2_OFF)
#define MK_ACT_LN2(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_LN2_OFF)
#define MK_ACT_LN2_MEAN(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_LN2_MEAN_OFF)
#define MK_ACT_LN2_RSTD(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_LN2_RSTD_OFF)
#define MK_ACT_MLP_FC(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_MLP_FC_OFF)
#define MK_ACT_MLP_FC_GELU(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_MLP_FC_GELU_OFF)
#define MK_ACT_MLP_PROJ(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_MLP_PROJ_OFF)
#define MK_ACT_RES3(base, layer) ((base) + MK_ACT_BLOCK_OFFSET(layer) + MK_ACT_BLK_RES3_OFF)
#define MK_ACT_LN_F(base) ((base) + MK_ACT_LN_F_OFFSET)
#define MK_ACT_LN_F_MEAN(base) ((base) + MK_ACT_LN_F_MEAN_OFFSET)
#define MK_ACT_LN_F_RSTD(base) ((base) + MK_ACT_LN_F_RSTD_OFFSET)
#define MK_ACT_LOGITS(base) ((base) + MK_ACT_LOGITS_OFFSET)
#define MK_ACT_PROBS(base) ((base) + MK_ACT_PROBS_OFFSET)
#define MK_ACT_LOSSES(base) ((base) + MK_ACT_LOSSES_OFFSET)

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
    float *activations_memory;  // Contiguous memory for all activations
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
    int start_b_x, end_b_x;
    int start_b_y, end_b_y;

    int bar_idx;
    int expected;
    bool inc;
    int instr_idx;  // Index of this instruction within its SM
} instruction_t;

// Maximum instructions per SM (for timing arrays)
#define MAX_INSTR_PER_SM 2000

typedef struct {
    int n;
    instruction_t *instructions;
} stream_t;

int *bar;  // [B, 1 + (L * 10 + 3) + 1 + (5 + L * 13 + 1)] global atomics
stream_t *streams[NUM_SM];  // Host streams
stream_t **d_streams;  // Device streams array

// Timing buffers for measuring time spent in each SM
long long *d_sm_start_times;  // Start time for each SM (clock64)
long long *d_sm_end_times;    // End time for each SM (clock64)

// Per-instruction timing: bar_enter_time[sm * MAX_INSTR_PER_SM + instr_idx]
long long *d_bar_enter_time;  // Time before entering spin loop
long long *d_bar_exit_time;   // Time after exiting spin loop
int *h_instr_counts;          // Number of instructions per SM (for output)

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
    float *params,
    float *grads,

    float *acts,
    float *grad_acts,

    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,

    int *bar,
    stream_t **streams,
    long long *sm_start_times,
    long long *sm_end_times,
    long long *bar_enter_time,
    long long *bar_exit_time
);

__device__ void execute_stream(
    float *params,
    float *grads,

    float *acts,
    float *grad_acts,

    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,

    int *bar,
    stream_t *stream,
    long long *bar_enter_time,
    long long *bar_exit_time
);

__device__ void execute_instruction(
    float *params,
    float *grads,

    float *acts,
    float *grad_acts,

    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,

    int *bar,
    instruction_t instr,
    long long *bar_enter_time,
    long long *bar_exit_time
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

    // allocate global bar
    int bar_size = config.batch_size * (1 + (config.n_layer * 10 + 3) + 1 + (5 + config.n_layer * 14 + 1));
    gpuErrchk(cudaMalloc(&bar, bar_size * sizeof(int)));

    int shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    int threads_per_block = 1024;
    printf("Shared memory size per block: %d bytes\n", shared_mem_size);

    // Allocate timing buffers for SM timing measurement
    gpuErrchk(cudaMalloc(&d_sm_start_times, NUM_SM * sizeof(long long)));
    gpuErrchk(cudaMalloc(&d_sm_end_times, NUM_SM * sizeof(long long)));
    long long *h_sm_start_times = (long long *)malloc(NUM_SM * sizeof(long long));
    long long *h_sm_end_times = (long long *)malloc(NUM_SM * sizeof(long long));

    // Allocate per-instruction timing buffers for bar enter/exit times
    gpuErrchk(cudaMalloc(&d_bar_enter_time, NUM_SM * MAX_INSTR_PER_SM * sizeof(long long)));
    gpuErrchk(cudaMalloc(&d_bar_exit_time, NUM_SM * MAX_INSTR_PER_SM * sizeof(long long)));
    long long *h_bar_enter_time = (long long *)malloc(NUM_SM * MAX_INSTR_PER_SM * sizeof(long long));
    long long *h_bar_exit_time = (long long *)malloc(NUM_SM * MAX_INSTR_PER_SM * sizeof(long long));

    // Store instruction counts per SM for output
    h_instr_counts = (int *)malloc(NUM_SM * sizeof(int));
    for (int sm = 0; sm < NUM_SM; sm++) {
        h_instr_counts[sm] = streams[sm]->n;
    }
    
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
    // MARK: ITER
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
            model.params_memory,
            g_model.params_memory,
            buffers.activations_memory,
            g_buffers.activations_memory,
            T,
            d_input_tokens,
            d_target_tokens,
            bar,
            d_streams,
            d_sm_start_times,
            d_sm_end_times,
            d_bar_enter_time,
            d_bar_exit_time
        );
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        float mean_loss = compute_mean_loss(&buffers.losses, B, T);
        
        // Update parameters
        gpt2_update(&model, &g_model, &opt_state);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        printf("step %d: loss %f (took %f ms)\n", step, mean_loss, time_elapsed_s * 1000);

        // Copy timing data from device to host and print
        gpuErrchk(cudaMemcpy(h_sm_start_times, d_sm_start_times, NUM_SM * sizeof(long long), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_sm_end_times, d_sm_end_times, NUM_SM * sizeof(long long), cudaMemcpyDeviceToHost));
        
        // Find min start time to compute relative times
        long long min_start = h_sm_start_times[0];
        for (int sm = 1; sm < NUM_SM; sm++) {
            if (h_sm_start_times[sm] < min_start) {
                min_start = h_sm_start_times[sm];
            }
        }
        
        printf("SM Timing (step %d):\n", step);
        for (int sm = 0; sm < NUM_SM; sm++) {
            long long rel_start = h_sm_start_times[sm] - min_start;
            long long rel_end = h_sm_end_times[sm] - min_start;
            long long duration = h_sm_end_times[sm] - h_sm_start_times[sm];
            printf("  SM %2d: start=%12lld, end=%12lld, duration=%12lld ms\n", sm, rel_start, rel_end, duration / 1000000);
        }

        // Copy per-instruction timing data and write to timer.txt
        gpuErrchk(cudaMemcpy(h_bar_enter_time, d_bar_enter_time, NUM_SM * MAX_INSTR_PER_SM * sizeof(long long), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_bar_exit_time, d_bar_exit_time, NUM_SM * MAX_INSTR_PER_SM * sizeof(long long), cudaMemcpyDeviceToHost));

        // Write to timer.txt
        {
            FILE *f = fopen("timer.txt", step == 0 ? "w" : "a");
            if (f) {
                fprintf(f, "=== Step %d ===\n", step);
                for (int sm = 0; sm < NUM_SM; sm++) {
                    fprintf(f, "SM%d:\n", sm);
                    for (int i = 0; i < h_instr_counts[sm]; i++) {
                        int idx = sm * MAX_INSTR_PER_SM + i;
                        long long enter = h_bar_enter_time[idx];
                        long long exit = h_bar_exit_time[idx];
                        long long duration = exit - enter;
                        fprintf(f, "  instr %d: enter=%lld, exit=%lld, duration=%lld ms\n", i, enter - min_start, exit - min_start, duration / 1000000);
                    }
                }
                fprintf(f, "\n");
                fclose(f);
            }
        }

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
    cudaFree(d_sm_start_times);
    cudaFree(d_sm_end_times);
    free(h_sm_start_times);
    free(h_sm_end_times);
    cudaFree(d_bar_enter_time);
    cudaFree(d_bar_exit_time);
    free(h_bar_enter_time);
    free(h_bar_exit_time);
    free(h_instr_counts);
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

    // Calculate total size needed for all activations
    size_t total_size = 0;
    
    // encoded: B * S * h
    total_size += B * S * h;
    
    // Per layer activations
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
    
    // Final layers
    total_size += B * S * h;  // ln_f
    total_size += B * S;      // ln_f_mean
    total_size += B * S;      // ln_f_rstd
    total_size += B * S * V;  // logits
    total_size += B * S * V;  // probs
    total_size += B * S;      // losses
    
    // Allocate single contiguous block on GPU
    cudaError_t err = cudaMalloc(&buffers->activations_memory, total_size * sizeof(float));
    if (err != cudaSuccess) {
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
    if (buffers->activations_memory) {
        cudaFree(buffers->activations_memory);
        buffers->activations_memory = NULL;
    }
    
    // No need to free individual tensors since they're now pointers into the contiguous block
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

// Helper function to create instructions with block ranges
// For 1D grids: pass num_blocks_y = 1
// Creates at most NUM_SM instructions per operation
void add_instructions_1d(instruction_t *all_instructions, int *instruction_count, 
                         int op, int prev_op, int layer, int bar_idx, int expected, 
                         int num_blocks) {
    int num_instructions = (num_blocks <= NUM_SM) ? num_blocks : NUM_SM;
    int blocks_per_instruction = (num_blocks + num_instructions - 1) / num_instructions;
    
    for (int i = 0; i < num_instructions; i++) {
        int start_b_x = i * blocks_per_instruction;
        int end_b_x = ((i + 1) * blocks_per_instruction < num_blocks) ? 
                      ((i + 1) * blocks_per_instruction - 1) : (num_blocks - 1);
        
        if (start_b_x < num_blocks) {
            all_instructions[(*instruction_count)++] = (instruction_t){
                .op = op,
                .prev_op = prev_op,
                .layer = layer,
                .start_b_x = start_b_x,
                .end_b_x = end_b_x,
                .start_b_y = 0,
                .end_b_y = 0,
                .bar_idx = bar_idx,
                .expected = expected
            };
        }
    }
}

void add_instructions_2d(instruction_t *all_instructions, int *instruction_count,
                         int op, int prev_op, int layer, int bar_idx, int expected,
                         int num_blocks_x, int num_blocks_y) {
    int total_blocks = num_blocks_x * num_blocks_y;
    int num_instructions = (total_blocks <= NUM_SM) ? total_blocks : NUM_SM;
    int blocks_per_instruction = (total_blocks + num_instructions - 1) / num_instructions;
    
    for (int i = 0; i < num_instructions; i++) {
        int start_linear = i * blocks_per_instruction;
        int end_linear = ((i + 1) * blocks_per_instruction < total_blocks) ?
                         ((i + 1) * blocks_per_instruction - 1) : (total_blocks - 1);
        
        if (start_linear < total_blocks) {
            int start_b_y = start_linear / num_blocks_x;
            int start_b_x = start_linear % num_blocks_x;
            int end_b_y = end_linear / num_blocks_x;
            int end_b_x = end_linear % num_blocks_x;
            
            all_instructions[(*instruction_count)++] = (instruction_t){
                .op = op,
                .prev_op = prev_op,
                .layer = layer,
                .start_b_x = start_b_x,
                .end_b_x = end_b_x,
                .start_b_y = start_b_y,
                .end_b_y = end_b_y,
                .bar_idx = bar_idx,
                .expected = expected
            };
        }
    }
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
    int max_instructions = 1000000; // estimate
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
 
        add_instructions_1d(all_instructions, &instruction_count, op, prev_op, layer, bar_idx, expected, num_blocks);
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
 
            add_instructions_1d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks);
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
 
            add_instructions_2d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks_x, num_blocks_y);
            prev_op = op;
        }
 
        // OP 4: Attention - [CEIL_DIV(B * S * n_head, thr) blocks, 1D grid]
        {
            int op = 4;
            int bar_idx = 1 + layer_idx * 10 + 1;
            dim3 grid = MLP_FORWARD_GRID(h * 3, B, S);
            int expected = grid.x * grid.y;
            int num_blocks = CEIL_DIV(B * S * n_head, thr);
 
            add_instructions_1d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks);
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
 
            add_instructions_2d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks_x, num_blocks_y);
            prev_op = op;
        }
 
        // OP 6: Residual 2 - [CEIL_DIV(B * S * h, thr) blocks, 1D grid]
        {
            int op = 6;
            int bar_idx = 1 + layer_idx * 10 + 3;
            dim3 grid = MLP_FORWARD_GRID(h, B, S);
            int expected = grid.x * grid.y;
            int num_blocks = CEIL_DIV(B * S * h, thr);
 
            add_instructions_1d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks);
            prev_op = op;
        }
 
        // OP 7: LayerNorm 2 - [B blocks, 1D grid]
        {
            int op = 7;
            int bar_idx = 1 + layer_idx * 10 + 4;
            int expected = CEIL_DIV(B * S * h, thr);
            int num_blocks = B;
 
            add_instructions_1d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks);
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
 
            add_instructions_2d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks_x, num_blocks_y);
            prev_op = op;
        }
 
        // OP 9: GELU - [CEIL_DIV(B * S * 4 * h, thr) blocks, 1D grid]
        {
            int op = 9;
            int bar_idx = 1 + layer_idx * 10 + 6;
            dim3 grid = MLP_FORWARD_GRID(h * 4, B, S);
            int expected = grid.x * grid.y;
            int num_blocks = CEIL_DIV(B * S * 4 * h, thr);
 
            add_instructions_1d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks);
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
 
            add_instructions_2d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks_x, num_blocks_y);
            prev_op = op;
        }
 
        // OP 11: Residual 3 - [CEIL_DIV(B * S * h, thr) blocks, 1D grid]
        {
            int op = 11;
            int bar_idx = 1 + layer_idx * 10 + 8;
            dim3 grid = MLP_FORWARD_GRID(h, B, S);
            int expected = grid.x * grid.y;
            int num_blocks = CEIL_DIV(B * S * h, thr);
 
            add_instructions_1d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks);
            prev_op = op;
        }
    }
 
    // OP 12: Final LayerNorm - [B blocks, 1D grid]
    {
        int op = 12;
        int bar_idx = 1 + (L - 1) * 10 + 9;
        int expected = CEIL_DIV(B * S * h, thr);
        int num_blocks = B;
 
        add_instructions_1d(all_instructions, &instruction_count, op, prev_op, -1, bar_idx, expected, num_blocks);
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
 
        add_instructions_2d(all_instructions, &instruction_count, op, prev_op, -1, bar_idx, expected, num_blocks_x, num_blocks_y);
        prev_op = op;
    }
 
    // OP 14: Softmax - [CEIL_DIV(B * S * V, thr) blocks, 1D grid]
    {
        int op = 14;
        int bar_idx = 1 + (L * 10) + 1;
        dim3 grid = MLP_FORWARD_GRID(V, B, S);
        int expected = grid.x * grid.y;
        int num_blocks = CEIL_DIV(B * S * V, thr);
 
        add_instructions_1d(all_instructions, &instruction_count, op, prev_op, -1, bar_idx, expected, num_blocks);
        prev_op = op;
    }
 
    // OP 15: Cross-entropy - [CEIL_DIV(B * S, thr) blocks, 1D grid]
    {
        int op = 15;
        int bar_idx = 1 + (L * 10) + 2;
        int expected = CEIL_DIV(B * S * V, thr);
        int num_blocks = CEIL_DIV(B * S, thr);
 
        add_instructions_1d(all_instructions, &instruction_count, op, prev_op, -1, bar_idx, expected, num_blocks);
        prev_op = op;
    }
 
    printf("Forward pass instructions scheduled: %d\n", instruction_count);
 
    // OP 16: Cross-entropy backward - [CEIL_DIV(B * S, thr) blocks, 1D grid]
    {
        int op = 16;
        int bar_idx = 1 + (L * 10) + 3;
        int expected = CEIL_DIV(B * S, thr);
        int num_blocks = CEIL_DIV(B * S, thr);
 
        add_instructions_1d(all_instructions, &instruction_count, op, prev_op, -1, bar_idx, expected, num_blocks);
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
 
        add_instructions_2d(all_instructions, &instruction_count, op, prev_op, -1, bar_idx, expected, num_blocks_x, num_blocks_y);
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
 
        add_instructions_2d(all_instructions, &instruction_count, op, prev_op, -1, bar_idx, expected, num_blocks_x, num_blocks_y);
        prev_op = op;
    }
 
    // OP 19: Final LayerNorm backward - [B blocks, 1D grid]
    {
        int op = 19;
        int bar_idx = 1 + (L * 10) + 3 + 3;
        dim3 grid_prev = MLP_BACKWARD_WEIGHT_GRID(V, h);
        int expected = grid_prev.x * grid_prev.y;
        int num_blocks = B;
 
        add_instructions_1d(all_instructions, &instruction_count, op, prev_op, -1, bar_idx, expected, num_blocks);
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
 
            add_instructions_1d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks);
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
 
            add_instructions_2d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks_x, num_blocks_y);
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
 
            add_instructions_2d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks_x, num_blocks_y);
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
 
            add_instructions_1d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks);
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
 
            add_instructions_2d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks_x, num_blocks_y);
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
 
            add_instructions_2d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks_x, num_blocks_y);
            prev_op = op;
        }
 
        // OP 26: LayerNorm 2 backward - [B blocks, 1D grid]
        {
            int op = 26;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 5;
            dim3 grid_prev = MLP_BACKWARD_WEIGHT_GRID(h * 4, h);
            int expected = grid_prev.x * grid_prev.y;
            int num_blocks = B;
 
            add_instructions_1d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks);
            prev_op = op;
        }
 
 
        // OP 27: Residual backward (res_2) - [CEIL_DIV(B * S * h, thr) blocks, 1D grid]
        {
            int op = 27;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 6;
            int expected = B;
            int num_blocks = CEIL_DIV(B * S * h, thr);
 
            add_instructions_1d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks);
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
 
            add_instructions_2d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks_x, num_blocks_y);
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
 
            add_instructions_2d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks_x, num_blocks_y);
            prev_op = op;
        }
 
        // OP 30: Attention backward - [CEIL_DIV(B * S * n_head, thr) blocks, 1D grid]
        {
            int op = 30;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 9;
            dim3 grid_prev = MLP_BACKWARD_WEIGHT_GRID(h, h);
            int expected = grid_prev.x * grid_prev.y;
            int num_blocks = CEIL_DIV(B * S * n_head, thr);
 
            add_instructions_1d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks);
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
 
            add_instructions_2d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks_x, num_blocks_y);
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
 
            add_instructions_2d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks_x, num_blocks_y);
            prev_op = op;
        }
 
        // OP 33: LayerNorm 1 backward - [B blocks, 1D grid]
        {
            int op = 33;
            int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1 - layer_idx) * 14) + 12;
            dim3 grid_prev = MLP_BACKWARD_WEIGHT_GRID(h * 3, h);
            int expected = grid_prev.x * grid_prev.y;
            int num_blocks = B;
 
            add_instructions_1d(all_instructions, &instruction_count, op, prev_op, layer_idx, bar_idx, expected, num_blocks);
            prev_op = op;
        }
    }
 
    // OP 34: Embedding backward - [CEIL_DIV(B * S, thr) blocks, 1D grid]
    {
        int op = 34;
        int bar_idx = 1 + (L * 10) + 3 + 5 + ((L - 1) * 14) + 13;
        int expected = B;
        int num_blocks = CEIL_DIV(B * S, thr);
 
        add_instructions_1d(all_instructions, &instruction_count, op, prev_op, -1, bar_idx, expected, num_blocks);
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
 
    // Second pass: distribute instructions and set instr_idx
    int *sm_indices = (int *)calloc(NUM_SM, sizeof(int));
    for (int i = 0; i < instruction_count; i++) {
        int sm_id = i % NUM_SM;
        all_instructions[i].instr_idx = sm_indices[sm_id];  // Set the instruction index within this SM
        streams[sm_id]->instructions[sm_indices[sm_id]++] = all_instructions[i];
    }
 
    // Cleanup
    free(all_instructions);
    free(sm_counts);
    free(sm_indices);
 
    printf("Scheduled %d instructions across %d SMs\n", instruction_count, NUM_SM);
    for (int sm = 0; sm < NUM_SM; sm++) {
        printf("%d, ", streams[sm]->n);
    }
    printf("\n");

    // Write instructions per SM to temp.txt
    {
        const char* op_names[] = {
            "NONE",                      // 0
            "EMB_FWD",                   // 1
            "LN1_FWD",                   // 2
            "QKV_FWD",                   // 3
            "ATTN_FWD",                  // 4
            "ATTN_PROJ_FWD",             // 5
            "RES2_FWD",                  // 6
            "LN2_FWD",                   // 7
            "MLP_FC_FWD",                // 8
            "GELU_FWD",                  // 9
            "MLP_PROJ_FWD",              // 10
            "RES3_FWD",                  // 11
            "LN_F_FWD",                  // 12
            "LOGITS_FWD",                // 13
            "SOFTMAX_FWD",               // 14
            "CE_FWD",                    // 15
            "CE_BWD",                    // 16
            "LOGITS_BWD",                // 17
            "EMB_W_GRAD",                // 18
            "LN_F_BWD",                  // 19
            "RES3_BWD",                  // 20
            "MLP_PROJ_BWD_IN",           // 21
            "MLP_PROJ_BWD_W",            // 22
            "GELU_BWD",                  // 23
            "MLP_FC_BWD_IN",             // 24
            "MLP_FC_BWD_W",              // 25
            "LN2_BWD",                   // 26
            "RES2_BWD",                  // 27
            "ATTN_PROJ_BWD_IN",          // 28
            "ATTN_PROJ_BWD_W",           // 29
            "ATTN_BWD",                  // 30
            "QKV_BWD_IN",                // 31
            "QKV_BWD_W",                 // 32
            "LN1_BWD",                   // 33
            "EMB_BWD",                   // 34
        };
        int num_op_names = sizeof(op_names) / sizeof(op_names[0]);

        FILE *f = fopen("temp.txt", "w");
        if (f) {
            for (int sm = 0; sm < NUM_SM; sm++) {
                fprintf(f, "SM%d\n", sm);
                for (int i = 0; i < streams[sm]->n; i++) {
                    int op = streams[sm]->instructions[i].op;
                    if (op >= 0 && op < num_op_names) {
                        fprintf(f, "%d: %s\n", i, op_names[op]);
                    } else {
                        fprintf(f, "%d: OP%d\n", i, op);
                    }
                }
                fprintf(f, "\n");
            }
            fclose(f);
            printf("Wrote instructions per SM to temp.txt\n");
        }
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

__device__ __forceinline__ unsigned long long read_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
 
__global__ void megakernel(
    float *params,
    float *grads,

    float *acts,
    float *grad_acts,

    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,

    int *bar,
    stream_t **streams,
    long long *sm_start_times,
    long long *sm_end_times,
    long long *bar_enter_time,
    long long *bar_exit_time
) {
    int sm_id = blockIdx.x;  // Each SM gets its own block
    
    // Record start time using clock64() - only thread 0 records
    if (threadIdx.x == 0) {
        // sm_start_times[sm_id] = clock64();
        sm_start_times[sm_id] = read_globaltimer();
    }
    __syncthreads();
    
    stream_t *stream = streams[sm_id];
    execute_stream(params, grads, acts, grad_acts, seq_len, d_input_tokens, d_target_tokens, bar, stream, bar_enter_time, bar_exit_time);
    
    // Record end time - only thread 0 records
    __syncthreads();
    if (threadIdx.x == 0) {
        // sm_end_times[sm_id] = clock64();
        sm_end_times[sm_id] = read_globaltimer();
    }
}
 
 
__device__ void execute_stream(
    float *params,
    float *grads,

    float *acts,
    float *grad_acts,

    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,

    int *bar,
    stream_t *stream,
    long long *bar_enter_time,
    long long *bar_exit_time
) {
    // if (threadIdx.x == 0) {
    //     printf("SM %d starting execution of %d instructions\n", blockIdx.x, stream->n);
    // }
 
    int n = stream->n;
    for (int i = 0; i < n; i++) {
        instruction_t instr = stream->instructions[i];
        execute_instruction(params, grads, acts, grad_acts, seq_len, d_input_tokens, d_target_tokens, bar, instr, bar_enter_time, bar_exit_time);
    }
}
 
 
__device__ void execute_instruction(
    float *params,
    float *grads,

    float *acts,
    float *grad_acts,

    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,

    int *bar,
    instruction_t instr,
    long long *bar_enter_time,
    long long *bar_exit_time
) {
    int L = MK_N_LAYER;
    int B = MK_BATCH_SIZE;
    int S = seq_len;
    int h = MK_N_EMBD;
    int n_head = MK_N_HEAD;
    int V = MK_VOCAB_SIZE;

    int start_b_x = instr.start_b_x;
    int start_b_y = instr.start_b_y;
    int end_b_x = instr.end_b_x; 
    int end_b_y = instr.end_b_y;
    int op = instr.op;
    int expected = instr.expected;

    int sm_id = blockIdx.x;
    int timing_idx = sm_id * MAX_INSTR_PER_SM + instr.instr_idx;
 
    volatile int *vbar = (volatile int *)bar;
 
    // Shared memory for MLP operations (2 * TILE_SIZE * TILE_SIZE floats)
    extern __shared__ float shared_mem[];
 
    if (instr.op != 1) {
        // Record time before entering spin loop
        if (threadIdx.x == 0) {
            bar_enter_time[timing_idx] = read_globaltimer();
        }

        if (threadIdx.x == 0) {
            int exp = min(NUM_SM, expected);
            while (vbar[instr.bar_idx] < exp) {
                // __nanosleep(10);
            }
        }

        // Record time after exiting spin loop
        if (threadIdx.x == 0) {
            bar_exit_time[timing_idx] = read_globaltimer();
        }
 
        __syncthreads();
    } else {
        // For op 1 (no spin loop), record the same time for both
        if (threadIdx.x == 0) {
            long long t = read_globaltimer();
            bar_enter_time[timing_idx] = t;
            bar_exit_time[timing_idx] = t;
        }
        __syncthreads();
    }
 
    switch (op) {
        case 1: {
            // OP 1: Embedding forward
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                embedding_forward_device(MK_ACT_ENCODED(acts), d_input_tokens, MK_WTE(params), MK_WPE(params), S, h, V, MK_N_POSITIONS, b_x);
            }
            break;
        }
 
        case 2: {
            // OP 2: LayerNorm 1
            float *res = (instr.layer == 0) ? MK_ACT_ENCODED(acts) : MK_ACT_RES3(acts, instr.layer - 1);
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                layernorm_forward_device(MK_ACT_LN1(acts, instr.layer), res, MK_LN1_W(params, instr.layer), MK_LN1_B(params, instr.layer), MK_ACT_LN1_MEAN(acts, instr.layer), MK_ACT_LN1_RSTD(acts, instr.layer), S, h, b_x);
            }
            break;
        }
 
        case 3: {
            // OP 3: QKV projection
            dim3 grid = MLP_FORWARD_GRID(h * 3, B, S);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_forward_device(MK_ACT_QKV(acts, instr.layer), MK_ACT_LN1(acts, instr.layer), MK_QKV_W(params, instr.layer), MK_QKV_B(params, instr.layer), B, S, h, h * 3, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 4: {
            // OP 4: Attention
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                attention_forward_device(MK_ACT_ATTY(acts, instr.layer), MK_ACT_PREATT(acts, instr.layer), MK_ACT_ATT(acts, instr.layer), MK_ACT_QKV(acts, instr.layer), B, S, n_head, h, b_x);
            }
            break;
        }
 
        case 5: {
            // OP 5: Attention projection
            dim3 grid = MLP_FORWARD_GRID(h, B, S);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_forward_device(MK_ACT_ATT_PROJ(acts, instr.layer), MK_ACT_ATTY(acts, instr.layer), MK_ATTN_PROJ_W(params, instr.layer), MK_ATTN_PROJ_B(params, instr.layer), B, S, h, h, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 6: {
            // OP 6: Residual 2
            float *res = (instr.layer == 0) ? MK_ACT_ENCODED(acts) : MK_ACT_RES3(acts, instr.layer - 1);
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                residual_forward_device(MK_ACT_RES2(acts, instr.layer), MK_ACT_ATT_PROJ(acts, instr.layer), res, B, S, h, b_x);
            }
            break;
        }
 
        case 7: {
            // OP 7: LayerNorm 2
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                layernorm_forward_device(MK_ACT_LN2(acts, instr.layer), MK_ACT_RES2(acts, instr.layer), MK_LN2_W(params, instr.layer), MK_LN2_B(params, instr.layer), MK_ACT_LN2_MEAN(acts, instr.layer), MK_ACT_LN2_RSTD(acts, instr.layer), S, h, b_x);
            }
            break;
        }
 
        case 8: {
            // OP 8: MLP FC
            dim3 grid = MLP_FORWARD_GRID(h * 4, B, S);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_forward_device(MK_ACT_MLP_FC(acts, instr.layer), MK_ACT_LN2(acts, instr.layer), MK_MLP_FC_W(params, instr.layer), MK_MLP_FC_B(params, instr.layer), B, S, h, h * 4, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 9: {
            // OP 9: GELU
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                gelu_forward_device(MK_ACT_MLP_FC_GELU(acts, instr.layer), MK_ACT_MLP_FC(acts, instr.layer), B, S, h * 4, b_x);
            }
            break;
        }
 
        case 10: {
            // OP 10: MLP projection
            dim3 grid = MLP_FORWARD_GRID(h, B, S);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_forward_device(MK_ACT_MLP_PROJ(acts, instr.layer), MK_ACT_MLP_FC_GELU(acts, instr.layer), MK_MLP_PROJ_W(params, instr.layer), MK_MLP_PROJ_B(params, instr.layer), B, S, h * 4, h, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 11: {
            // OP 11: Residual 3
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                residual_forward_device(MK_ACT_RES3(acts, instr.layer), MK_ACT_MLP_PROJ(acts, instr.layer), MK_ACT_RES2(acts, instr.layer), B, S, h, b_x);
            }
            break;
        }
 
        case 12: {
            // OP 12: Final LayerNorm
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                layernorm_forward_device(MK_ACT_LN_F(acts), MK_ACT_RES3(acts, L - 1), MK_LN_F_W(params), MK_LN_F_B(params), MK_ACT_LN_F_MEAN(acts), MK_ACT_LN_F_RSTD(acts), S, h, b_x);
            }
            break;
        }
 
        case 13: {
            // OP 13: Logits
            dim3 grid = MLP_FORWARD_GRID(V, B, S);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_forward_device(MK_ACT_LOGITS(acts), MK_ACT_LN_F(acts), MK_WTE(params), NULL, B, S, h, V, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 14: {
            // OP 14: Softmax
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                softmax_forward_device(MK_ACT_PROBS(acts), MK_ACT_LOGITS(acts), B, S, V, b_x, shared_mem);
            }
            break;
        }
 
        case 15: {
            // OP 15: Cross-entropy forward
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                cross_entropy_forward_device(MK_ACT_LOSSES(acts), MK_ACT_PROBS(acts), d_target_tokens, B, S, V, b_x);
            }
            break;
        }
 
        case 16: {
            // OP 16: Cross-entropy backward
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                cross_entropy_backward_device(MK_ACT_LOGITS(grad_acts), MK_ACT_PROBS(acts), d_target_tokens, B, S, V, b_x);
            }
            break;
        }
 
        case 17: {
            // OP 17: Logits backward (input gradient)
            dim3 grid = MLP_BACKWARD_INPUT_GRID(h, B, S);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_backward_input_device(MK_ACT_LN_F(grad_acts), MK_ACT_LOGITS(grad_acts), MK_WTE(params), B, S, h, V, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 18: {
            // OP 18: Embedding weight gradient
            dim3 grid = MLP_BACKWARD_WEIGHT_GRID(V, h);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_backward_weight_device(MK_WTE(grads), NULL, MK_ACT_LOGITS(grad_acts), MK_ACT_LN_F(acts), B, S, h, V, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 19: {
            // OP 19: Final LayerNorm backward
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                layernorm_backward_device(MK_ACT_RES3(grad_acts, L - 1), MK_LN_F_W(grads), MK_LN_F_B(grads), MK_ACT_LN_F(grad_acts), MK_ACT_RES3(acts, L - 1), MK_LN_F_W(params), MK_ACT_LN_F_MEAN(acts), MK_ACT_LN_F_RSTD(acts), B, S, h, b_x);
            }
            break;
        }
 
        case 20: {
            // OP 20: Residual backward (res_3)
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                residual_backward_device(MK_ACT_RES2(grad_acts, instr.layer), MK_ACT_MLP_PROJ(grad_acts, instr.layer), MK_ACT_RES3(grad_acts, instr.layer), B * S * h, b_x);
            }
            break;
        }
 
        case 21: {
            // OP 21: MLP projection backward input
            dim3 grid = MLP_BACKWARD_INPUT_GRID(h * 4, B, S);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_backward_input_device(MK_ACT_MLP_FC_GELU(grad_acts, instr.layer), MK_ACT_MLP_PROJ(grad_acts, instr.layer), MK_MLP_PROJ_W(params, instr.layer), B, S, h * 4, h, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 22: {
            // OP 22: MLP projection backward weight
            dim3 grid = MLP_BACKWARD_WEIGHT_GRID(h, h * 4);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_backward_weight_device(MK_MLP_PROJ_W(grads, instr.layer), MK_MLP_PROJ_B(grads, instr.layer), MK_ACT_MLP_PROJ(grad_acts, instr.layer), MK_ACT_MLP_FC_GELU(acts, instr.layer), B, S, h * 4, h, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 23: {
            // OP 23: GELU backward
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                gelu_backward_device(MK_ACT_MLP_FC(grad_acts, instr.layer), MK_ACT_MLP_FC(acts, instr.layer), MK_ACT_MLP_FC_GELU(grad_acts, instr.layer), B * S * 4 * h, b_x);
            }
            break;
        }
 
        case 24: {
            // OP 24: MLP FC backward input
            dim3 grid = MLP_BACKWARD_INPUT_GRID(h, B, S);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_backward_input_device(MK_ACT_LN2(grad_acts, instr.layer), MK_ACT_MLP_FC(grad_acts, instr.layer), MK_MLP_FC_W(params, instr.layer), B, S, h, h * 4, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 25: {
            // OP 25: MLP FC backward weight
            dim3 grid = MLP_BACKWARD_WEIGHT_GRID(h * 4, h);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_backward_weight_device(MK_MLP_FC_W(grads, instr.layer), MK_MLP_FC_B(grads, instr.layer), MK_ACT_MLP_FC(grad_acts, instr.layer), MK_ACT_LN2(acts, instr.layer), B, S, h, h * 4, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 26: {
            // OP 26: LayerNorm 2 backward
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                layernorm_backward_device(MK_ACT_RES2(grad_acts, instr.layer), MK_LN2_W(grads, instr.layer), MK_LN2_B(grads, instr.layer), MK_ACT_LN2(grad_acts, instr.layer), MK_ACT_RES2(acts, instr.layer), MK_LN2_W(params, instr.layer), MK_ACT_LN2_MEAN(acts, instr.layer), MK_ACT_LN2_RSTD(acts, instr.layer), B, S, h, b_x);
            }
            break;
        }
 
        case 27: {
            // OP 27: Residual backward (res_2)
            float *g_res = (instr.layer == 0) ? MK_ACT_ENCODED(grad_acts) : MK_ACT_RES3(grad_acts, instr.layer - 1);
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                residual_backward_device(g_res, MK_ACT_ATT_PROJ(grad_acts, instr.layer), MK_ACT_RES2(grad_acts, instr.layer), B * S * h, b_x);
            }
            break;
        }
 
        case 28: {
            // OP 28: Attention projection backward input
            dim3 grid = MLP_BACKWARD_INPUT_GRID(h, B, S);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_backward_input_device(MK_ACT_ATTY(grad_acts, instr.layer), MK_ACT_ATT_PROJ(grad_acts, instr.layer), MK_ATTN_PROJ_W(params, instr.layer), B, S, h, h, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 29: {
            // OP 29: Attention projection backward weight
            dim3 grid = MLP_BACKWARD_WEIGHT_GRID(h, h);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_backward_weight_device(MK_ATTN_PROJ_W(grads, instr.layer), MK_ATTN_PROJ_B(grads, instr.layer), MK_ACT_ATT_PROJ(grad_acts, instr.layer), MK_ACT_ATTY(acts, instr.layer), B, S, h, h, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 30: {
            // OP 30: Attention backward
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                attention_backward_device(MK_ACT_QKV(grad_acts, instr.layer), MK_ACT_PREATT(grad_acts, instr.layer), MK_ACT_ATT(grad_acts, instr.layer), MK_ACT_ATTY(grad_acts, instr.layer), MK_ACT_QKV(acts, instr.layer), MK_ACT_ATT(acts, instr.layer), B, S, h, n_head, b_x);
            }
            break;
        }
 
        case 31: {
            // OP 31: QKV backward input
            dim3 grid = MLP_BACKWARD_INPUT_GRID(h, B, S);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_backward_input_device(MK_ACT_LN1(grad_acts, instr.layer), MK_ACT_QKV(grad_acts, instr.layer), MK_QKV_W(params, instr.layer), B, S, h, h * 3, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 32: {
            // OP 32: QKV backward weight
            dim3 grid = MLP_BACKWARD_WEIGHT_GRID(h * 3, h);
            int num_blocks_x = grid.x;
            int start_linear = start_b_y * num_blocks_x + start_b_x;
            int end_linear = end_b_y * num_blocks_x + end_b_x;
            for (int linear_idx = start_linear; linear_idx <= end_linear; linear_idx++) {
                int b_y = linear_idx / num_blocks_x;
                int b_x = linear_idx % num_blocks_x;
                mlp_backward_weight_device(MK_QKV_W(grads, instr.layer), MK_QKV_B(grads, instr.layer), MK_ACT_QKV(grad_acts, instr.layer), MK_ACT_LN1(acts, instr.layer), B, S, h, h * 3, b_x, b_y, shared_mem);
            }
            break;
        }
 
        case 33: {
            // OP 33: LayerNorm 1 backward
            float *g_res = (instr.layer == 0) ? MK_ACT_ENCODED(grad_acts) : MK_ACT_RES3(grad_acts, instr.layer - 1);
            float *res = (instr.layer == 0) ? MK_ACT_ENCODED(acts) : MK_ACT_RES3(acts, instr.layer - 1);
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                layernorm_backward_device(g_res, MK_LN1_W(grads, instr.layer), MK_LN1_B(grads, instr.layer), MK_ACT_LN1(grad_acts, instr.layer), res, MK_LN1_W(params, instr.layer), MK_ACT_LN1_MEAN(acts, instr.layer), MK_ACT_LN1_RSTD(acts, instr.layer), B, S, h, b_x);
            }
            break;
        }
 
        case 34: {
            // OP 34: Embedding backward
            for (int b_x = start_b_x; b_x <= end_b_x; b_x++) {
                embedding_backward_device(MK_WTE(grads), MK_WPE(grads), MK_ACT_ENCODED(grad_acts), d_input_tokens, B, S, h, b_x);
            }
            break;
        }
    }
 
    __syncthreads();

    // Only the instructions marked with `inc` perform the atomic increment.
    if (instr.inc && threadIdx.x == 0) {
        // printf("[INC]\n");
        atomicAdd(&bar[instr.bar_idx + 1], 1);
    }
}
