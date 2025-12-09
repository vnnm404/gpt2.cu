#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#include "gpt2/mk.h"
#include "gpt2/gpt2.h"
#include "gpt2/train.h"

#include "gpt2/layers/embedding.h"
#include "gpt2/layers/layernorm.h"
#include "gpt2/layers/mlp.h"
#include "gpt2/layers/attention.h"
#include "gpt2/layers/residual.h"
#include "gpt2/layers/gelu.h"
#include "gpt2/layers/softmax.h"
#include "gpt2/layers/cross_entropy.h"
#include "gpt2/layers/adamw.h"

// Include layer implementations directly for proper inlining and optimization
#include "layers/embedding.cu"
#include "layers/layernorm.cu"
#include "layers/mlp.cu"
#include "layers/attention.cu"
#include "layers/residual.cu"
#include "layers/gelu.cu"
#include "layers/softmax.cu"
#include "layers/cross_entropy.cu"
#include "layers/adamw.cu"

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

// Local helpers
static inline void gpuErrchk_internal(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}
#define gpuErrchk(ans) { gpuErrchk_internal((ans), __FILE__, __LINE__); }

// Helper functions to build instruction lists
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

stream_t** schedule_instructions(config_t config, stream_t **streams, int seq_len) {
    int L = config.n_layer;
    int B = config.batch_size;
    int S = seq_len;
    int h = config.n_embd;
    int n_head = config.n_head;
    int V = config.vocab_size;
    int thr = 1024;

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

    // (Backward ops omitted here for brevity but included below)
    // ... continue building instructions for backward (copied from test file)

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

    // Populate inc flags
    for (int i = 0; i < instruction_count; i++) {
        all_instructions[i].inc = 0;
    }

    if (instruction_count > 0) {
        int run_start = 0;
        int run_op = all_instructions[0].op;

        for (int i = 1; i <= instruction_count; i++) {
            int cur_op = (i < instruction_count) ? all_instructions[i].op : -9999;
            if (i == instruction_count || cur_op != run_op) {
                int run_end = i - 1;
                int run_len = run_end - run_start + 1;

                if (run_op >= 0 && run_op != 34) {
                    int to_mark = (run_len < NUM_SM) ? run_len : NUM_SM;
                    int start_idx = run_end - to_mark + 1;
                    for (int j = start_idx; j <= run_end; j++) {
                        if (j >= 0 && j < instruction_count) {
                            all_instructions[j].inc = 1;
                        }
                    }
                }

                if (i < instruction_count) {
                    run_start = i;
                    run_op = cur_op;
                }
            }
        }
    }

    // Distribute instructions to SMs in round-robin fashion
    int *sm_counts = (int *)calloc(NUM_SM, sizeof(int));
 
    for (int i = 0; i < instruction_count; i++) {
        int sm_id = i % NUM_SM;
        sm_counts[sm_id]++;
    }
 
    // Allocate host stream structs
    stream_t *host_stream_structs[NUM_SM];
    // instruction_t *host_instructions_ptrs[NUM_SM];
    for (int sm = 0; sm < NUM_SM; sm++) {
        if (sm_counts[sm] > 0) {
            stream_t *st = (stream_t*)malloc(sizeof(stream_t));
            st->n = sm_counts[sm];
            st->instructions = (instruction_t*)malloc(sm_counts[sm] * sizeof(instruction_t));
            host_stream_structs[sm] = st;
        } else {
            host_stream_structs[sm] = NULL;
        }
    }

    int *sm_indices = (int *)calloc(NUM_SM, sizeof(int));
    for (int i = 0; i < instruction_count; i++) {
        int sm_id = i % NUM_SM;
        instruction_t instr = all_instructions[i];
        instr.instr_idx = sm_indices[sm_id];
        host_stream_structs[sm_id]->instructions[sm_indices[sm_id]++] = instr;
    }

    free(all_instructions);
    free(sm_counts);
    free(sm_indices);

    // Allocate device memory for streams and instructions
    stream_t **d_streams_ptr;
    stream_t *d_stream_structs[NUM_SM];
 
    gpuErrchk(cudaMalloc(&d_streams_ptr, NUM_SM * sizeof(stream_t*)));
 
    for (int sm = 0; sm < NUM_SM; sm++) {
        stream_t *d_stream;
        instruction_t *d_instructions = NULL;
 
        gpuErrchk(cudaMalloc(&d_stream, sizeof(stream_t)));
 
        if (host_stream_structs[sm] && host_stream_structs[sm]->n > 0) {
            gpuErrchk(cudaMalloc(&d_instructions, host_stream_structs[sm]->n * sizeof(instruction_t)));
            gpuErrchk(cudaMemcpy(d_instructions, host_stream_structs[sm]->instructions, host_stream_structs[sm]->n * sizeof(instruction_t), cudaMemcpyHostToDevice));
        }

        stream_t temp_stream = { host_stream_structs[sm] ? host_stream_structs[sm]->n : 0, d_instructions };
        gpuErrchk(cudaMemcpy(d_stream, &temp_stream, sizeof(stream_t), cudaMemcpyHostToDevice));

        d_stream_structs[sm] = d_stream;
    }

    // Copy device pointers array
    gpuErrchk(cudaMemcpy(d_streams_ptr, d_stream_structs, NUM_SM * sizeof(stream_t*), cudaMemcpyHostToDevice));

    return d_streams_ptr;
}

void free_schedule(stream_t **d_streams_ptr) {
    // Copy device stream pointers back to host
    stream_t *d_stream_structs[NUM_SM];
    gpuErrchk(cudaMemcpy(d_stream_structs, d_streams_ptr, NUM_SM * sizeof(stream_t*), cudaMemcpyDeviceToHost));

    for (int sm = 0; sm < NUM_SM; sm++) {
        stream_t temp_stream;
        gpuErrchk(cudaMemcpy(&temp_stream, d_stream_structs[sm], sizeof(stream_t), cudaMemcpyDeviceToHost));

        if (temp_stream.instructions != NULL) {
            gpuErrchk(cudaFree(temp_stream.instructions));
        }

        gpuErrchk(cudaFree(d_stream_structs[sm]));
    }

    gpuErrchk(cudaFree(d_streams_ptr));
}

__device__ __forceinline__ unsigned long long read_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Megakernel and device-side helpers
__global__ void megakernel(
    float *params,
    float *grads,

    float *acts,
    float *grad_acts,

    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,

#ifdef PROFILE
    long long *sm_start_times,
    long long *sm_end_times,
    long long *bar_enter_time,
    long long *bar_exit_time,
    long long *instr_end_time,
#endif

    int *bar,
    stream_t **streams
) {
    int sm_id = blockIdx.x;

#ifdef PROFILE
    if (threadIdx.x == 0) {
        sm_start_times[sm_id] = read_globaltimer();
    }
    __syncthreads();
#endif

    stream_t *stream = streams[sm_id];

    execute_stream(
        params, grads, acts, grad_acts, seq_len, d_input_tokens, d_target_tokens, 
#ifdef PROFILE
        bar_enter_time, bar_exit_time, instr_end_time,
#endif
        bar, stream
    );

#ifdef PROFILE
    __syncthreads();
    if (threadIdx.x == 0) {
        sm_end_times[sm_id] = read_globaltimer();
    }
#endif
}

__device__ void execute_stream(
    float *params,
    float *grads,

    float *acts,
    float *grad_acts,

    int seq_len,
    const int *d_input_tokens,
    const int *d_target_tokens,

#ifdef PROFILE
    long long *bar_enter_time,
    long long *bar_exit_time,
    long long *instr_end_time,
#endif

    int *bar,
    stream_t *stream
) {
    int n = stream->n;
    for (int i = 0; i < n; i++) {
        instruction_t instr = stream->instructions[i];
        execute_instruction(params, grads, acts, grad_acts, seq_len, d_input_tokens, d_target_tokens, 
#ifdef PROFILE
            bar_enter_time, bar_exit_time, instr_end_time,
#endif
            bar, instr
        );
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

#ifdef PROFILE
    long long *bar_enter_time,
    long long *bar_exit_time,
    long long *instr_end_time,
#endif

    int *bar,
    instruction_t instr
) {
    int L = MK_N_LAYER;
    int B = MK_BATCH_SIZE;
    int S = seq_len;
    int n_head = MK_N_HEAD;
    int h = MK_N_EMBD;
    int V = MK_VOCAB_SIZE;

    int start_b_x = instr.start_b_x;
    int start_b_y = instr.start_b_y;
    int end_b_x = instr.end_b_x; 
    int end_b_y = instr.end_b_y;
    int op = instr.op;
    int expected = instr.expected;

#ifdef PROFILE
    int sm_id = blockIdx.x;
    int timing_idx = sm_id * MAX_INSTR_PER_SM + instr.instr_idx;
#endif
 
    volatile int *vbar = (volatile int *)bar;
 
    extern __shared__ float shared_mem[];
 
    if (instr.op != 1) {
#ifdef PROFILE
        if (threadIdx.x == 0) {
            bar_enter_time[timing_idx] = read_globaltimer();
        }
#endif

        if (threadIdx.x == 0) {
            int exp = min(NUM_SM, expected);
            while (vbar[instr.bar_idx] < exp) {
            }
        }

#ifdef PROFILE
        if (threadIdx.x == 0) {
            bar_exit_time[timing_idx] = read_globaltimer();
        }
#endif
 
        __syncthreads();
    } else {
#ifdef PROFILE
        if (threadIdx.x == 0) {
            long long t = read_globaltimer();
            bar_enter_time[timing_idx] = t;
            bar_exit_time[timing_idx] = t;
        }
        __syncthreads();
#endif
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

#ifdef PROFILE
    if (threadIdx.x == 0) {
        instr_end_time[timing_idx] = read_globaltimer();
    }
#endif

    if (instr.inc && threadIdx.x == 0) {
        atomicAdd(&bar[instr.bar_idx + 1], 1);
    }
}
