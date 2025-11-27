#ifndef GPT2_GPT2_H
#define GPT2_GPT2_H

/* GPT-2 model structures and functions will be defined here */

#include "tensor.h"

#define NUM_PARAMETER_TENSORS 16
#define NUM_ACTIVATION_TENSORS 23

#define NUM_LAYERS 12
#define NUM_HEADS 12
#define EMBEDDING_SIZE 768
#define CONTEXT_SIZE 1024
#define VOCAB_SIZE 50257
#define PADDED_VOCAB_SIZE 50304

typedef struct
{
    int vocab_size;  // vocab size, e.g. 50257
    int batch_size;  // batch size, e.g. 1
    int n_layer;     // number of layers, e.g. 12
    int n_head;      // number of heads in attention, e.g. 12
    int n_embd;      // number of channels, e.g. 768
    int n_positions; // max sequence length, e.g. 1024
    int n_ctx;       // context size, e.g. 1024
} config_t;

typedef struct
{
    tensor_t *wte; // (h, V)
    tensor_t *wpe; // (maxT, h)
} emb_t;

typedef struct
{
    tensor_t *w; // (h)
    tensor_t *b; // (h)
} ln_t;

typedef struct
{
    tensor_t *qkv_w;  // (h, 3*h)
    tensor_t *qkv_b;  // (3*h)
    tensor_t *proj_w; // (h, h)
    tensor_t *proj_b; // (h)
} attn_t;

typedef struct
{
    tensor_t *fc_w;   // (h, 4*h)
    tensor_t *fc_b;   // (4*h)
    tensor_t *proj_w; // (4*h, h)
    tensor_t *proj_b; // (h)
} ffn_t;

typedef struct
{
    ln_t ln_1;
    attn_t attn;
    ln_t ln_2;
    ffn_t mlp;
} block_t;

typedef struct
{
    emb_t emb;
    block_t h[NUM_LAYERS];
    ln_t ln_f;
    float *params_memory; // base pointer for contiguous parameter memory
    size_t num_parameters; // total number of parameters
} gpt2_t;

int gpt2_initialize(gpt2_t *model, const config_t *config);
int gpt2_load_weights(gpt2_t *model, FILE *file);
void gpt2_free(gpt2_t *model);

#endif /* GPT2_GPT2_H */