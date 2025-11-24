#ifndef GPT2_GPT2_H
#define GPT2_GPT2_H

/* GPT-2 model structures and functions will be defined here */

#define NUM_PARAMETER_TENSORS 16
#define NUM_ACTIVATION_TENSORS 23

typedef struct
{
    int max_seq_len;       // max sequence length, e.g. 1024
    int vocab_size;        // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers;        // number of layers, e.g. 12
    int num_heads;         // number of heads in attention, e.g. 12
    int channels;          // number of channels, e.g. 768
} GPT2Config;

typedef struct
{
    float *wte;      // (V, C)
    float *wpe;      // (maxT, C)
    float *ln1w;     // (L, C)
    float *ln1b;     // (L, C)
    float *qkvw;     // (L, 3*C, C)
    float *qkvb;     // (L, 3*C)
    float *attprojw; // (L, C, C)
    float *attprojb; // (L, C)
    float *ln2w;     // (L, C)
    float *ln2b;     // (L, C)
    float *fcw;      // (L, 4*C, C)
    float *fcb;      // (L, 4*C)
    float *fcprojw;  // (L, C, 4*C)
    float *fcprojb;  // (L, C)
    float *lnfw;     // (C)
    float *lnfb;     // (C)
} ParameterTensors;

typedef struct
{
    float *encoded;   // (B, T, C)
    float *ln1;       // (L, B, T, C)
    float *ln1_mean;  // (L, B, T)
    float *ln1_rstd;  // (L, B, T)
    float *qkv;       // (L, B, T, 3*C)
    float *atty;      // (L, B, T, C)
    float *preatt;    // (L, B, NH, T, T)
    float *att;       // (L, B, NH, T, T)
    float *attproj;   // (L, B, T, C)
    float *residual2; // (L, B, T, C)
    float *ln2;       // (L, B, T, C)
    float *ln2_mean;  // (L, B, T)
    float *ln2_rstd;  // (L, B, T)
    float *fch;       // (L, B, T, 4*C)
    float *fch_gelu;  // (L, B, T, 4*C)
    float *fcproj;    // (L, B, T, C)
    float *residual3; // (L, B, T, C)
    float *lnf;       // (B, T, C)
    float *lnf_mean;  // (B, T)
    float *lnf_rstd;  // (B, T)
    float *logits;    // (B, T, V)
    float *probs;     // (B, T, V)
    float *losses;    // (B, T)
} ActivationTensors;

typedef struct
{
    GPT2Config config;

    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float *params_memory;
    size_t num_parameters;

    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float *acts_memory;
    size_t num_activations;
} GPT2;

#endif /* GPT2_GPT2_H */