#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "gpt2/gpt2.h"
#include "gpt2/tensor.h"

#define MAX_INSTR_PER_SM 2000
#define NUM_SM 28


typedef struct
{
    int op;
    int prev_op;
    int layer;
    int start_b_x, end_b_x;
    int start_b_y, end_b_y;

    int bar_idx;
    int expected;
    int inc;
    int instr_idx;
} instruction_t;

typedef struct
{
    int n;
    instruction_t *instructions;
} stream_t;

stream_t **schedule_instructions(config_t config, stream_t **streams, int seq_len);
void free_schedule(stream_t **d_streams_ptr);
void schedule_get_host_instr_counts(int *out_counts);

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

    float *m_memory,
    float *v_memory,
    int t,

    int *bar,
    stream_t **streams);

__device__ __forceinline__ void execute_stream(
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

    float *m_memory,
    float *v_memory,
    int t,

    int *bar,
    stream_t *stream);

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

    float *m_memory,
    float *v_memory,
    int t,

    int *bar,
    instruction_t instr);
