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

#include "gpt2/mk.h"
#include "gpt2/train.h"

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

// Instances of training buffers and optimizer state used by the test
train_buffers_t buffers;
train_buffers_t g_buffers;
adamw_state_t opt_state;

int *bar;  // [B, 1 + (L * 10 + 3) + 1 + (5 + L * 13 + 1)] global atomics
stream_t *streams[NUM_SM];  // Host streams
stream_t **d_streams;  // Device streams array

long long *d_sm_start_times;  // Start time for each SM (clock64)
long long *d_sm_end_times;    // End time for each SM (clock64)
long long *d_bar_enter_time;   // Time before entering spin loop
long long *d_bar_exit_time;    // Time after exiting spin loop
long long *d_instr_end_time;   // Time after instruction execution completes
int *h_instr_counts;           // Number of instructions per SM (for output)

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
    printf("Model Params: %ld\n", model.num_parameters);
    
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
    d_streams = schedule_instructions(config, streams, T);
    setup_train_buffers(config, &buffers, T);
    setup_train_buffers(config, &g_buffers, T);

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

    // Allocate per-instruction timing buffers for bar enter/exit times and instruction end times
    gpuErrchk(cudaMalloc(&d_bar_enter_time, NUM_SM * MAX_INSTR_PER_SM * sizeof(long long)));
    gpuErrchk(cudaMalloc(&d_bar_exit_time, NUM_SM * MAX_INSTR_PER_SM * sizeof(long long)));
    gpuErrchk(cudaMalloc(&d_instr_end_time, NUM_SM * MAX_INSTR_PER_SM * sizeof(long long)));
    long long *h_bar_enter_time = (long long *)malloc(NUM_SM * MAX_INSTR_PER_SM * sizeof(long long));
    long long *h_bar_exit_time = (long long *)malloc(NUM_SM * MAX_INSTR_PER_SM * sizeof(long long));
    long long *h_instr_end_time = (long long *)malloc(NUM_SM * MAX_INSTR_PER_SM * sizeof(long long));

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
        zero_activation_grads(config, &g_buffers);
        gpuErrchk(cudaMemset(bar, 0, bar_size * sizeof(int)));
        
        // forward(config, model, buffers, d_input_tokens, T);
        // cross_entropy(config, buffers, d_target_tokens, T);
        // backward(config, model, buffers, g_model, g_buffers, d_input_tokens, d_target_tokens, T);


        megakernel<<<NUM_SM, threads_per_block, shared_mem_size>>>(
            model.params_memory,
            g_model.params_memory,
            buffers.activations_memory,
            g_buffers.activations_memory,
            T,
            d_input_tokens,
            d_target_tokens,

#ifdef PROFILE
            d_sm_start_times,
            d_sm_end_times,
            d_bar_enter_time,
            d_bar_exit_time,
            d_instr_end_time,
#endif

            bar,
            d_streams
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
        
        // printf("SM Timing (step %d):\n", step);
        // for (int sm = 0; sm < NUM_SM; sm++) {
        //     long long rel_start = h_sm_start_times[sm] - min_start;
        //     long long rel_end = h_sm_end_times[sm] - min_start;
        //     long long duration = h_sm_end_times[sm] - h_sm_start_times[sm];
        //     printf("  SM %2d: start=%12lld, end=%12lld, duration=%12lld ms\n", sm, rel_start, rel_end, duration / 1000000);
        // }

        // Copy per-instruction timing data and write to timer.txt
        gpuErrchk(cudaMemcpy(h_bar_enter_time, d_bar_enter_time, NUM_SM * MAX_INSTR_PER_SM * sizeof(long long), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_bar_exit_time, d_bar_exit_time, NUM_SM * MAX_INSTR_PER_SM * sizeof(long long), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_instr_end_time, d_instr_end_time, NUM_SM * MAX_INSTR_PER_SM * sizeof(long long), cudaMemcpyDeviceToHost));

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
                        long long end = h_instr_end_time[idx];
                        long long spin_duration = exit - enter;
                        long long exec_duration = end - exit;
                        fprintf(f, "  instr %d: bar_enter=%lld, bar_exit=%lld, instr_end=%lld, spin_wait=%lld, exec_time=%lld\n", 
                                i, enter - min_start, exit - min_start, end - min_start, spin_duration / 1000000, exec_duration / 1000000);
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
    cudaFree(d_instr_end_time);
    free(h_bar_enter_time);
    free(h_bar_exit_time);
    free(h_instr_end_time);
    free(h_instr_counts);
    gpt2_free(&model);
    gpt2_free(&g_model);
    
    return allok ? 0 : 1;
}
