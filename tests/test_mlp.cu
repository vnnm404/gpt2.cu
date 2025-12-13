#include "unity.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>

#include "gpt2/layers/mlp.h"
#define TOLERANCE 1e-4f

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void setUp(void) {}
void tearDown(void) {}

void test_mlp_forward_basic(void) {
    const int B = 1;
    const int S = 2;
    const int M = B * S;
    const int K = 4;   // input_dim
    const int N = 6;   // output_dim

    std::vector<float> h_inp(M * K);
    std::vector<float> h_w(K * N);
    std::vector<float> h_b(N);
    std::vector<float> h_out(M * N);
    std::vector<float> h_ref(M * N);

    // dummy inputs
    for (int i = 0; i < M*K; i++) h_inp[i] = float(i + 1); 
    for (int i = 0; i < K*N; i++) h_w[i]   = float(0.01 * (i+1));
    for (int i = 0; i < N; i++)    h_b[i]  = float(i * 0.1);

    // cpu ref: out = inp @ w + b
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += h_inp[m*K + k] * h_w[k*N + n];
            }
            h_ref[m*N + n] = acc + h_b[n];
        }
    }

    // device buffers
    float *d_inp, *d_w, *d_b, *d_out;
    gpuErrchk(cudaMalloc(&d_inp, M * K * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_w,   K * N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_b,   N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_out, M * N * sizeof(float)));

    gpuErrchk(cudaMemcpy(d_inp, h_inp.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_w,   h_w.data(),   K*N*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b,   h_b.data(),   N*sizeof(float),   cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_out, 0, M*N*sizeof(float)));

    // warp-tiled + double buffered kernel config
    // constexpr int MLP_TILE = 16;
    constexpr int MLP_TILE = 32;
    // constexpr int MLP_TILE = 64;
    
    // each block covers MLP_TILE x MLP_TILE output elements
    dim3 grid((N + MLP_TILE - 1) / MLP_TILE, (M + MLP_TILE - 1) / MLP_TILE);
    
    // 2x2 microtiling. for MLP_TILE=16: 8*8 = 64 threads
    constexpr int THREADS_PER_BLOCK = (MLP_TILE / 2) * (MLP_TILE / 2);
    dim3 block(THREADS_PER_BLOCK);
    
    // two double-buffered MLP_TILE x MLP_TILE tiles (input + weight)
    // 4x space for double buffering
    size_t smem = 4 * MLP_TILE * MLP_TILE * sizeof(float);

    mlp_forward<MLP_TILE><<<grid, block, smem>>>(
        d_out, d_inp, d_w, d_b,
        B, S, K, N
    );
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_out.data(), d_out, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    auto ok = true;
    for (int i = 0; i < M*N; i++) {
        if(fabs(h_ref[i] - h_out[i]) > TOLERANCE) {
            printf("Mismatch at index %d: ref=%f, out=%f\n", i, h_ref[i], h_out[i]);
            ok = false;
        }
    }
    TEST_ASSERT(ok);

    cudaFree(d_inp);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
}

// MLP backward weight --------------------------------------------------------------------------------
void init_array(float *arr, int size, float scale) {
    for (int i = 0; i < size; i++) {
        arr[i] = scale * ((float)rand() / RAND_MAX - 0.5f);
    }
}

bool arrays_match(const float *ref, const float *out, int size, float tol = TOLERANCE) {
    int mismatches = 0;
    for (int i = 0; i < size && mismatches < 10; i++) {
        float diff = fabsf(ref[i] - out[i]);
        float rel_diff = diff / (fabsf(ref[i]) + 1e-8f);
        if (diff > tol && rel_diff > tol) {
            printf("Mismatch at index %d: ref=%.6f, out=%.6f (diff=%.6f)\n", 
                   i, ref[i], out[i], diff);
            mismatches++;
        }
    }
    return mismatches == 0;
}

// Reference CPU implementation: g_weight = input^T @ g_out
void mlp_backward_weight_cpu(float *g_weight, float *g_bias, const float *g_out, 
                             const float *input, int batch_size, int seq_len, 
                             int input_dim, int output_dim) {
    int M = batch_size * seq_len;
    
    // Initialize outputs to zero
    for (int i = 0; i < input_dim * output_dim; i++) {
        g_weight[i] = 0.0f;
    }
    if (g_bias != NULL) {
        for (int i = 0; i < output_dim; i++) {
            g_bias[i] = 0.0f;
        }
    }
    
    // g_weight[k, n] = sum_m(input[m, k] * g_out[m, n])
    for (int k = 0; k < input_dim; k++) {
        for (int n = 0; n < output_dim; n++) {
            float sum = 0.0f;
            for (int m = 0; m < M; m++) {
                sum += input[m * input_dim + k] * g_out[m * output_dim + n];
            }
            g_weight[k * output_dim + n] = sum;
        }
    }
    
    // g_bias[n] = sum_m(g_out[m, n])
    if (g_bias != NULL) {
        for (int n = 0; n < output_dim; n++) {
            float sum = 0.0f;
            for (int m = 0; m < M; m++) {
                sum += g_out[m * output_dim + n];
            }
            g_bias[n] = sum;
        }
    }
}


// Test backward weight gradient
void test_mlp_backward_weight_basic() {
    const int B = 2;
    const int S = 4;
    const int K = 128;  // input_dim
    const int N = 256;  // output_dim
    const int M = B * S;
    
    // Allocate host memory
    float *h_g_out = (float*)malloc(M * N * sizeof(float));
    float *h_input = (float*)malloc(M * K * sizeof(float));
    float *h_g_weight_ref = (float*)malloc(K * N * sizeof(float));
    float *h_g_weight_out = (float*)malloc(K * N * sizeof(float));
    float *h_g_bias_ref = (float*)malloc(N * sizeof(float));
    float *h_g_bias_out = (float*)malloc(N * sizeof(float));
    
    // Initialize inputs
    srand(43);
    init_array(h_g_out, M * N, 1.0f);
    init_array(h_input, M * K, 1.0f);
    
    // Compute reference on CPU
    mlp_backward_weight_cpu(h_g_weight_ref, h_g_bias_ref, h_g_out, h_input, B, S, K, N);
    
    // Allocate device memory
    float *d_g_out, *d_input, *d_g_weight, *d_g_bias;
    cudaMalloc(&d_g_out, M * N * sizeof(float));
    cudaMalloc(&d_input, M * K * sizeof(float));
    cudaMalloc(&d_g_weight, K * N * sizeof(float));
    cudaMalloc(&d_g_bias, N * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_g_out, h_g_out, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_g_weight, 0, K * N * sizeof(float));
    cudaMemset(d_g_bias, 0, N * sizeof(float));
    
    // Launch kernel
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (K + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE * TILE_SIZE);
    
    mlp_backward_weight<<<grid, block>>>(d_g_weight, d_g_bias, d_g_out, d_input, B, S, K, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        TEST_ASSERT(false);
    }
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        TEST_ASSERT(false);
    }
    
    // Copy results back
    cudaMemcpy(h_g_weight_out, d_g_weight, K * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_bias_out, d_g_bias, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare
    printf("Testing weight gradients...\n");
    bool weight_passed = arrays_match(h_g_weight_ref, h_g_weight_out, K * N);
    
    printf("Testing bias gradients...\n");
    bool bias_passed = arrays_match(h_g_bias_ref, h_g_bias_out, N);
    
    // Cleanup
    free(h_g_out);
    free(h_input);
    free(h_g_weight_ref);
    free(h_g_weight_out);
    free(h_g_bias_ref);
    free(h_g_bias_out);
    cudaFree(d_g_out);
    cudaFree(d_input);
    cudaFree(d_g_weight);
    cudaFree(d_g_bias);
    
    TEST_ASSERT(weight_passed);
    TEST_ASSERT(bias_passed);
}

// MLP backward input --------------------------------------------------------------------------------

// Reference CPU implementation: g_input = g_out @ weight^T
void mlp_backward_input_cpu(float *g_input, const float *g_out, const float *weight,
                            int batch_size, int seq_len, int input_dim, int output_dim) {
    int M = batch_size * seq_len;
    
    // Initialize g_input to zero
    for (int i = 0; i < M * input_dim; i++) {
        g_input[i] = 0.0f;
    }
    
    // g_input[m, k] = sum_n(g_out[m, n] * weight[k, n])
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < input_dim; k++) {
            float sum = 0.0f;
            for (int n = 0; n < output_dim; n++) {
                sum += g_out[m * output_dim + n] * weight[k * output_dim + n];
            }
            g_input[m * input_dim + k] = sum;
        }
    }
}

// Test backward input gradient
void test_mlp_backward_input_basic() {
    const int B = 2;
    const int S = 4;
    const int K = 128;  // input_dim
    const int N = 256;  // output_dim
    const int M = B * S;
    
    // Allocate host memory
    float *h_g_out = (float*)malloc(M * N * sizeof(float));
    float *h_weight = (float*)malloc(K * N * sizeof(float));
    float *h_g_input_ref = (float*)malloc(M * K * sizeof(float));
    float *h_g_input_out = (float*)malloc(M * K * sizeof(float));
    
    // Initialize inputs
    srand(42);
    init_array(h_g_out, M * N, 1.0f);
    init_array(h_weight, K * N, 1.0f);
    
    // Compute reference on CPU
    mlp_backward_input_cpu(h_g_input_ref, h_g_out, h_weight, B, S, K, N);
    
    // Allocate device memory
    float *d_g_out, *d_weight, *d_g_input;
    cudaMalloc(&d_g_out, M * N * sizeof(float));
    cudaMalloc(&d_weight, K * N * sizeof(float));
    cudaMalloc(&d_g_input, M * K * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_g_out, h_g_out, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_g_input, 0, M * K * sizeof(float));
    
    // Launch kernel
    dim3 grid((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE * TILE_SIZE);
    
    mlp_backward_input<<<grid, block>>>(d_g_input, d_g_out, d_weight, B, S, K, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        TEST_ASSERT(false);
    }
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        TEST_ASSERT(false);
    }
    
    // Copy result back
    cudaMemcpy(h_g_input_out, d_g_input, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare
    bool passed = arrays_match(h_g_input_ref, h_g_input_out, M * K);
    
    // Cleanup
    free(h_g_out);
    free(h_weight);
    free(h_g_input_ref);
    free(h_g_input_out);
    cudaFree(d_g_out);
    cudaFree(d_weight);
    cudaFree(d_g_input);
    
    TEST_ASSERT(passed);
}

// MLP backward GPT2 sizes -------------------------------------------------------------------------
void test_mlp_backward_gpt2_small() {
    const int B = 4;
    const int S = 64;
    const int K = 768;
    const int N = 3072;
    const int M = B * S;
    
    printf("Testing backward pass with GPT-2 Small dimensions (B=%d, S=%d, K=%d, N=%d)...\n", 
           B, S, K, N);
    
    // Allocate host memory
    float *h_g_out = (float*)malloc(M * N * sizeof(float));
    float *h_input = (float*)malloc(M * K * sizeof(float));
    float *h_weight = (float*)malloc(K * N * sizeof(float));
    float *h_g_input_out = (float*)malloc(M * K * sizeof(float));
    
    // Initialize inputs
    srand(44);
    init_array(h_g_out, M * N, 0.01f);
    init_array(h_input, M * K, 0.01f);
    init_array(h_weight, K * N, 0.01f);
    
    // Allocate device memory
    float *d_g_out, *d_input, *d_weight, *d_g_input;
    cudaMalloc(&d_g_out, M * N * sizeof(float));
    cudaMalloc(&d_input, M * K * sizeof(float));
    cudaMalloc(&d_weight, K * N * sizeof(float));
    cudaMalloc(&d_g_input, M * K * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_g_out, h_g_out, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_g_input, 0, M * K * sizeof(float));
    
    // Launch backward input kernel
    dim3 grid_input((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE * TILE_SIZE);
    
    mlp_backward_input<<<grid_input, block>>>(d_g_input, d_g_out, d_weight, B, S, K, N);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Backward input kernel error: %s\n", cudaGetErrorString(err));
        TEST_ASSERT(false);
    }
    
    // Just verify it doesnt crash for now
    cudaMemcpy(h_g_input_out, d_g_input, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check for NaN or Inf
    bool valid = true;
    for (int i = 0; i < M * K && valid; i++) {
        if (isnan(h_g_input_out[i]) || isinf(h_g_input_out[i])) {
            printf("Invalid value at index %d: %.6f\n", i, h_g_input_out[i]);
            valid = false;
        }
    }
    
    // Cleanup
    free(h_g_out);
    free(h_input);
    free(h_weight);
    free(h_g_input_out);
    cudaFree(d_g_out);
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_g_input);
    
    printf("GPT-2 smoke test: %s\n", valid ? "PASSED" : "FAILED");
    TEST_ASSERT(valid);
}

int main() {
    UNITY_BEGIN();
    
    // Forward pass tests
    RUN_TEST(test_mlp_forward_basic);
    
    // Backward pass tests
    RUN_TEST(test_mlp_backward_input_basic);
    RUN_TEST(test_mlp_backward_weight_basic);
    RUN_TEST(test_mlp_backward_gpt2_small);
    
    return UNITY_END();
}
