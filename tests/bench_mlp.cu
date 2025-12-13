#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

#include "gpt2/layers/mlp.h"

static void bench_mlp_forward(nvbench::state &state) {
    const int B = 4;
    const int S = state.get_int64("S");
    const int M = B * S;
    const int K = 768;
    const int N = 3072;

    float *d_in = nullptr;
    float *d_w = nullptr;
    float *d_b = nullptr;
    float *d_out = nullptr;

    cudaMalloc(&d_in, M * K * sizeof(float));
    cudaMalloc(&d_w, K * N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out, M * N * sizeof(float));

    state.add_global_memory_reads<uint8_t>(static_cast<std::size_t>(M) * K * sizeof(float) +
                                           static_cast<std::size_t>(K) * N * sizeof(float) +
                                           static_cast<std::size_t>(N) * sizeof(float));
    state.add_global_memory_writes<uint8_t>(static_cast<std::size_t>(M) * N * sizeof(float));

    constexpr int MLP_TILE = 32;
    dim3 grid((N + MLP_TILE - 1) / MLP_TILE, (M + MLP_TILE - 1) / MLP_TILE);
    constexpr int THREADS_PER_BLOCK = (MLP_TILE / 2) * (MLP_TILE / 2);
    dim3 block(THREADS_PER_BLOCK);
    size_t smem = 4 * MLP_TILE * MLP_TILE * sizeof(float);

    state.exec([&](nvbench::launch &launch) {
        mlp_forward<MLP_TILE><<<grid, block, smem, launch.get_stream()>>>(
            d_out, d_in, d_w, d_b, B, S, K, N);
    });

    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
}

static void bench_mlp_backward_input(nvbench::state &state) {
    const int B = 4;
    const int S = state.get_int64("S");
    const int M = B * S;
    const int K = 768;
    const int N = 3072;

    float *d_g_input = nullptr;
    float *d_g_out = nullptr;
    float *d_weight = nullptr;

    cudaMalloc(&d_g_input, M * K * sizeof(float));
    cudaMalloc(&d_g_out, M * N * sizeof(float));
    cudaMalloc(&d_weight, K * N * sizeof(float));

    state.add_global_memory_reads<uint8_t>(static_cast<std::size_t>(M) * N * sizeof(float) +
                                           static_cast<std::size_t>(K) * N * sizeof(float));
    state.add_global_memory_writes<uint8_t>(static_cast<std::size_t>(M) * K * sizeof(float));

    constexpr int MLP_TILE = 32;
    dim3 grid((K + MLP_TILE - 1) / MLP_TILE, (M + MLP_TILE - 1) / MLP_TILE);
    constexpr int THREADS_PER_BLOCK = (MLP_TILE / 2) * (MLP_TILE / 2);
    dim3 block(THREADS_PER_BLOCK);
    size_t smem = 4 * MLP_TILE * MLP_TILE * sizeof(float);

    state.exec([&](nvbench::launch &launch) {
        mlp_backward_input<MLP_TILE><<<grid, block, smem, launch.get_stream()>>>(
            d_g_input, d_g_out, d_weight, B, S, K, N);
    });

    cudaFree(d_g_input);
    cudaFree(d_g_out);
    cudaFree(d_weight);
}

static void bench_mlp_backward_weight(nvbench::state &state) {
    const int B = 4;
    const int S = state.get_int64("S");
    const int M = B * S;
    const int K = 768;
    const int N = 3072;

    float *d_g_weight = nullptr;
    float *d_g_bias = nullptr;
    float *d_g_out = nullptr;
    float *d_input = nullptr;

    cudaMalloc(&d_g_weight, K * N * sizeof(float));
    cudaMalloc(&d_g_bias, N * sizeof(float));
    cudaMalloc(&d_g_out, M * N * sizeof(float));
    cudaMalloc(&d_input, M * K * sizeof(float));

    state.add_global_memory_reads<uint8_t>(static_cast<std::size_t>(M) * N * sizeof(float) +
                                           static_cast<std::size_t>(M) * K * sizeof(float));
    state.add_global_memory_writes<uint8_t>(static_cast<std::size_t>(K) * N * sizeof(float) +
                                            static_cast<std::size_t>(N) * sizeof(float));

    constexpr int MLP_TILE = 32;
    dim3 grid((N + MLP_TILE - 1) / MLP_TILE, (K + MLP_TILE - 1) / MLP_TILE);
    constexpr int THREADS_PER_BLOCK = (MLP_TILE / 2) * (MLP_TILE / 2);
    dim3 block(THREADS_PER_BLOCK);
    size_t smem = 4 * MLP_TILE * MLP_TILE * sizeof(float);

    state.exec([&](nvbench::launch &launch) {
        mlp_backward_weight<MLP_TILE><<<grid, block, smem, launch.get_stream()>>>(
            d_g_weight, d_g_bias, d_g_out, d_input, B, S, K, N);
    });

    cudaFree(d_g_weight);
    cudaFree(d_g_bias);
    cudaFree(d_g_out);
    cudaFree(d_input);
}

NVBENCH_BENCH(bench_mlp_forward)
    .add_int64_axis("S", nvbench::range(64, 2048, 64))
    .set_timeout(10);

NVBENCH_BENCH(bench_mlp_backward_input)
    .add_int64_axis("S", nvbench::range(64, 2048, 64))
    .set_timeout(10);

NVBENCH_BENCH(bench_mlp_backward_weight)
    .add_int64_axis("S", nvbench::range(64, 2048, 64))
    .set_timeout(10);