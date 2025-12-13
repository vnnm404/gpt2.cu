#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

#include "gpt2/layers/mlp.h"

static void bench_mlp_forward(nvbench::state &state) {
    // these can be nvbench axes
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

    // Report something
    state.add_global_memory_reads<uint8_t>(static_cast<std::size_t>(M) * K * sizeof(float) +
                                           static_cast<std::size_t>(K) * N * sizeof(float) +
                                           static_cast<std::size_t>(N) * sizeof(float));
    state.add_global_memory_writes<uint8_t>(static_cast<std::size_t>(M) * N * sizeof(float));

    // warp-tiled + double buffered kernel config
    constexpr int MLP_TILE = 32;

    // each block covers MLP_TILE x MLP_TILE output elements
    dim3 grid((N + MLP_TILE - 1) / MLP_TILE, (M + MLP_TILE - 1) / MLP_TILE);

    // 2x2 microtiling. for MLP_TILE=16: 8*8 = 64 threads
    constexpr int THREADS_PER_BLOCK = (MLP_TILE / 2) * (MLP_TILE / 2);
    dim3 block(THREADS_PER_BLOCK);

    // two double-buffered MLP_TILE x MLP_TILE tiles (input + weight)
    // 4x space for double buffering
    size_t smem = 4 * MLP_TILE * MLP_TILE * sizeof(float);

    state.exec([&](nvbench::launch &launch)
              //  { mlp_forward<TILE_SIZE><<<grid, block, smem, launch.get_stream()>>>(
               { mlp_forward<MLP_TILE><<<grid, block, smem, launch.get_stream()>>>(
                     d_out, d_in, d_w, d_b, B, S, K, N); });

    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
}

NVBENCH_BENCH(bench_mlp_forward)
  .add_int64_axis("S", nvbench::range(64, 64, 1)) // declare it so we can use it from cli later but disable
  .set_timeout(2);