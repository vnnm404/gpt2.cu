#include "unity.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include "gpt2/layers/mlp.h"

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

#ifdef MLP_WMMA
    // warp-tiled kernel config
    // constexpr int MLP_TILE = 16;
    constexpr int MLP_TILE = 32;
    
    // each block covers MLP_TILE x MLP_TILE output elements
    dim3 grid((N + MLP_TILE - 1) / MLP_TILE, (M + MLP_TILE - 1) / MLP_TILE);
    
    // 2x2 microtiling. for MLP_TILE=16: 8*8 = 64 threads
    constexpr int THREADS_PER_BLOCK = (MLP_TILE / 2) * (MLP_TILE / 2);
    dim3 block(THREADS_PER_BLOCK);
    
#ifdef MLP_DOUBLE_BUFFER
    // two double-buffered MLP_TILE x MLP_TILE tiles (input + weight)
    // 4x space for double buffering
    size_t smem = 4 * MLP_TILE * MLP_TILE * sizeof(float);
#else
    // two MLP_TILE x MLP_TILE tiles (input + weight)
    size_t smem = 2 * MLP_TILE * MLP_TILE * sizeof(float);
#endif
#else
    // og kernel config
    constexpr int MLP_TILE = TILE_SIZE;  // 32
    
    // each block covers TILE_SIZE x TILE_SIZE output elements
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // one thread per output element
    dim3 block(TILE_SIZE * TILE_SIZE);
    
    // two TILE_SIZE x TILE_SIZE tiles
    size_t smem = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
#endif

    mlp_forward<MLP_TILE><<<grid, block, smem>>>(
        d_out, d_inp, d_w, d_b,
        B, S, K, N
    );
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_out.data(), d_out, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    auto ok = true;
    for (int i = 0; i < M*N; i++) {
        if(fabs(h_ref[i] - h_out[i]) > 1e-4) {
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

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_mlp_forward_basic);
    return UNITY_END();
}
