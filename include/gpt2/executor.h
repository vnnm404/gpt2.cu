#pragma once
#include <cuda_runtime_api.h>

namespace gpt2 {
enum class Code : int { gemm, add, gelu, gelu_backward, norm, norm_backward,
                       norm_parameters, embedding, embedding_backward,
                       attention, attention_backward, attention_kv_backward,
                       cross_entropy, sum_rows, adamw, clear, advance, sum_splits, pack };

// Device storage is owned by the caller; token and packed operand pointers
// are interpreted as int32 and BF16 respectively. GEMM flags: transpose A/B, add C;
// bit 3 marks three-plane BF16 operands on Hopper (otherwise operands are FP32).
// Bits 8+ hold the reduction partition count. group counts independent operations
// in a stage; only the first operation's group field is used by the executor.
struct Operation {
    Code code;
    int tiles, m, n, k, flags, group;
    float *p[8];
    float scalar[4];
};
static_assert(sizeof(Operation) == 112);
} // namespace gpt2

extern "C" {
int gpt2_gemm_tile_m();
int gpt2_gemm_tile_n();
int gpt2_occupancy(int *workers);
int gpt2_gemm_parameters(const gpt2::Operation *op, void *host_buffer);
int gpt2_operation(const gpt2::Operation *host_op, cudaStream_t stream);
int gpt2_launch(const gpt2::Operation *device_ops, int count, int workers, cudaStream_t stream);
}
