#pragma once
#include <cuda_runtime_api.h>

namespace gpt2 {
enum class Code : int { gemm, add, gelu, gelu_backward, norm, norm_backward,
                       norm_parameters, embedding, embedding_backward,
                       attention, attention_backward, attention_kv_backward,
                       cross_entropy, sum_rows, adamw, clear, advance, sum_splits, tile_graph, page_norm };

// Device pointers are owned by the caller. GEMM flags: transpose A/B, add C;
// bits 8+ hold the reduction partition count. group counts independent operations
// in a stage; only the first operation's group field is used by the executor.
// tile_graph is an internal cooperative instruction: p[0..7] hold operations,
// tasks, edges, ready queue, dependency counters, queue control, roots, trace.
// page_norm uses two CTA-local pages; flags is rows/task and k selects warp roles.
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
