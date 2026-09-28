#pragma once
#include "gemm.cuh"

namespace gpt2 {
// Each worker owns a tile of query/key rows from one attention head. All
// intermediates retain [B,T,H,T] layout; QKV and dQKV retain [B,T,3,C].
using AttentionScore = CutlassGemm<false, true>;
using AttentionValue = CutlassGemm<false, false>;
using AttentionGradient = CutlassGemm<true, false>;
struct AttentionParameters {
    AttentionScore::Params score;
    AttentionValue::Params value;
};
struct AttentionKVParameters {
    AttentionGradient::Params key, value;
};

template<class Kernel>
__device__ __forceinline__ void attention_product(const typename Kernel::Params &params,
                                                 int row, int column, float *scratch) {
    static_assert(Kernel::kThreadCount == threads);
    static_assert(sizeof(typename Kernel::SharedStorage) + 128 <= shared_bytes);
    if (threadIdx.x == 0) {
        reinterpret_cast<int *>(scratch)[0] = row;
        reinterpret_cast<int *>(scratch)[1] = column;
    }
    __syncthreads();
    auto &storage = *reinterpret_cast<typename Kernel::SharedStorage *>(scratch + 32);
    Kernel{}(params, storage);
    __syncthreads();
}

template<bool Backward>
__device__ __forceinline__ void attention_rows(const Operation &o, int task, float *scratch) {
    int row_tiles = (o.k + tile_m - 1) / tile_m;
    int head = task / row_tiles, row_tile = task % row_tiles;
    const auto &params = reinterpret_cast<const AttentionParameters *>(o.p[7])[head];
    for (int column = 0; column < (o.k + tile_n - 1) / tile_n; ++column)
        attention_product<AttentionScore>(params.score, row_tile, column, scratch);
    int heads = o.n / 64, batch = head / heads, h = head % heads;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int q = row_tile * tile_m + warp; q < min(o.k, (row_tile + 1) * tile_m); q += threads / 32) {
        int base = ((batch * o.k + q) * heads + h) * o.k;
        if constexpr (Backward) {
            float dot = 0;
            for (int j = lane; j <= q; j += 32) dot += o.p[2][base + j] * o.p[4][base + j];
            dot = warp_sum(dot);
            for (int j = lane; j < o.k; j += 32)
                o.p[4][base + j] = j <= q ? o.p[2][base + j] * (o.p[4][base + j] - dot) * 0.125f : 0.f;
        } else {
            float mx = -INFINITY, sum = 0;
            for (int j = lane; j <= q; j += 32) mx = fmaxf(mx, o.p[2][base + j] * 0.125f);
            mx = warp_max(mx);
            for (int j = lane; j <= q; j += 32) sum += expf(o.p[2][base + j] * 0.125f - mx);
            sum = warp_sum(sum);
            for (int j = lane; j < o.k; j += 32)
                o.p[2][base + j] = j <= q ? expf(o.p[2][base + j] * 0.125f - mx) / sum : 0.f;
        }
    }
    // The following product reads only this CTA's completed rows. No grid
    // barrier is needed between score, softmax, and value/query products.
    __syncthreads();
    attention_product<AttentionValue>(params.value, row_tile, 0, scratch);
}

__device__ __forceinline__ void attention(const Operation &o, int task, float *scratch) {
    attention_rows<false>(o, task, scratch);
}
__device__ __forceinline__ void attention_backward(const Operation &o, int task, float *scratch) {
    attention_rows<true>(o, task, scratch);
}
__device__ __forceinline__ void attention_kv_backward(const Operation &o, int task, float *scratch) {
    int row_tiles = (o.k + tile_m - 1) / tile_m;
    const auto &params = reinterpret_cast<const AttentionKVParameters *>(o.p[7])[task / row_tiles];
    int row = task % row_tiles;
    // dK/dV require all dScores rows. The outer dependency scheduler publishes
    // that stage before any worker reaches this instruction.
    attention_product<AttentionGradient>(params.key, row, 0, scratch);
    attention_product<AttentionGradient>(params.value, row, 0, scratch);
}

// Params contain strided tensor references and are prepared once on the host.
// This avoids rebuilding iterator state or descriptors inside the megakernel.
template<class Kernel>
inline typename Kernel::Params attention_params(int m, int n, int k,
                                                float *a, int lda, float *b, int ldb,
                                                float *out, int ldc) {
    using A = typename Kernel::Mma::IteratorA::Layout;
    using B = typename Kernel::Mma::IteratorB::Layout;
    return typename Kernel::Params({m, n, k}, {(m + tile_m - 1) / tile_m, (n + tile_n - 1) / tile_n, 1},
        {a, A(lda)}, {b, B(ldb)}, {out, cutlass::layout::RowMajor(ldc)},
        {out, cutlass::layout::RowMajor(ldc)}, {1.f, 0.f});
}
inline int attention_parameters(const Operation &o, void *buffer) {
    int heads = o.n / 64, batches = o.m / o.k;
    bool kv = o.code == Code::attention_kv_backward;
    size_t bytes = kv ? sizeof(AttentionKVParameters) : sizeof(AttentionParameters);
    if (!buffer) return batches * heads * bytes;
    for (int batch = 0; batch < batches; ++batch) for (int head = 0; head < heads; ++head) {
        int qkv = batch * o.k * 3 * o.n + head * 64;
        int out = batch * o.k * o.n + head * 64;
        int scores = (batch * o.k * heads + head) * o.k;
        float *q = o.p[0] + qkv, *k = q + o.n, *v = k + o.n;
        float *probabilities = o.p[2] + scores;
        char *destination = static_cast<char *>(buffer) + (batch * heads + head) * bytes;
        if (kv) {
            AttentionKVParameters p{
                attention_params<AttentionGradient>(o.k, 64, o.k, o.p[4] + scores, heads * o.k, q, 3 * o.n, o.p[3] + qkv + o.n, 3 * o.n),
                attention_params<AttentionGradient>(o.k, 64, o.k, probabilities, heads * o.k, o.p[1] + out, o.n, o.p[3] + qkv + 2 * o.n, 3 * o.n)};
            std::memcpy(destination, &p, sizeof(p));
        } else if (o.code == Code::attention_backward) {
            AttentionParameters p{
                attention_params<AttentionScore>(o.k, o.k, 64, o.p[1] + out, o.n, v, 3 * o.n, o.p[4] + scores, heads * o.k),
                attention_params<AttentionValue>(o.k, 64, o.k, o.p[4] + scores, heads * o.k, k, 3 * o.n, o.p[3] + qkv, 3 * o.n)};
            std::memcpy(destination, &p, sizeof(p));
        } else {
            AttentionParameters p{
                attention_params<AttentionScore>(o.k, o.k, 64, q, 3 * o.n, k, 3 * o.n, probabilities, heads * o.k),
                attention_params<AttentionValue>(o.k, 64, o.k, probabilities, heads * o.k, v, 3 * o.n, o.p[1] + out, o.n)};
            std::memcpy(destination, &p, sizeof(p));
        }
    }
    return 0;
}
} // namespace gpt2
