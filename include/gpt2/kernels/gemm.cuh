#pragma once
#include <cstring>
#include "program.cuh"
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace gpt2 {
// CUTLASS supplies an inline threadblock implementation, not a nested kernel
// launch. Logical tile coordinates come from the persistent worker's current
// task, never from its physical block index.
struct WorkerSwizzle {
    CUTLASS_HOST_DEVICE int get_log_tile(cutlass::gemm::GemmCoord) const { return 0; }
    CUTLASS_DEVICE cutlass::gemm::GemmCoord get_tile_offset(int) const {
        extern __shared__ float scratch[];
        const int *coords = reinterpret_cast<const int *>(scratch);
        return {coords[0], coords[1], 0};
    }
};

template<bool TA, bool TB>
using CutlassGemm = typename cutlass::gemm::kernel::DefaultGemm<
    float, typename std::conditional<TA, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type, 1,
    float, typename std::conditional<TB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type, 1,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<tile_m, tile_n, 8>, cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    WorkerSwizzle, 2, false, cutlass::arch::OpMultiplyAdd>::GemmKernel;

template<bool TA, bool TB>
__device__ __forceinline__ void gemm(const Operation &o, int task, float *s) {
    using Kernel = CutlassGemm<TA, TB>;
    static_assert(Kernel::kThreadCount == threads);
    static_assert(sizeof(typename Kernel::SharedStorage) + 128 <= shared_bytes);
    int nc = (o.n + tile_n - 1) / tile_n, count = ((o.m + tile_m - 1) / tile_m) * nc;
    int split = task / count, tile = task % count;
    if (threadIdx.x == 0) {
        // Visit a small group of output rows across columns. Nearby workers
        // reuse the same weight tiles in L2, especially for the vocabulary head.
        int nr = (o.m + tile_m - 1) / tile_m;
        int first_row = tile / (8 * nc) * 8;
        int rows = min(8, nr - first_row), within = tile % (8 * nc);
        reinterpret_cast<int *>(s)[0] = first_row + within % rows;
        reinterpret_cast<int *>(s)[1] = within / rows;
    }
    __syncthreads();
    const auto *params = reinterpret_cast<const typename Kernel::Params *>(o.p[7]);
    auto &storage = *reinterpret_cast<typename Kernel::SharedStorage *>(s + 32);
    Kernel{}(params[split], storage);
}

template<bool TA, bool TB>
int gemm_parameters(const Operation &o, void *buffer) {
    using Kernel = CutlassGemm<TA, TB>;
    using LayoutA = typename Kernel::Mma::IteratorA::Layout;
    using LayoutB = typename Kernel::Mma::IteratorB::Layout;
    int splits = max(1, o.flags >> 8);
    int span = ((o.k + splits * 32 - 1) / (splits * 32)) * 32;
    if (!buffer) return splits * sizeof(typename Kernel::Params);
    for (int split = 0; split < splits; ++split) {
        int begin = split * span, length = min(span, o.k - begin);
        float *a = o.p[0] + (TA ? begin * o.m : begin);
        float *b = o.p[1] + (TB ? begin : begin * o.n);
        float *out = o.p[2] + split * o.m * o.n;
        float *source = o.p[3] ? o.p[3] : out;
        float beta = o.p[3] || (o.flags & 4) ? 1.f : 0.f;
        typename Kernel::Params p({o.m, o.n, length}, {(o.m + tile_m - 1) / tile_m, (o.n + tile_n - 1) / tile_n, 1},
            {a, LayoutA(TA ? o.m : o.k)}, {b, LayoutB(TB ? o.k : o.n)},
            {source, cutlass::layout::RowMajor(o.p[3] ? 0 : o.n)},
            {out, cutlass::layout::RowMajor(o.n)}, {1.f, beta});
        std::memcpy(static_cast<char *>(buffer) + split * sizeof(p), &p, sizeof(p));
    }
    return 0;
}
} // namespace gpt2
