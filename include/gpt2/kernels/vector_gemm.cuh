#pragma once
#include "program.cuh"
#include <cutlass/gemm/threadblock/mma_pipelined.h>
#include <cutlass/gemm/kernel/gemm.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/pitch_linear_thread_map.h>

namespace gpt2 {
// A vector global load may need a scalar transpose when entering the SIMT
// shared-memory layout. Keep that mapping separate from the warp MMA policy.
template<bool A, bool Transpose>
struct VectorSharedStore {
    using Element = float;
    using Layout = typename std::conditional<A, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type;
    using TensorRef = cutlass::TensorRef<float, Layout>;
    using Fragment = cutlass::Array<float, (A ? tile_m : tile_n) * 8 / threads>;
    float *pointer;
    int stride, linear;
    __device__ __forceinline__ VectorSharedStore(TensorRef ref, int thread) : pointer(ref.data()), stride(ref.stride(0)), linear(thread * 4) {}
    __device__ __forceinline__ VectorSharedStore &operator++() { pointer += 8 * stride; return *this; }
    __device__ __forceinline__ void add_tile_offset(cutlass::MatrixCoord offset) {
        pointer += (A ? offset.column() : offset.row()) * 8 * stride;
    }
    __device__ __forceinline__ void store(const Fragment &fragment) {
        #pragma unroll
        for (int v = 0; v < Fragment::kElements / 4; ++v) {
            int at = linear + v * threads * 4;
            int row = A ? (Transpose ? at % tile_m : at / 8) : (Transpose ? at % 8 : at / tile_n);
            int col = A ? (Transpose ? at / tile_m : at % 8) : (Transpose ? at / 8 : at % tile_n);
            float *out = pointer + (A ? col * stride + row : row * stride + col);
            if constexpr (A == Transpose) {
                *reinterpret_cast<float4 *>(out) = make_float4(fragment[v * 4], fragment[v * 4 + 1], fragment[v * 4 + 2], fragment[v * 4 + 3]);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) out[j * stride] = fragment[v * 4 + j];
            }
        }
    }
};

template<bool TA, bool TB, class Base, class Swizzle>
struct VectorGemmConfig {
    using Shape = cutlass::gemm::GemmShape<tile_m, tile_n, 8>;
    using LayoutA = typename std::conditional<TA, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type;
    using LayoutB = typename std::conditional<TB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type;
    using MapA = cutlass::transform::PitchLinearStripminedThreadMap<cutlass::layout::PitchLinearShape<TA ? tile_m : 8, TA ? 8 : tile_m>, threads, 4>;
    using MapB = cutlass::transform::PitchLinearStripminedThreadMap<cutlass::layout::PitchLinearShape<TB ? 8 : tile_n, TB ? tile_n : 8>, threads, 4>;
    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<tile_m, 8>, float, LayoutA, 1, MapA, 4>;
    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<8, tile_n>, float, LayoutB, 0, MapB, 4>;
    using Mma = cutlass::gemm::threadblock::MmaPipelined<Shape, IteratorA, VectorSharedStore<true, TA>, IteratorB,
        VectorSharedStore<false, TB>, float, cutlass::layout::RowMajor, typename Base::Mma::Policy>;
    using Kernel = cutlass::gemm::kernel::Gemm<Mma, typename Base::Epilogue, Swizzle, false>;
};

__host__ __device__ inline bool vector_gemm_eligible(const Operation &o) {
    if ((o.m | o.n | o.k) & 3) return false;
    return ((reinterpret_cast<uintptr_t>(o.p[0]) | reinterpret_cast<uintptr_t>(o.p[1])) & 15) == 0;
}
} // namespace gpt2
