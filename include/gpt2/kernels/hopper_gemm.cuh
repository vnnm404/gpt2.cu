#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include "program.cuh"

namespace gpt2 {
// Three BF16 components approximate each FP32 operand. Six products retain
// high*high, both high*middle terms, middle*middle, and both high*low terms.
// Separate FP32 accumulators keep small corrections from being rounded away.
// Tensor maps describe the packed planes; split-K only changes coordinates.
struct alignas(128) HopperParameters {
    CUtensorMap maps[6];
};

__host__ __device__ inline bool hopper_eligible(const Operation &o) {
    return (o.flags & 8) && o.m % 4 == 0 && o.n % 4 == 0 && o.k % 8 == 0 &&
           (reinterpret_cast<uintptr_t>(o.p[0]) & 15) == 0 &&
           (reinterpret_cast<uintptr_t>(o.p[1]) & 15) == 0;
}

inline int hopper_parameters(const Operation &o, void *buffer) {
    if (!buffer) return sizeof(HopperParameters);
    HopperParameters params{};
    for (int map = 0; map < 6; ++map) {
        int operand = map % 2, part = map / 2;
        int rows = operand == 0 ? o.m : o.n;
        int tile = operand == 0 ? tile_m : tile_n;
        uint64_t dimensions[2] = {uint64_t(o.k), uint64_t(rows)};
        uint64_t stride = dimensions[0] * sizeof(uint16_t);
        uint32_t box[2] = {32, uint32_t(tile)}, steps[2] = {1, 1};
        auto *pointer = reinterpret_cast<uint16_t *>(o.p[operand]) + size_t(part) * rows * o.k;
        CUresult error = cuTensorMapEncodeTiled(&params.maps[map],
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pointer, dimensions, &stride,
            box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (error != CUDA_SUCCESS) return int(error);
    }
    std::memcpy(buffer, &params, sizeof(params));
    return 0;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// Convert each operand once per GEMM, rather than once per output tile.
// A padded shared tile transposes global operands without bank conflicts.
__device__ __forceinline__ void hopper_pack(const Operation &o, int tile, float *s) {
    int columns = (o.n + 31) / 32;
    int row = tile / columns * 32, col = tile % columns * 32;
    int i = threadIdx.x * 4, r = i / 32, k = i % 32;
    float4 values = make_float4(0, 0, 0, 0);
    bool transpose = o.flags & 1;
    int source_row = transpose ? col + r : row + r;
    int source_col = transpose ? row + k : col + k;
    int source_rows = transpose ? o.n : o.m, source_columns = transpose ? o.m : o.n;
    const float *source = o.p[0] + source_row * source_columns + source_col;
    if (source_row < source_rows) {
        if (source_col + 3 < source_columns && (reinterpret_cast<uintptr_t>(source) & 15) == 0)
            values = *reinterpret_cast<const float4 *>(source);
        else {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                if (source_col + j < source_columns) reinterpret_cast<float *>(&values)[j] = source[j];
        }
    }
    if (transpose) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) s[r * 33 + k + j] = reinterpret_cast<float *>(&values)[j];
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 4; ++j) reinterpret_cast<float *>(&values)[j] = s[(k + j) * 33 + r];
    }
    if (row + r >= o.m || col + k >= o.n) return;
    uint2 high_bits = make_uint2(0, 0), mid_bits = make_uint2(0, 0), low_bits = make_uint2(0, 0);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float x = reinterpret_cast<float *>(&values)[j];
        auto high = __float2bfloat16_rn(x);
        float residual = x - __bfloat162float(high);
        auto middle = __float2bfloat16_rn(residual);
        auto low = __float2bfloat16_rn(residual - __bfloat162float(middle));
        reinterpret_cast<uint32_t *>(&high_bits)[j / 2] |= uint32_t(__bfloat16_as_ushort(high)) << (16 * (j % 2));
        reinterpret_cast<uint32_t *>(&mid_bits)[j / 2] |= uint32_t(__bfloat16_as_ushort(middle)) << (16 * (j % 2));
        reinterpret_cast<uint32_t *>(&low_bits)[j / 2] |= uint32_t(__bfloat16_as_ushort(low)) << (16 * (j % 2));
    }
    size_t count = size_t(o.m) * o.n;
    size_t index = size_t(row + r) * o.n + col + k;
    auto *out = reinterpret_cast<uint16_t *>(o.p[1]);
    if (col + k + 3 < o.n && (reinterpret_cast<uintptr_t>(out + index) & 7) == 0 && count % 4 == 0) {
        *reinterpret_cast<uint2 *>(out + index) = high_bits;
        *reinterpret_cast<uint2 *>(out + count + index) = mid_bits;
        *reinterpret_cast<uint2 *>(out + 2 * count + index) = low_bits;
    } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) if (col + k + j < o.n) {
            out[index + j] = reinterpret_cast<uint16_t *>(&high_bits)[j];
            out[count + index + j] = reinterpret_cast<uint16_t *>(&mid_bits)[j];
            out[2 * count + index + j] = reinterpret_cast<uint16_t *>(&low_bits)[j];
        }
    }
}

__device__ __forceinline__ void hopper_gemm(const Operation &o, int task, float *scratch) {
    using namespace cute;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    constexpr int K = 32, A = tile_m * K, B = tile_n * K;
    constexpr int pipe_elements = 3 * (A + B);
    static_assert(3 * pipe_elements * sizeof(bfloat16_t) + 3 * sizeof(uint64_t) <= shared_bytes);
    auto *storage = reinterpret_cast<bfloat16_t *>(scratch);
    auto *barriers = reinterpret_cast<uint64_t *>(storage + 3 * pipe_elements);
    int nc = (o.n + tile_n - 1) / tile_n;
    int nr = (o.m + tile_m - 1) / tile_m;
    int count = nr * nc, split = task / count, tile = task % count;
    constexpr int row_group = 32;
    int first = tile / (row_group * nc) * row_group;
    int rows = min(row_group, nr - first), within = tile % (row_group * nc);
    int row = (first + within % rows) * tile_m, col = (within / rows) * tile_n;
    int splits = max(1, o.flags >> 8);
    int span = ((o.k + splits * 32 - 1) / (splits * 32)) * 32;
    int begin = split * span, length = min(span, o.k - begin);
    int stages = (length + K - 1) / K;
    const auto &params = *reinterpret_cast<const HopperParameters *>(o.p[7]);
    if (threadIdx.x == 0) {
        Barrier::init(barriers, 1);
        Barrier::init(barriers + 1, 1);
        Barrier::init(barriers + 2, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    auto load = [&](int stage) {
        int pipe = stage % 3, kk = begin + stage * K;
        auto *buffer = storage + pipe * pipe_elements;
        Barrier::arrive_and_expect_tx(barriers + pipe, pipe_elements * sizeof(bfloat16_t));
        #pragma unroll
        for (int part = 0; part < 3; ++part) {
            SM90_TMA_LOAD_2D::copy(&params.maps[2 * part], barriers + pipe, 0,
                buffer + part * (A + B), kk, row);
            SM90_TMA_LOAD_2D::copy(&params.maps[2 * part + 1], barriers + pipe, 0,
                buffer + part * (A + B) + A, kk, col);
        }
    };
    if (threadIdx.x == 0) {
        if (stages > 0) load(0);
        if (stages > 1) load(1);
        if (stages > 2) load(2);
    }
    auto layout_a = tile_to_shape(GMMA::Layout_K_SW64_Atom<bfloat16_t>{}, Shape<Int<tile_m>, Int<K>>{});
    auto layout_b = tile_to_shape(GMMA::Layout_K_SW64_Atom<bfloat16_t>{}, Shape<Int<tile_n>, Int<K>>{});
    auto mma = make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}, Layout<Shape<_1,_2,_1>>{});
    auto thread = mma.get_thread_slice(threadIdx.x);
    auto coords = thread.partition_C(make_identity_tensor(Shape<Int<tile_m>, Int<tile_n>>{}));
    auto acc = thread.make_fragment_C(coords);
    auto correction = thread.make_fragment_C(coords);
    clear(acc);
    clear(correction);
    for (int stage = 0; stage < stages; ++stage) {
        int pipe = stage % 3;
        Barrier::wait(barriers + pipe, (stage / 3) & 1);
        auto *hi = storage + pipe * pipe_elements, *mid = hi + A + B, *lo = mid + A + B;
        auto ah = make_tensor(make_smem_ptr(hi), layout_a);
        auto am = make_tensor(make_smem_ptr(mid), layout_a);
        auto al = make_tensor(make_smem_ptr(lo), layout_a);
        auto bh = make_tensor(make_smem_ptr(hi + A), layout_b);
        auto bm = make_tensor(make_smem_ptr(mid + A), layout_b);
        auto bl = make_tensor(make_smem_ptr(lo + A), layout_b);
        auto a_hi = thread.make_fragment_A(thread.partition_A(ah));
        auto a_mid = thread.make_fragment_A(thread.partition_A(am));
        auto a_lo = thread.make_fragment_A(thread.partition_A(al));
        auto b_hi = thread.make_fragment_B(thread.partition_B(bh));
        auto b_mid = thread.make_fragment_B(thread.partition_B(bm));
        auto b_lo = thread.make_fragment_B(thread.partition_B(bl));
        #pragma unroll
        for (int i = 0; i < size(acc); ++i) {
            warpgroup_fence_operand(acc(i));
            warpgroup_fence_operand(correction(i));
        }
        warpgroup_arrive();
        gemm(mma, a_mid, b_mid, correction);
        gemm(mma, a_mid, b_hi, correction);
        gemm(mma, a_hi, b_mid, correction);
        gemm(mma, a_lo, b_hi, correction);
        gemm(mma, a_hi, b_lo, correction);
        gemm(mma, a_hi, b_hi, acc);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        #pragma unroll
        for (int i = 0; i < size(acc); ++i) {
            warpgroup_fence_operand(acc(i));
            warpgroup_fence_operand(correction(i));
        }
        // Both warp groups must finish before TMA can overwrite this stage.
        __syncthreads();
        if (threadIdx.x == 0 && stage + 3 < stages) load(stage + 3);
    }
    for (int i = 0; i < size(acc); ++i) {
        int r = row + get<0>(coords(i)), c = col + get<1>(coords(i));
        if (r < o.m && c < o.n) {
            int index = split * o.m * o.n + r * o.n + c;
            float value = acc(i) + correction(i);
            if (o.p[3]) value += o.p[3][c];
            if (o.flags & 4) value += o.p[2][index];
            o.p[2][index] = value;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(cast_smem_ptr_to_uint(barriers)) : "memory");
        asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(cast_smem_ptr_to_uint(barriers + 1)) : "memory");
        asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(cast_smem_ptr_to_uint(barriers + 2)) : "memory");
    }
}
#endif
} // namespace gpt2
