#include <cooperative_groups.h>
#include "gpt2/kernels/gemm.cuh"
#include "gpt2/kernels/elementwise.cuh"
#include "gpt2/kernels/attention.cuh"
#include "gpt2/kernels/page_pipeline.cuh"
#include "gpt2/kernels/tile_schedule.cuh"

namespace gpt2 {
__device__ __forceinline__ void execute(const Operation &o, int tile, float *s) {
    if (o.code == Code::gemm) {
        switch (o.flags & 3) {
            case 0: gemm<false, false>(o, tile, s); break;
            case 1: gemm<true, false>(o, tile, s); break;
            case 2: gemm<false, true>(o, tile, s); break;
            case 3: gemm<true, true>(o, tile, s); break;
        }
    } else switch (o.code) {
        case Code::page_norm: page_norm(o, tile, s); break;
        case Code::adamw: adamw(o, tile); break;
        case Code::norm: norm(o, tile, s); break;
        case Code::norm_backward: norm_backward(o, tile, s); break;
        case Code::norm_parameters: case Code::sum_rows: sum_rows(o, tile, s); break;
        case Code::attention: attention(o, tile, s); break;
        case Code::attention_backward: attention_backward(o, tile, s); break;
        case Code::attention_kv_backward: attention_kv_backward(o, tile); break;
        case Code::cross_entropy: cross_entropy(o, tile, s); break;
        default: elementwise(o, tile); break;
    }
}

__global__ __launch_bounds__(256, 2) void standalone(Operation op) {
    extern __shared__ float scratch[];
    execute(op, blockIdx.x, scratch);
}

__global__ __launch_bounds__(256, 2) void persistent(const Operation *ops, int count) {
    extern __shared__ float scratch[];
    auto grid = cooperative_groups::this_grid();
    for (int i = 0; i < count;) {
        if (ops[i].code == Code::tile_graph) {
            tile_graph(ops[i], scratch);
            grid.sync();
            ++i;
            continue;
        }
        int end = min(count, i + max(1, ops[i].group));
        int tiles = 0;
        for (int j = i; j < end; ++j) tiles += ops[j].tiles;
        // All workers participate, even when there are fewer tiles than blocks.
        // A stage can contain independent operations. Workers share the entire
        // tile space, so short bias/reduction work overlaps longer GEMMs.
        for (int task = blockIdx.x; task < tiles; task += gridDim.x) {
            int index = i, tile = task;
            while (index + 1 < end && tile >= ops[index].tiles) tile -= ops[index++].tiles;
            const Operation op = ops[index];
            execute(op, tile, scratch);
            __syncthreads();
        }
        grid.sync(); // Publishes all tensor writes before the next operation.
        i = end;
    }
}
} // namespace gpt2

extern "C" int gpt2_gemm_tile_n() { return gpt2::tile_n; }

extern "C" int gpt2_gemm_tile_m() { return gpt2::tile_m; }

extern "C" int gpt2_occupancy(int *workers) {
    int device, active;
    static thread_local int cached_device = -1, cached_capacity = 0;
    cudaDeviceProp prop;
    cudaError_t e = cudaGetDevice(&device);
    if (e != cudaSuccess) return e;
    if (device == cached_device) { *workers = cached_capacity; return cudaSuccess; }
    if ((e = cudaGetDeviceProperties(&prop, device)) != cudaSuccess) return e;
    if (!prop.cooperativeLaunch) return cudaErrorNotSupported;
    if ((e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, gpt2::persistent, gpt2::threads, gpt2::shared_bytes)) != cudaSuccess) return e;
    *workers = active * prop.multiProcessorCount;
    if (active) { cached_device = device; cached_capacity = *workers; }
    return active ? cudaSuccess : cudaErrorInvalidConfiguration;
}

extern "C" int gpt2_launch(const gpt2::Operation *device_ops, int count,
                            int workers, cudaStream_t stream) {
    int capacity;
    int e = gpt2_occupancy(&capacity);
    if (e) return e;
    if (workers == 0) workers = capacity;
    if (workers < 1 || workers > capacity || count < 1) return cudaErrorInvalidValue;
    void *args[] = {&device_ops, &count};
    return cudaLaunchCooperativeKernel((void *)gpt2::persistent, workers,
        gpt2::threads, args, gpt2::shared_bytes, stream);
}

extern "C" int gpt2_operation(const gpt2::Operation *host_op, cudaStream_t stream) {
    if (host_op->tiles < 1) return cudaErrorInvalidValue;
    if (host_op->code == gpt2::Code::tile_graph) return cudaErrorNotSupported;
    gpt2::standalone<<<host_op->tiles, gpt2::threads, gpt2::shared_bytes, stream>>>(*host_op);
    return cudaGetLastError();
}

extern "C" int gpt2_gemm_parameters(const gpt2::Operation *op, void *buffer) {
    switch (op->flags & 3) {
        case 0: return gpt2::gemm_parameters<false, false>(*op, buffer);
        case 1: return gpt2::gemm_parameters<true, false>(*op, buffer);
        case 2: return gpt2::gemm_parameters<false, true>(*op, buffer);
        default: return gpt2::gemm_parameters<true, true>(*op, buffer);
    }
}
