#pragma once
#include "sync.cuh"
#include "gemm.cuh"
#include "elementwise.cuh"
#include <cooperative_groups.h>

namespace gpt2 {
struct TileTask {
    int op, tile, columns, edge_begin, edge_count, dependencies;
    int group, ready_begin, ready_count;
};
static_assert(sizeof(TileTask) == 9 * sizeof(int));

// The queue contains only runnable tasks. Consumers never reserve a blocked
// instruction, so even one resident CTA can execute any acyclic graph.
__device__ __forceinline__ int take_ready(int *control, int *queue) {
    DeviceAtomic head(control[1]), tail(control[2]);
    int h = head.load(cuda::memory_order_relaxed);
    while (h < tail.load(cuda::memory_order_relaxed)) {
        int task = DeviceAtomic(queue[h]).load(cuda::memory_order_acquire);
        if (task < 0) return -1; // A producer has reserved but not published.
        if (head.compare_exchange_weak(h, h + 1, cuda::memory_order_relaxed)) return task;
    }
    return -1;
}

__device__ __forceinline__ void tile_graph(const Operation &graph, float *scratch) {
    auto grid = cooperative_groups::this_grid();
    // Keep scheduler state out of registers while an inlined GEMM executes.
    // Volatile pointer fields deliberately reload after the operation rather
    // than extending their live ranges across CUTLASS's register-heavy body.
    struct SharedState {
        const Operation * volatile ops;
        const TileTask * volatile tasks;
        const int * volatile edges;
        int * volatile queue;
        int * volatile remaining;
        int * volatile control;
        const int * volatile roots;
        unsigned long long * volatile trace;
        volatile int selected, continuation, finished, next_static;
    };
    __shared__ SharedState state;
    if (threadIdx.x == 0) {
        state.ops = reinterpret_cast<const Operation *>(graph.p[0]);
        state.tasks = reinterpret_cast<const TileTask *>(graph.p[1]);
        state.edges = reinterpret_cast<const int *>(graph.p[2]);
        state.queue = reinterpret_cast<int *>(graph.p[3]);
        state.remaining = reinterpret_cast<int *>(graph.p[4]);
        state.control = reinterpret_cast<int *>(graph.p[5]);
        state.roots = reinterpret_cast<const int *>(graph.p[6]);
        state.trace = reinterpret_cast<unsigned long long *>(graph.p[7]);
        state.continuation = -1;
        state.finished = 0;
        state.next_static = blockIdx.x;
    }
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < graph.m; i += gridDim.x * blockDim.x) {
        state.remaining[i] = state.tasks[i].dependencies;
        state.queue[i] = -1;
    }
    if (blockIdx.x == 0 && threadIdx.x < 4) state.control[threadIdx.x] = 0;
    grid.sync();
    while (true) {
        if (threadIdx.x == 0) {
            int task = state.continuation;
            state.continuation = -1;
            if (graph.flags & 1) {
                task = state.next_static < graph.m ? state.next_static : -1;
                state.next_static += gridDim.x;
                if (task >= 0) {
                    // Task IDs and each worker stream are topological. The
                    // least unfinished task cannot depend on a later waiter.
                    while (DeviceAtomic(state.remaining[state.tasks[task].group]).load(cuda::memory_order_acquire))
                        __nanosleep(64);
                }
            }
            while (!(graph.flags & 1) && task < 0) {
                task = take_ready(state.control, state.queue);
                if (task >= 0) break;
                if (DeviceAtomic(state.control[0]).load(cuda::memory_order_relaxed) < graph.n) {
                    int root = DeviceAtomic(state.control[0]).fetch_add(1, cuda::memory_order_relaxed);
                    if (root < graph.n) { task = state.roots[root]; break; }
                }
                if (state.finished) {
                    DeviceAtomic(state.control[3]).fetch_add(state.finished, cuda::memory_order_release);
                    state.finished = 0;
                }
                if (DeviceAtomic(state.control[3]).load(cuda::memory_order_acquire) == graph.m) break;
                __nanosleep(64);
            }
            state.selected = task;
            if (state.trace && task >= 0) {
                unsigned long long now;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                state.trace[3 * task] = now;
                state.trace[3 * task + 2] = blockIdx.x;
            }
        }
        __syncthreads(); // Includes the leader's acquire of the ready task.
        int index = state.selected;
        if (index < 0) break;
        TileTask task = state.tasks[index];
        const Operation &op = state.ops[task.op];
        if (task.columns) {
            int nc = (task.columns + tile_n - 1) / tile_n;
            int row = task.tile / nc * tile_m, col = task.tile % nc * tile_n;
            for (int i = threadIdx.x; i < tile_m * tile_n; i += threads) {
                int r = row + i / tile_n, c = col + i % tile_n;
                if (c < task.columns && r * task.columns + c < op.m)
                    elementwise_at(op, r * task.columns + c);
            }
        } else if (op.code == Code::gemm) {
            // The tile compiler accepts only GEMM, pointwise, and row-reduction
            // instructions. Avoid duplicating the entire attention/optimizer
            // dispatcher inside this path of the persistent translation unit.
            switch (op.flags & 3) {
                case 0: gemm<false, false>(op, task.tile, scratch); break;
                case 1: gemm<true, false>(op, task.tile, scratch); break;
                case 2: gemm<false, true>(op, task.tile, scratch); break;
                case 3: gemm<true, true>(op, task.tile, scratch); break;
            }
        } else if (op.code == Code::sum_rows) sum_rows(op, task.tile, scratch);
        else asm volatile("trap;");
        __syncthreads(); // Publish every lane's output through the leader.
        if (state.trace) {
            if (threadIdx.x == 0) {
                unsigned long long now;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                state.trace[3 * state.selected + 1] = now;
            }
            __syncthreads();
        }
        if (graph.flags & 1) {
            TileTask completed = state.tasks[state.selected];
            for (int e = threadIdx.x; e < completed.edge_count; e += threads)
                DeviceAtomic(state.remaining[state.edges[completed.edge_begin + e]]).fetch_sub(1, cuda::memory_order_acq_rel);
            __syncthreads();
        } else if (threadIdx.x < 32) {
            TileTask completed = state.tasks[state.selected];
            int lane = threadIdx.x;
            for (int offset = 0; offset < completed.edge_count; offset += 32) {
                int begin = 0, count = 0, first = -1;
                if (offset + lane < completed.edge_count) {
                    int candidate = state.edges[completed.edge_begin + offset + lane];
                    // One counter represents an identical set of prerequisites.
                    if (state.tasks[candidate].dependencies == 1 ||
                        DeviceAtomic(state.remaining[candidate]).fetch_sub(1, cuda::memory_order_acq_rel) == 1) {
                        begin = state.tasks[candidate].ready_begin;
                        count = state.tasks[candidate].ready_count;
                        first = state.edges[begin];
                    }
                }
                unsigned ready = __ballot_sync(0xffffffff, count > 0);
                int next = lane == 0 ? state.continuation : -1;
                int take = __shfl_sync(0xffffffff, next, 0) < 0 ? __ffs(ready) - 1 : -1;
                if (take >= 0) {
                    next = __shfl_sync(0xffffffff, first, take);
                    if (lane == 0) state.continuation = next;
                    if (lane == take) { ++begin; --count; }
                }
                int prefix = count;
                #pragma unroll
                for (int distance = 1; distance < 32; distance *= 2) {
                    int prior = __shfl_up_sync(0xffffffff, prefix, distance);
                    if (lane >= distance) prefix += prior;
                }
                int total = __shfl_sync(0xffffffff, prefix, 31), base = 0;
                if (lane == 0 && total)
                    base = DeviceAtomic(state.control[2]).fetch_add(total, cuda::memory_order_relaxed);
                base = __shfl_sync(0xffffffff, base, 0) + prefix - count;
                for (int child = 0; child < count; ++child)
                    DeviceAtomic(state.queue[base + child]).store(state.edges[begin + child], cuda::memory_order_release);
                __syncwarp();
            }
            __syncwarp();
            if (lane == 0) ++state.finished;
        }
    }
}
} // namespace gpt2
