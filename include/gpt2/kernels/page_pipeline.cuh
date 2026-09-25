#pragma once
#include "program.cuh"
#include "sync.cuh"

namespace gpt2 {
// Two CTA-local pages span residual-add and normalization instructions. The
// next row's asynchronous input copies overlap the current row's computation.
// A page is reclaimed only after every lane finishes reading/writing it.
__device__ __forceinline__ void page_norm_collective(const Operation &o, int tile, float *scratch) {
    constexpr int page_floats = shared_bytes / (2 * sizeof(float));
    int first = tile * o.flags, rows = min(o.flags, o.m - first), C = o.n;
    auto load = [&](int step) {
        float *page = scratch + (step & 1) * page_floats;
        for (int c = threadIdx.x * 4; c < C; c += threads * 4) {
            unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(page + c));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                         "r"(address), "l"(o.p[0] + (first + step) * C + c) : "memory");
            address = static_cast<unsigned>(__cvta_generic_to_shared(page + C + c));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                         "r"(address), "l"(o.p[1] + (first + step) * C + c) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    load(0);
    for (int step = 0; step < rows; ++step) {
        if (step + 1 < rows) {
            load(step + 1);
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads(); // Each issuing thread waited; all lanes may now read.
        float *page = scratch + (step & 1) * page_floats;
        float *reduction = page + 2 * C;
        int row = first + step;
        float sum = 0;
        for (int c = threadIdx.x; c < C; c += threads) {
            float value = page[c] + page[C + c];
            page[c] = value;
            o.p[2][row * C + c] = value;
            sum += value;
        }
        float mean = reduce(sum, reduction) / C;
        sum = 0;
        for (int c = threadIdx.x; c < C; c += threads)
            sum += (page[c] - mean) * (page[c] - mean);
        float rstd = rsqrtf(reduce(sum, reduction) / C + 1e-5f);
        for (int c = threadIdx.x; c < C; c += threads)
            o.p[5][row * C + c] = (page[c] - mean) * rstd * o.p[3][c] + o.p[4][c];
        if (threadIdx.x == 0) { o.p[6][row * 2] = mean; o.p[6][row * 2 + 1] = rstd; }
        __syncthreads(); // Release this page before loading row step + 2.
    }
}

// Four loader warps and four consumer warps share two pages. Named barriers
// synchronize each group independently. Virtual lanes preserve the original
// 256-thread normalization reduction order with 128 computing threads.
__device__ __forceinline__ float page_reduce(float a, float b, float *scratch) {
    int t = threadIdx.x;
    a = warp_sum(a); b = warp_sum(b);
    if ((t & 31) == 0) { scratch[t / 32] = a; scratch[4 + t / 32] = b; }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    a = t < 8 ? scratch[t] : 0.f;
    a = warp_sum(a);
    if (t == 0) scratch[8] = a;
    asm volatile("bar.sync 1, 128;" ::: "memory");
    a = scratch[8];
    asm volatile("bar.sync 1, 128;" ::: "memory");
    return a;
}

__device__ __forceinline__ void page_norm_warps(const Operation &o, int tile, float *scratch) {
    constexpr int page_floats = shared_bytes / (2 * sizeof(float));
    __shared__ unsigned long long ready[2], released[2];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int page = 0; page < 2; ++page) {
            PageBarrier(ready[page]).init();
            PageBarrier(released[page]).init();
        }
    }
    __syncthreads();
    int first = tile * o.flags, rows = min(o.flags, o.m - first), C = o.n;
    auto *trace = reinterpret_cast<unsigned long long *>(o.p[7]);
    auto record = [&](int row, int event) {
        if (trace) {
            unsigned long long now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            trace[row * 4 + event] = now;
        }
    };
    if (threadIdx.x >= 128) {
        for (int step = 0; step < rows; ++step) {
            int slot = step & 1, phase = step / 2, row = first + step;
            float *page = scratch + slot * page_floats;
            if (threadIdx.x == 128) {
                if (step >= 2) PageBarrier(released[slot]).wait(phase - 1);
            }
            asm volatile("bar.sync 2, 128;" ::: "memory");
            if (threadIdx.x == 128) record(row, 0);
            for (int c = (threadIdx.x - 128) * 4; c < C; c += 128 * 4) {
                unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(page + c));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                             "r"(address), "l"(o.p[0] + row * C + c) : "memory");
                address = static_cast<unsigned>(__cvta_generic_to_shared(page + C + c));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                             "r"(address), "l"(o.p[1] + row * C + c) : "memory");
            }
            asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
            asm volatile("bar.sync 2, 128;" ::: "memory");
            if (threadIdx.x == 128) {
                record(row, 1);
                PageBarrier(ready[slot]).arrive();
            }
        }
    } else {
        for (int step = 0; step < rows; ++step) {
            int slot = step & 1, phase = step / 2, row = first + step;
            float *page = scratch + slot * page_floats, *reduction = page + 2 * C;
            if (threadIdx.x == 0) {
                PageBarrier(ready[slot]).wait(phase);
                record(row, 2);
            }
            asm volatile("bar.sync 1, 128;" ::: "memory");
            float sum[2] = {0, 0};
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                for (int c = threadIdx.x + half * 128; c < C; c += threads) {
                    float value = page[c] + page[C + c];
                    page[c] = value;
                    o.p[2][row * C + c] = value;
                    sum[half] += value;
                }
            }
            float mean = page_reduce(sum[0], sum[1], reduction) / C;
            sum[0] = sum[1] = 0;
            #pragma unroll
            for (int half = 0; half < 2; ++half)
                for (int c = threadIdx.x + half * 128; c < C; c += threads)
                    sum[half] += (page[c] - mean) * (page[c] - mean);
            float rstd = rsqrtf(page_reduce(sum[0], sum[1], reduction) / C + 1e-5f);
            for (int c = threadIdx.x; c < C; c += 128)
                o.p[5][row * C + c] = (page[c] - mean) * rstd * o.p[3][c] + o.p[4][c];
            if (threadIdx.x == 0) { o.p[6][row * 2] = mean; o.p[6][row * 2 + 1] = rstd; }
            asm volatile("bar.sync 1, 128;" ::: "memory");
            if (threadIdx.x == 0) {
                record(row, 3);
                PageBarrier(released[slot]).arrive();
            }
        }
    }
    __syncthreads(); // Drain both worker groups before reclaiming the pool.
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int page = 0; page < 2; ++page) {
            PageBarrier(ready[page]).invalidate();
            PageBarrier(released[page]).invalidate();
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void page_norm(const Operation &o, int tile, float *scratch) {
    if (o.k) page_norm_warps(o, tile, scratch);
    else page_norm_collective(o, tile, scratch);
}
} // namespace gpt2
