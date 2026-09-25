#pragma once
#include <cuda/atomic>

namespace gpt2 {
// Explicit scoped PTX avoids CUDA 12.4 atomic_ref pointer temporaries in local
// memory. All calls use constant orders; the unused branches compile away.
struct DeviceAtomic {
    int *pointer;
    __device__ __forceinline__ explicit DeviceAtomic(int &value) : pointer(&value) {}
    __device__ __forceinline__ int load(cuda::memory_order order) const {
        int value;
        if (order == cuda::memory_order_relaxed)
            asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(pointer) : "memory");
        else
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(pointer) : "memory");
        return value;
    }
    __device__ __forceinline__ void store(int value, cuda::memory_order) const {
        asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(pointer), "r"(value) : "memory");
    }
    __device__ __forceinline__ int fetch_add(int value, cuda::memory_order order) const {
        int previous;
        if (order == cuda::memory_order_relaxed)
            asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(previous) : "l"(pointer), "r"(value) : "memory");
        else if (order == cuda::memory_order_release)
            asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(previous) : "l"(pointer), "r"(value) : "memory");
        else
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(previous) : "l"(pointer), "r"(value) : "memory");
        return previous;
    }
    __device__ __forceinline__ int fetch_sub(int value, cuda::memory_order order) const {
        return fetch_add(-value, order);
    }
    __device__ __forceinline__ bool compare_exchange_weak(int &expected, int desired, cuda::memory_order) const {
        int previous;
        asm volatile("atom.relaxed.gpu.global.cas.b32 %0, [%1], %2, %3;" : "=r"(previous) :
                     "l"(pointer), "r"(expected), "r"(desired) : "memory");
        bool success = previous == expected;
        expected = previous;
        return success;
    }
};

// Ampere shared-memory barriers provide page ownership independently of the
// CTA-wide barrier. Parity is safe because each page has one producer/consumer
// and cannot advance again until the consumer explicitly releases it.
struct PageBarrier {
    unsigned address;
    __device__ __forceinline__ explicit PageBarrier(unsigned long long &value)
        : address(static_cast<unsigned>(__cvta_generic_to_shared(&value))) {}
    __device__ __forceinline__ void init() const {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(address) : "memory");
    }
    __device__ __forceinline__ void arrive() const {
        asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(address) : "memory");
    }
    __device__ __forceinline__ void wait(int phase) const {
        int done;
        do {
            asm volatile("{ .reg .pred done; mbarrier.test_wait.parity.shared.b64 done, [%1], %2; "
                         "selp.u32 %0, 1, 0, done; }" : "=r"(done) : "r"(address), "r"(phase & 1) : "memory");
            if (!done) __nanosleep(32);
        } while (!done);
    }
    __device__ __forceinline__ void invalidate() const {
        asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(address) : "memory");
    }
};
} // namespace gpt2
