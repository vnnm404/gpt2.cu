#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

static inline void gpuErrchk_internal(cudaError_t code, const char *file,
                                      int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}
#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuErrchk_internal((ans), __FILE__, __LINE__);                             \
  }
