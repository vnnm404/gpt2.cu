#ifndef GPT2_LAYERS_MLP_H
#define GPT2_LAYERS_MLP_H

#include <cuda_runtime.h>

/* MLP layer structures and functions */

#define TILE_SIZE 32

// Helper macros for launching optimized MLP kernels
#define MLP_FORWARD_GRID(output_dim, batch_size, seq_len) \
    dim3(((output_dim) + TILE_SIZE - 1) / TILE_SIZE, \
         ((batch_size) * (seq_len) + TILE_SIZE - 1) / TILE_SIZE)

#define MLP_BACKWARD_INPUT_GRID(input_dim, batch_size, seq_len) \
    dim3(((input_dim) + TILE_SIZE - 1) / TILE_SIZE, \
         ((batch_size) * (seq_len) + TILE_SIZE - 1) / TILE_SIZE)

#define MLP_BACKWARD_WEIGHT_GRID(output_dim, input_dim) \
    dim3(((output_dim) + TILE_SIZE - 1) / TILE_SIZE, \
         ((input_dim) + TILE_SIZE - 1) / TILE_SIZE)

#define MLP_BLOCK_DIM dim3(TILE_SIZE * TILE_SIZE)

// sanity check to make sure its enabled
// static_assert(
// #ifdef MLP_DOUBLE_BUFFER
// true
// #else
// false
// #endif
// , "MLP_DOUBLE_BUFFER OFF");

// WMMA is a legacy misnomer, its not using WMMA, but this will be deleted soon so doesnt rly matter
#ifndef MLP_WMMA
template<int MLP_TILE>
__device__ void mlp_forward_device(float *out, const float *input, const float *w,
    const float *b, int batch_size, int seq_len, int input_dim, int output_dim,
    int blockIdx_x, int blockIdx_y, float *shared_mem) {
    float (*s_input)[MLP_TILE] = (float (*)[MLP_TILE])shared_mem;
    float (*s_weight)[MLP_TILE] = (float (*)[MLP_TILE])(shared_mem + MLP_TILE * MLP_TILE);

    int tx = threadIdx.x % MLP_TILE;
    int ty = threadIdx.x / MLP_TILE;

    int out_col = blockIdx_x * MLP_TILE + tx;
    int out_row = blockIdx_y * MLP_TILE + ty;

    float acc = 0.0f;
    int num_tiles = (input_dim + MLP_TILE - 1) / MLP_TILE;

    for (int tile = 0; tile < num_tiles; tile++) {
        int input_col = tile * MLP_TILE + tx;
        if (out_row < batch_size * seq_len && input_col < input_dim) {
            s_input[ty][tx] = input[out_row * input_dim + input_col];
        } else {
            s_input[ty][tx] = 0.0f;
        }

        int weight_row = tile * MLP_TILE + ty;
        if (weight_row < input_dim && out_col < output_dim) {
            s_weight[ty][tx] = w[weight_row * output_dim + out_col];
        } else {
            s_weight[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < MLP_TILE; k++) {
            acc += s_input[ty][k] * s_weight[k][tx];
        }

        __syncthreads();
    }

    if (out_row < batch_size * seq_len && out_col < output_dim) {
        float bias_val = (b == NULL) ? 0.0f : b[out_col];
        out[out_row * output_dim + out_col] = acc + bias_val;
    }
}
#else

#ifdef MLP_DOUBLE_BUFFER
/*
 * Double buffered, warp microtiling, f32, cuda::pipeline.
 */
#include <cuda/pipeline>

template<int MLP_TILE>
__device__ void mlp_forward_device(
    float* out,
    const float* __restrict__ input,
    const float* __restrict__ w,
    const float* __restrict__ b,
    int batch_size,
    int seq_len,
    int input_dim,
    int output_dim,
    int blockIdx_x,
    int blockIdx_y,
    float* shared_mem
) {
    static_assert(MLP_TILE % 2 == 0, "MLP_TILE must be even for 2x2 micro-tiling.");

    // double buffered tiles: [2][MLP_TILE][MLP_TILE]
    float (*s_input)[2][MLP_TILE][MLP_TILE]  = (float (*)[2][MLP_TILE][MLP_TILE])shared_mem;
    float (*s_weight)[2][MLP_TILE][MLP_TILE] = (float (*)[2][MLP_TILE][MLP_TILE])(shared_mem + 2 * MLP_TILE * MLP_TILE);

    // launch (MLP_TILE/2)*(MLP_TILE/2) threads per block
    //   eg for MLP_TILE=32 => 16*16 = 256 threads (8 warps)
    constexpr int TX_THREADS = MLP_TILE / 2;
    constexpr int TY_THREADS = MLP_TILE / 2;

    // map threadIdx.x into a base (tx, ty) for a 2x2 microtile
    int t = threadIdx.x;
    int tx2 = t % TX_THREADS;
    int ty2 = t / TX_THREADS;

    int tx_base = tx2 * 2;
    int ty_base = ty2 * 2;

    // output tile origin in C
    int out_col_base = blockIdx_x * MLP_TILE + tx_base;
    int out_row_base = blockIdx_y * MLP_TILE + ty_base;

    // 2x2 register accs per thread
    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    int rows_total = batch_size * seq_len;
    int num_tiles = (input_dim + MLP_TILE - 1) / MLP_TILE;

    // create pipeline with per-thread scope
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // prefetch first tile into buffer 0
    int buf_write = 0;
    {
        int tile = 0;
        int a_col0 = tile * MLP_TILE + (tx_base + 0);
        int a_col1 = tile * MLP_TILE + (tx_base + 1);
        int a_row0 = out_row_base + 0;
        int a_row1 = out_row_base + 1;

        pipe.producer_acquire();

        // async copy input tile
        if (a_row0 < rows_total && a_col0 < input_dim) {
            cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 0][tx_base + 0],
                             &input[a_row0 * input_dim + a_col0], sizeof(float), pipe);
        } else {
            (*s_input)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
        }
        if (a_row0 < rows_total && a_col1 < input_dim) {
            cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 0][tx_base + 1],
                             &input[a_row0 * input_dim + a_col1], sizeof(float), pipe);
        } else {
            (*s_input)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
        }
        if (a_row1 < rows_total && a_col0 < input_dim) {
            cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 1][tx_base + 0],
                             &input[a_row1 * input_dim + a_col0], sizeof(float), pipe);
        } else {
            (*s_input)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
        }
        if (a_row1 < rows_total && a_col1 < input_dim) {
            cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 1][tx_base + 1],
                             &input[a_row1 * input_dim + a_col1], sizeof(float), pipe);
        } else {
            (*s_input)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
        }

        int b_row0 = tile * MLP_TILE + (ty_base + 0);
        int b_row1 = tile * MLP_TILE + (ty_base + 1);
        int b_col0 = out_col_base + 0;
        int b_col1 = out_col_base + 1;

        // async copy weight tile
        if (b_row0 < input_dim && b_col0 < output_dim) {
            cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 0][tx_base + 0],
                             &w[b_row0 * output_dim + b_col0], sizeof(float), pipe);
        } else {
            (*s_weight)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
        }
        if (b_row0 < input_dim && b_col1 < output_dim) {
            cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 0][tx_base + 1],
                             &w[b_row0 * output_dim + b_col1], sizeof(float), pipe);
        } else {
            (*s_weight)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
        }
        if (b_row1 < input_dim && b_col0 < output_dim) {
            cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 1][tx_base + 0],
                             &w[b_row1 * output_dim + b_col0], sizeof(float), pipe);
        } else {
            (*s_weight)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
        }
        if (b_row1 < input_dim && b_col1 < output_dim) {
            cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 1][tx_base + 1],
                             &w[b_row1 * output_dim + b_col1], sizeof(float), pipe);
        } else {
            (*s_weight)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
        }

        pipe.producer_commit();
    }
    
    // wait for first tile to complete
    pipe.consumer_wait();
    __syncthreads();

    for (int tile = 0; tile < num_tiles; tile++) {
        int buf_read = tile & 1;
        buf_write = 1 - buf_read;

        // prefetch next tile while computing current
        if (tile + 1 < num_tiles) {
            int next_tile = tile + 1;
            int a_col0 = next_tile * MLP_TILE + (tx_base + 0);
            int a_col1 = next_tile * MLP_TILE + (tx_base + 1);
            int a_row0 = out_row_base + 0;
            int a_row1 = out_row_base + 1;

            pipe.producer_acquire();

            // async copy input tile
            if (a_row0 < rows_total && a_col0 < input_dim) {
                cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 0][tx_base + 0],
                                 &input[a_row0 * input_dim + a_col0], sizeof(float), pipe);
            } else {
                (*s_input)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
            }
            if (a_row0 < rows_total && a_col1 < input_dim) {
                cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 0][tx_base + 1],
                                 &input[a_row0 * input_dim + a_col1], sizeof(float), pipe);
            } else {
                (*s_input)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
            }
            if (a_row1 < rows_total && a_col0 < input_dim) {
                cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 1][tx_base + 0],
                                 &input[a_row1 * input_dim + a_col0], sizeof(float), pipe);
            } else {
                (*s_input)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
            }
            if (a_row1 < rows_total && a_col1 < input_dim) {
                cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 1][tx_base + 1],
                                 &input[a_row1 * input_dim + a_col1], sizeof(float), pipe);
            } else {
                (*s_input)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
            }

            int b_row0 = next_tile * MLP_TILE + (ty_base + 0);
            int b_row1 = next_tile * MLP_TILE + (ty_base + 1);
            int b_col0 = out_col_base + 0;
            int b_col1 = out_col_base + 1;

            // async copy weight tile
            if (b_row0 < input_dim && b_col0 < output_dim) {
                cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 0][tx_base + 0],
                                 &w[b_row0 * output_dim + b_col0], sizeof(float), pipe);
            } else {
                (*s_weight)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
            }
            if (b_row0 < input_dim && b_col1 < output_dim) {
                cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 0][tx_base + 1],
                                 &w[b_row0 * output_dim + b_col1], sizeof(float), pipe);
            } else {
                (*s_weight)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
            }
            if (b_row1 < input_dim && b_col0 < output_dim) {
                cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 1][tx_base + 0],
                                 &w[b_row1 * output_dim + b_col0], sizeof(float), pipe);
            } else {
                (*s_weight)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
            }
            if (b_row1 < input_dim && b_col1 < output_dim) {
                cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 1][tx_base + 1],
                                 &w[b_row1 * output_dim + b_col1], sizeof(float), pipe);
            } else {
                (*s_weight)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
            }

            pipe.producer_commit();
        }

        // compute on current tile from buf_read (OVERLAPS with async loads above!)
        #pragma unroll
        for (int k = 0; k < MLP_TILE; k++) {
            float a0 = (*s_input)[buf_read][ty_base + 0][k];
            float a1 = (*s_input)[buf_read][ty_base + 1][k];

            float b0 = (*s_weight)[buf_read][k][tx_base + 0];
            float b1 = (*s_weight)[buf_read][k][tx_base + 1];

            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        // release current buffer and wait for next tile to finish loading
        pipe.consumer_release();
        if (tile + 1 < num_tiles) {
            pipe.consumer_wait();
        }
        __syncthreads();
    }

    // bias is per output column
    float bias0 = 0.0f;
    float bias1 = 0.0f;
    if (b != nullptr) {
        if (out_col_base + 0 < output_dim) bias0 = b[out_col_base + 0];
        if (out_col_base + 1 < output_dim) bias1 = b[out_col_base + 1];
    }

    // store row0
    if (out_row_base + 0 < rows_total) {
        int r0 = out_row_base + 0;
        if (out_col_base + 0 < output_dim) out[r0 * output_dim + (out_col_base + 0)] = acc00 + bias0;
        if (out_col_base + 1 < output_dim) out[r0 * output_dim + (out_col_base + 1)] = acc01 + bias1;
    }
    // store row1
    if (out_row_base + 1 < rows_total) {
        int r1 = out_row_base + 1;
        if (out_col_base + 0 < output_dim) out[r1 * output_dim + (out_col_base + 0)] = acc10 + bias0;
        if (out_col_base + 1 < output_dim) out[r1 * output_dim + (out_col_base + 1)] = acc11 + bias1;
    }
}
#else
/*
 * Single buffered, warp microtiling, f32.
 */
template<int MLP_TILE>
__device__ void mlp_forward_device(
    float* out,
    const float* __restrict__ input,
    const float* __restrict__ w,
    const float* __restrict__ b,
    int batch_size,
    int seq_len,
    int input_dim,
    int output_dim,
    int blockIdx_x,
    int blockIdx_y,
    float* shared_mem
) {
    static_assert(MLP_TILE % 2 == 0, "MLP_TILE must be even for 2x2 micro-tiling.");

    // s_input[ty][tx] corresponds to A row=out_row, col=input_col
    float (*s_input)[MLP_TILE]  = (float (*)[MLP_TILE])shared_mem; // (MLP_TILE, MLP_TILE)

    // s_weight[ty][tx] corresponds to B row=weight_row, col=out_col
    float (*s_weight)[MLP_TILE] = (float (*)[MLP_TILE])(shared_mem + MLP_TILE * MLP_TILE);  // (MLP_TILE, MLP_TILE)

    // launch (MLP_TILE/2)*(MLP_TILE/2) threads per block
    //   eg for MLP_TILE=16 => 8*8 = 64 threads (2 warps)
    constexpr int TX_THREADS = MLP_TILE / 2;
    constexpr int TY_THREADS = MLP_TILE / 2;

    // map threadIdx.x into a base (tx, ty) for a 2x2 microtile
    int t = threadIdx.x;                // 0 .. (TX_THREADS*TY_THREADS - 1)
    int tx2 = t % TX_THREADS;           // 0 .. (MLP_TILE/2 - 1)
    int ty2 = t / TX_THREADS;           // 0 .. (MLP_TILE/2 - 1)

    int tx_base = tx2 * 2;              // 0, 2, 4, ..., MLP_TILE-2
    int ty_base = ty2 * 2;              // 0, 2, 4, ..., MLP_TILE-2

    // output tile origin in C
    int out_col_base = blockIdx_x * MLP_TILE + tx_base;
    int out_row_base = blockIdx_y * MLP_TILE + ty_base;

    // 2x2 register accs per thread
    float acc00 = 0.0f;  // C[row+0, col+0]
    float acc01 = 0.0f;  // C[row+0, col+1]
    float acc10 = 0.0f;  // C[row+1, col+0]
    float acc11 = 0.0f;  // C[row+1, col+1]

    int rows_total = batch_size * seq_len;
    int num_tiles = (input_dim + MLP_TILE - 1) / MLP_TILE;

    for (int tile = 0; tile < num_tiles; tile++) {
        // collectively load of 16x16 tile, each thread loads a 2x2 patch
        //  of s_input and a 2x2 patch of s_weight (4 elements each = 256).

        // A tile -> s_input
        // global A coords:
        //   row = out_row_base + {0,1}
        //   col = tile*MLP_TILE + (tx_base + {0,1})
        int a_col0 = tile * MLP_TILE + (tx_base + 0);
        int a_col1 = tile * MLP_TILE + (tx_base + 1);

        int a_row0 = out_row_base + 0;
        int a_row1 = out_row_base + 1;

        s_input[ty_base + 0][tx_base + 0] =
            (a_row0 < rows_total && a_col0 < input_dim) ? input[a_row0 * input_dim + a_col0] : 0.0f;
        s_input[ty_base + 0][tx_base + 1] =
            (a_row0 < rows_total && a_col1 < input_dim) ? input[a_row0 * input_dim + a_col1] : 0.0f;
        s_input[ty_base + 1][tx_base + 0] =
            (a_row1 < rows_total && a_col0 < input_dim) ? input[a_row1 * input_dim + a_col0] : 0.0f;
        s_input[ty_base + 1][tx_base + 1] =
            (a_row1 < rows_total && a_col1 < input_dim) ? input[a_row1 * input_dim + a_col1] : 0.0f;

        // B tile -> s_weight
        // global B coords:
        //   row = tile*MLP_TILE + (ty_base + {0,1})
        //   col = out_col_base + {0,1}
        int b_row0 = tile * MLP_TILE + (ty_base + 0);
        int b_row1 = tile * MLP_TILE + (ty_base + 1);

        int b_col0 = out_col_base + 0;
        int b_col1 = out_col_base + 1;

        s_weight[ty_base + 0][tx_base + 0] =
            (b_row0 < input_dim && b_col0 < output_dim) ? w[b_row0 * output_dim + b_col0] : 0.0f;
        s_weight[ty_base + 0][tx_base + 1] =
            (b_row0 < input_dim && b_col1 < output_dim) ? w[b_row0 * output_dim + b_col1] : 0.0f;
        s_weight[ty_base + 1][tx_base + 0] =
            (b_row1 < input_dim && b_col0 < output_dim) ? w[b_row1 * output_dim + b_col0] : 0.0f;
        s_weight[ty_base + 1][tx_base + 1] =
            (b_row1 < input_dim && b_col1 < output_dim) ? w[b_row1 * output_dim + b_col1] : 0.0f;

        __syncthreads();

        // each thread updates a 2x2 block of C for this K-tile.
        // in shmem:
        //   A uses rows ty_base+{0,1} across k
        //   B uses cols tx_base+{0,1} across k
        #pragma unroll
        for (int k = 0; k < MLP_TILE; k++) {
            float a0 = s_input[ty_base + 0][k];
            float a1 = s_input[ty_base + 1][k];

            float b0 = s_weight[k][tx_base + 0];
            float b1 = s_weight[k][tx_base + 1];

            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        __syncthreads();
    }

    // bias is per output column
    float bias0 = 0.0f;
    float bias1 = 0.0f;
    if (b != nullptr) {
        if (out_col_base + 0 < output_dim) bias0 = b[out_col_base + 0];
        if (out_col_base + 1 < output_dim) bias1 = b[out_col_base + 1];
    }

    // store row0
    if (out_row_base + 0 < rows_total) {
        int r0 = out_row_base + 0;
        if (out_col_base + 0 < output_dim) out[r0 * output_dim + (out_col_base + 0)] = acc00 + bias0;
        if (out_col_base + 1 < output_dim) out[r0 * output_dim + (out_col_base + 1)] = acc01 + bias1;
    }
    // store row1
    if (out_row_base + 1 < rows_total) {
        int r1 = out_row_base + 1;
        if (out_col_base + 0 < output_dim) out[r1 * output_dim + (out_col_base + 0)] = acc10 + bias0;
        if (out_col_base + 1 < output_dim) out[r1 * output_dim + (out_col_base + 1)] = acc11 + bias1;
    }
}
#endif
#endif

template<int MLP_TILE>
__global__ void mlp_forward(float *out, const float *input, const float *w, const float *b, int batch_size, int seq_len, int input_dim, int output_dim) {
    extern __shared__ float shared_mem[];

    mlp_forward_device<MLP_TILE>(
        out,
        input,
        w,
        b,
        batch_size,
        seq_len,
        input_dim,
        output_dim,
        blockIdx.x,
        blockIdx.y,
        shared_mem
    );
}

__device__ void mlp_backward_input_device(float *g_input, const float *g_out, const float *weight,
                                          int batch_size, int seq_len, int input_dim, int output_dim,
                                          int blockIdx_x, int blockIdx_y, float *shared_mem);
__device__ void mlp_backward_weight_device(float *g_weight, float *g_bias, const float *g_out, const float *input,
                                           int batch_size, int seq_len, int input_dim, int output_dim,
                                           int blockIdx_x, int blockIdx_y, float *shared_mem);

__global__ void mlp_backward_input(float *g_input, const float *g_out, const float *weight, int batch_size, int seq_len, int input_dim, int output_dim);

__global__ void mlp_backward_weight(float *g_weight, float *g_bias, const float *g_out, const float *input, int batch_size, int seq_len, int input_dim, int output_dim);

#endif /* GPT2_LAYERS_MLP_H */