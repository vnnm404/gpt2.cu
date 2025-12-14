#ifndef GPT2_LAYERS_MLP_H
#define GPT2_LAYERS_MLP_H

#include <cuda_runtime.h>
#include <cuda/pipeline>

/* MLP layer structures and functions */

#define TILE_SIZE 32

// Helper macros for launching optimized MLP kernels
#define MLP_FORWARD_GRID(output_dim, batch_size, seq_len) \
    dim3(((output_dim) + TILE_SIZE - 1) / TILE_SIZE,      \
         ((batch_size) * (seq_len) + TILE_SIZE - 1) / TILE_SIZE)

#define MLP_BACKWARD_INPUT_GRID(input_dim, batch_size, seq_len) \
    dim3(((input_dim) + TILE_SIZE - 1) / TILE_SIZE,             \
         ((batch_size) * (seq_len) + TILE_SIZE - 1) / TILE_SIZE)

#define MLP_BACKWARD_WEIGHT_GRID(output_dim, input_dim) \
    dim3(((output_dim) + TILE_SIZE - 1) / TILE_SIZE,    \
         ((input_dim) + TILE_SIZE - 1) / TILE_SIZE)

// Block dimension uses 2x2 microtiling: (TILE_SIZE/2) * (TILE_SIZE/2) threads
#define MLP_BLOCK_DIM dim3((TILE_SIZE / 2) * (TILE_SIZE / 2))

// Shared memory size for double-buffered tiles: 4 * TILE_SIZE * TILE_SIZE floats
#define MLP_SHARED_MEM_SIZE (4 * TILE_SIZE * TILE_SIZE * sizeof(float))

/*
 * Double buffered, warp microtiling, f32, cuda::pipeline.
 */
template <int MLP_TILE>
__device__ void mlp_forward_device(
    float *out,
    const float *__restrict__ input,
    const float *__restrict__ w,
    const float *__restrict__ b,
    int batch_size,
    int seq_len,
    int input_dim,
    int output_dim,
    int blockIdx_x,
    int blockIdx_y,
    float *shared_mem)
{
    static_assert(MLP_TILE % 2 == 0, "MLP_TILE must be even for 2x2 micro-tiling.");

    // double buffered tiles: [2][MLP_TILE][MLP_TILE]
    float (*s_input)[2][MLP_TILE][MLP_TILE] = (float (*)[2][MLP_TILE][MLP_TILE])shared_mem;
    float (*s_weight)[2][MLP_TILE][MLP_TILE] = (float (*)[2][MLP_TILE][MLP_TILE])(shared_mem + 2 * MLP_TILE * MLP_TILE);

    // launch (MLP_TILE/2)*(MLP_TILE/2) threads per block
    //   eg for MLP_TILE=32 => 16*16 = 256 threads (8 warps)
    constexpr int TX_THREADS = MLP_TILE / 2;

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
        if (a_row0 < rows_total && a_col0 < input_dim)
        {
            cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 0][tx_base + 0],
                               &input[a_row0 * input_dim + a_col0], sizeof(float), pipe);
        }
        else
        {
            (*s_input)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
        }
        if (a_row0 < rows_total && a_col1 < input_dim)
        {
            cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 0][tx_base + 1],
                               &input[a_row0 * input_dim + a_col1], sizeof(float), pipe);
        }
        else
        {
            (*s_input)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
        }
        if (a_row1 < rows_total && a_col0 < input_dim)
        {
            cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 1][tx_base + 0],
                               &input[a_row1 * input_dim + a_col0], sizeof(float), pipe);
        }
        else
        {
            (*s_input)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
        }
        if (a_row1 < rows_total && a_col1 < input_dim)
        {
            cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 1][tx_base + 1],
                               &input[a_row1 * input_dim + a_col1], sizeof(float), pipe);
        }
        else
        {
            (*s_input)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
        }

        int b_row0 = tile * MLP_TILE + (ty_base + 0);
        int b_row1 = tile * MLP_TILE + (ty_base + 1);
        int b_col0 = out_col_base + 0;
        int b_col1 = out_col_base + 1;

        // async copy weight tile
        if (b_row0 < input_dim && b_col0 < output_dim)
        {
            cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 0][tx_base + 0],
                               &w[b_row0 * output_dim + b_col0], sizeof(float), pipe);
        }
        else
        {
            (*s_weight)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
        }
        if (b_row0 < input_dim && b_col1 < output_dim)
        {
            cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 0][tx_base + 1],
                               &w[b_row0 * output_dim + b_col1], sizeof(float), pipe);
        }
        else
        {
            (*s_weight)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
        }
        if (b_row1 < input_dim && b_col0 < output_dim)
        {
            cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 1][tx_base + 0],
                               &w[b_row1 * output_dim + b_col0], sizeof(float), pipe);
        }
        else
        {
            (*s_weight)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
        }
        if (b_row1 < input_dim && b_col1 < output_dim)
        {
            cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 1][tx_base + 1],
                               &w[b_row1 * output_dim + b_col1], sizeof(float), pipe);
        }
        else
        {
            (*s_weight)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
        }

        pipe.producer_commit();
    }

    // wait for first tile to complete
    pipe.consumer_wait();
    __syncthreads();

    for (int tile = 0; tile < num_tiles; tile++)
    {
        int buf_read = tile & 1;
        buf_write = 1 - buf_read;

        // prefetch next tile while computing current
        if (tile + 1 < num_tiles)
        {
            int next_tile = tile + 1;
            int a_col0 = next_tile * MLP_TILE + (tx_base + 0);
            int a_col1 = next_tile * MLP_TILE + (tx_base + 1);
            int a_row0 = out_row_base + 0;
            int a_row1 = out_row_base + 1;

            pipe.producer_acquire();

            // async copy input tile
            if (a_row0 < rows_total && a_col0 < input_dim)
            {
                cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 0][tx_base + 0],
                                   &input[a_row0 * input_dim + a_col0], sizeof(float), pipe);
            }
            else
            {
                (*s_input)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
            }
            if (a_row0 < rows_total && a_col1 < input_dim)
            {
                cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 0][tx_base + 1],
                                   &input[a_row0 * input_dim + a_col1], sizeof(float), pipe);
            }
            else
            {
                (*s_input)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
            }
            if (a_row1 < rows_total && a_col0 < input_dim)
            {
                cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 1][tx_base + 0],
                                   &input[a_row1 * input_dim + a_col0], sizeof(float), pipe);
            }
            else
            {
                (*s_input)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
            }
            if (a_row1 < rows_total && a_col1 < input_dim)
            {
                cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 1][tx_base + 1],
                                   &input[a_row1 * input_dim + a_col1], sizeof(float), pipe);
            }
            else
            {
                (*s_input)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
            }

            int b_row0 = next_tile * MLP_TILE + (ty_base + 0);
            int b_row1 = next_tile * MLP_TILE + (ty_base + 1);
            int b_col0 = out_col_base + 0;
            int b_col1 = out_col_base + 1;

            // async copy weight tile
            if (b_row0 < input_dim && b_col0 < output_dim)
            {
                cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 0][tx_base + 0],
                                   &w[b_row0 * output_dim + b_col0], sizeof(float), pipe);
            }
            else
            {
                (*s_weight)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
            }
            if (b_row0 < input_dim && b_col1 < output_dim)
            {
                cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 0][tx_base + 1],
                                   &w[b_row0 * output_dim + b_col1], sizeof(float), pipe);
            }
            else
            {
                (*s_weight)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
            }
            if (b_row1 < input_dim && b_col0 < output_dim)
            {
                cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 1][tx_base + 0],
                                   &w[b_row1 * output_dim + b_col0], sizeof(float), pipe);
            }
            else
            {
                (*s_weight)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
            }
            if (b_row1 < input_dim && b_col1 < output_dim)
            {
                cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 1][tx_base + 1],
                                   &w[b_row1 * output_dim + b_col1], sizeof(float), pipe);
            }
            else
            {
                (*s_weight)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
            }

            pipe.producer_commit();
        }

// compute on current tile from buf_read (overlap)
#pragma unroll
        for (int k = 0; k < MLP_TILE; k++)
        {
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
        if (tile + 1 < num_tiles)
        {
            pipe.consumer_wait();
        }
        __syncthreads();
    }

    // bias is per output column
    float bias0 = 0.0f;
    float bias1 = 0.0f;
    if (b != nullptr)
    {
        if (out_col_base + 0 < output_dim)
            bias0 = b[out_col_base + 0];
        if (out_col_base + 1 < output_dim)
            bias1 = b[out_col_base + 1];
    }

    // store row0
    if (out_row_base + 0 < rows_total)
    {
        int r0 = out_row_base + 0;
        if (out_col_base + 0 < output_dim)
            out[r0 * output_dim + (out_col_base + 0)] = acc00 + bias0;
        if (out_col_base + 1 < output_dim)
            out[r0 * output_dim + (out_col_base + 1)] = acc01 + bias1;
    }
    // store row1
    if (out_row_base + 1 < rows_total)
    {
        int r1 = out_row_base + 1;
        if (out_col_base + 0 < output_dim)
            out[r1 * output_dim + (out_col_base + 0)] = acc10 + bias0;
        if (out_col_base + 1 < output_dim)
            out[r1 * output_dim + (out_col_base + 1)] = acc11 + bias1;
    }
}

template <int MLP_TILE>
__global__ void mlp_forward(float *out, const float *input, const float *w, const float *b, int batch_size, int seq_len, int input_dim, int output_dim)
{
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
        shared_mem);
}

/*
 * Double buffered, warp microtiling, f32, cuda::pipeline - Backward Input.
 * Computes: g_input = g_out @ weight^T
 */
template <int MLP_TILE>
__device__ void mlp_backward_input_device(
    float *g_input,
    const float *__restrict__ g_out,
    const float *__restrict__ weight,
    int batch_size,
    int seq_len,
    int input_dim,
    int output_dim,
    int blockIdx_x,
    int blockIdx_y,
    float *shared_mem)
{
    static_assert(MLP_TILE % 2 == 0, "MLP_TILE must be even for 2x2 micro-tiling.");

    // double buffered tiles: [2][MLP_TILE][MLP_TILE]
    float (*s_g_out)[2][MLP_TILE][MLP_TILE] = (float (*)[2][MLP_TILE][MLP_TILE])shared_mem;
    float (*s_weight)[2][MLP_TILE][MLP_TILE] = (float (*)[2][MLP_TILE][MLP_TILE])(shared_mem + 2 * MLP_TILE * MLP_TILE);

    // launch (MLP_TILE/2)*(MLP_TILE/2) threads per block
    //   eg for MLP_TILE=32 => 16*16 = 256 threads (8 warps)
    constexpr int TX_THREADS = MLP_TILE / 2;
    // constexpr int TY_THREADS = MLP_TILE / 2;

    // map threadIdx.x into a base (tx, ty) for a 2x2 microtile
    int t = threadIdx.x;
    int tx2 = t % TX_THREADS;
    int ty2 = t / TX_THREADS;

    int tx_base = tx2 * 2;
    int ty_base = ty2 * 2;

    // output tile origin in g_input
    int out_col_base = blockIdx_x * MLP_TILE + tx_base; // input_dim
    int out_row_base = blockIdx_y * MLP_TILE + ty_base; // batch * seq_len

    // 2x2 register accs per thread
    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    int rows_total = batch_size * seq_len;
    int num_tiles = (output_dim + MLP_TILE - 1) / MLP_TILE;

    // create pipeline with per-thread scope
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // prefetch first tile into buffer 0
    int buf_write = 0;
    {
        int tile = 0;

        // load g_out tile
        int a_col0 = tile * MLP_TILE + (tx_base + 0);
        int a_col1 = tile * MLP_TILE + (tx_base + 1);
        int a_row0 = out_row_base + 0;
        int a_row1 = out_row_base + 1;

        pipe.producer_acquire();

        if (a_row0 < rows_total && a_col0 < output_dim)
        {
            cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 0][tx_base + 0],
                               &g_out[a_row0 * output_dim + a_col0], sizeof(float), pipe);
        }
        else
        {
            (*s_g_out)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
        }
        if (a_row0 < rows_total && a_col1 < output_dim)
        {
            cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 0][tx_base + 1],
                               &g_out[a_row0 * output_dim + a_col1], sizeof(float), pipe);
        }
        else
        {
            (*s_g_out)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
        }
        if (a_row1 < rows_total && a_col0 < output_dim)
        {
            cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 1][tx_base + 0],
                               &g_out[a_row1 * output_dim + a_col0], sizeof(float), pipe);
        }
        else
        {
            (*s_g_out)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
        }
        if (a_row1 < rows_total && a_col1 < output_dim)
        {
            cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 1][tx_base + 1],
                               &g_out[a_row1 * output_dim + a_col1], sizeof(float), pipe);
        }
        else
        {
            (*s_g_out)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
        }

        // load weight^T tile (transpose during load)
        int b_row0 = tile * MLP_TILE + (ty_base + 0);
        int b_row1 = tile * MLP_TILE + (ty_base + 1);
        int b_col0 = out_col_base + 0;
        int b_col1 = out_col_base + 1;

        // weight is [input_dim, output_dim], we want weight^T so swap indices
        if (b_col0 < input_dim && b_row0 < output_dim)
        {
            cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 0][tx_base + 0],
                               &weight[b_col0 * output_dim + b_row0], sizeof(float), pipe);
        }
        else
        {
            (*s_weight)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
        }
        if (b_col1 < input_dim && b_row0 < output_dim)
        {
            cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 0][tx_base + 1],
                               &weight[b_col1 * output_dim + b_row0], sizeof(float), pipe);
        }
        else
        {
            (*s_weight)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
        }
        if (b_col0 < input_dim && b_row1 < output_dim)
        {
            cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 1][tx_base + 0],
                               &weight[b_col0 * output_dim + b_row1], sizeof(float), pipe);
        }
        else
        {
            (*s_weight)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
        }
        if (b_col1 < input_dim && b_row1 < output_dim)
        {
            cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 1][tx_base + 1],
                               &weight[b_col1 * output_dim + b_row1], sizeof(float), pipe);
        }
        else
        {
            (*s_weight)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
        }

        pipe.producer_commit();
    }

    // wait for first tile to complete
    pipe.consumer_wait();
    __syncthreads();

    for (int tile = 0; tile < num_tiles; tile++)
    {
        int buf_read = tile & 1;
        buf_write = 1 - buf_read;

        // prefetch next tile while computing current
        if (tile + 1 < num_tiles)
        {
            int next_tile = tile + 1;

            // load g_out tile
            int a_col0 = next_tile * MLP_TILE + (tx_base + 0);
            int a_col1 = next_tile * MLP_TILE + (tx_base + 1);
            int a_row0 = out_row_base + 0;
            int a_row1 = out_row_base + 1;

            pipe.producer_acquire();

            if (a_row0 < rows_total && a_col0 < output_dim)
            {
                cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 0][tx_base + 0],
                                   &g_out[a_row0 * output_dim + a_col0], sizeof(float), pipe);
            }
            else
            {
                (*s_g_out)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
            }
            if (a_row0 < rows_total && a_col1 < output_dim)
            {
                cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 0][tx_base + 1],
                                   &g_out[a_row0 * output_dim + a_col1], sizeof(float), pipe);
            }
            else
            {
                (*s_g_out)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
            }
            if (a_row1 < rows_total && a_col0 < output_dim)
            {
                cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 1][tx_base + 0],
                                   &g_out[a_row1 * output_dim + a_col0], sizeof(float), pipe);
            }
            else
            {
                (*s_g_out)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
            }
            if (a_row1 < rows_total && a_col1 < output_dim)
            {
                cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 1][tx_base + 1],
                                   &g_out[a_row1 * output_dim + a_col1], sizeof(float), pipe);
            }
            else
            {
                (*s_g_out)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
            }

            // load weight^T tile (transpose during load)
            int b_row0 = next_tile * MLP_TILE + (ty_base + 0);
            int b_row1 = next_tile * MLP_TILE + (ty_base + 1);
            int b_col0 = out_col_base + 0;
            int b_col1 = out_col_base + 1;

            if (b_col0 < input_dim && b_row0 < output_dim)
            {
                cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 0][tx_base + 0],
                                   &weight[b_col0 * output_dim + b_row0], sizeof(float), pipe);
            }
            else
            {
                (*s_weight)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
            }
            if (b_col1 < input_dim && b_row0 < output_dim)
            {
                cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 0][tx_base + 1],
                                   &weight[b_col1 * output_dim + b_row0], sizeof(float), pipe);
            }
            else
            {
                (*s_weight)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
            }
            if (b_col0 < input_dim && b_row1 < output_dim)
            {
                cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 1][tx_base + 0],
                                   &weight[b_col0 * output_dim + b_row1], sizeof(float), pipe);
            }
            else
            {
                (*s_weight)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
            }
            if (b_col1 < input_dim && b_row1 < output_dim)
            {
                cuda::memcpy_async(&(*s_weight)[buf_write][ty_base + 1][tx_base + 1],
                                   &weight[b_col1 * output_dim + b_row1], sizeof(float), pipe);
            }
            else
            {
                (*s_weight)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
            }

            pipe.producer_commit();
        }

// compute on current tile from buf_read (overlap)
#pragma unroll
        for (int k = 0; k < MLP_TILE; k++)
        {
            float a0 = (*s_g_out)[buf_read][ty_base + 0][k];
            float a1 = (*s_g_out)[buf_read][ty_base + 1][k];

            float b0 = (*s_weight)[buf_read][k][tx_base + 0];
            float b1 = (*s_weight)[buf_read][k][tx_base + 1];

            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        // release current buffer and wait for next tile to finish loading
        pipe.consumer_release();
        if (tile + 1 < num_tiles)
        {
            pipe.consumer_wait();
        }
        __syncthreads();
    }

    // store results (no bias in backward input)
    if (out_row_base + 0 < rows_total)
    {
        int r0 = out_row_base + 0;
        if (out_col_base + 0 < input_dim)
            g_input[r0 * input_dim + (out_col_base + 0)] = acc00;
        if (out_col_base + 1 < input_dim)
            g_input[r0 * input_dim + (out_col_base + 1)] = acc01;
    }
    if (out_row_base + 1 < rows_total)
    {
        int r1 = out_row_base + 1;
        if (out_col_base + 0 < input_dim)
            g_input[r1 * input_dim + (out_col_base + 0)] = acc10;
        if (out_col_base + 1 < input_dim)
            g_input[r1 * input_dim + (out_col_base + 1)] = acc11;
    }
}

template <int MLP_TILE>
__global__ void mlp_backward_input(float *g_input, const float *g_out, const float *weight,
                                   int batch_size, int seq_len, int input_dim, int output_dim)
{
    extern __shared__ float shared_mem[];

    mlp_backward_input_device<MLP_TILE>(
        g_input,
        g_out,
        weight,
        batch_size,
        seq_len,
        input_dim,
        output_dim,
        blockIdx.x,
        blockIdx.y,
        shared_mem);
}

/*
 * Double buffered, warp microtiling, f32, cuda::pipeline - Backward Weight.
 * Computes: g_weight = input^T @ g_out
 */
template <int MLP_TILE>
__device__ void mlp_backward_weight_device(
    float *g_weight,
    float *g_bias,
    const float *__restrict__ g_out,
    const float *__restrict__ input,
    int batch_size,
    int seq_len,
    int input_dim,
    int output_dim,
    int blockIdx_x,
    int blockIdx_y,
    float *shared_mem)
{
    static_assert(MLP_TILE % 2 == 0, "MLP_TILE must be even for 2x2 micro-tiling.");

    // double buffered tiles: [2][MLP_TILE][MLP_TILE]
    float (*s_input)[2][MLP_TILE][MLP_TILE] = (float (*)[2][MLP_TILE][MLP_TILE])shared_mem;
    float (*s_g_out)[2][MLP_TILE][MLP_TILE] = (float (*)[2][MLP_TILE][MLP_TILE])(shared_mem + 2 * MLP_TILE * MLP_TILE);

    // launch (MLP_TILE/2)*(MLP_TILE/2) threads per block
    //   eg for MLP_TILE=32 => 16*16 = 256 threads (8 warps)
    constexpr int TX_THREADS = MLP_TILE / 2;
    // constexpr int TY_THREADS = MLP_TILE / 2;

    // map threadIdx.x into a base (tx, ty) for a 2x2 microtile
    int t = threadIdx.x;
    int tx2 = t % TX_THREADS;
    int ty2 = t / TX_THREADS;

    int tx_base = tx2 * 2;
    int ty_base = ty2 * 2;

    // output tile origin in g_weight
    int out_col_base = blockIdx_x * MLP_TILE + tx_base; // output_dim
    int out_row_base = blockIdx_y * MLP_TILE + ty_base; // input_dim

    // 2x2 register accs per thread
    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    int rows_total = batch_size * seq_len;
    int num_tiles = (rows_total + MLP_TILE - 1) / MLP_TILE;

    // create pipeline with per-thread scope
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // prefetch first tile into buffer 0
    int buf_write = 0;
    {
        int tile = 0;

        // load input^T tile (transpose during load)
        int a_col0 = tile * MLP_TILE + (tx_base + 0);
        int a_col1 = tile * MLP_TILE + (tx_base + 1);
        int a_row0 = out_row_base + 0;
        int a_row1 = out_row_base + 1;

        pipe.producer_acquire();

        // input is [rows_total, input_dim], we want input^T so swap indices
        if (a_col0 < rows_total && a_row0 < input_dim)
        {
            cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 0][tx_base + 0],
                               &input[a_col0 * input_dim + a_row0], sizeof(float), pipe);
        }
        else
        {
            (*s_input)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
        }
        if (a_col1 < rows_total && a_row0 < input_dim)
        {
            cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 0][tx_base + 1],
                               &input[a_col1 * input_dim + a_row0], sizeof(float), pipe);
        }
        else
        {
            (*s_input)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
        }
        if (a_col0 < rows_total && a_row1 < input_dim)
        {
            cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 1][tx_base + 0],
                               &input[a_col0 * input_dim + a_row1], sizeof(float), pipe);
        }
        else
        {
            (*s_input)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
        }
        if (a_col1 < rows_total && a_row1 < input_dim)
        {
            cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 1][tx_base + 1],
                               &input[a_col1 * input_dim + a_row1], sizeof(float), pipe);
        }
        else
        {
            (*s_input)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
        }

        // load g_out tile
        int b_row0 = tile * MLP_TILE + (ty_base + 0);
        int b_row1 = tile * MLP_TILE + (ty_base + 1);
        int b_col0 = out_col_base + 0;
        int b_col1 = out_col_base + 1;

        if (b_row0 < rows_total && b_col0 < output_dim)
        {
            cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 0][tx_base + 0],
                               &g_out[b_row0 * output_dim + b_col0], sizeof(float), pipe);
        }
        else
        {
            (*s_g_out)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
        }
        if (b_row0 < rows_total && b_col1 < output_dim)
        {
            cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 0][tx_base + 1],
                               &g_out[b_row0 * output_dim + b_col1], sizeof(float), pipe);
        }
        else
        {
            (*s_g_out)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
        }
        if (b_row1 < rows_total && b_col0 < output_dim)
        {
            cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 1][tx_base + 0],
                               &g_out[b_row1 * output_dim + b_col0], sizeof(float), pipe);
        }
        else
        {
            (*s_g_out)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
        }
        if (b_row1 < rows_total && b_col1 < output_dim)
        {
            cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 1][tx_base + 1],
                               &g_out[b_row1 * output_dim + b_col1], sizeof(float), pipe);
        }
        else
        {
            (*s_g_out)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
        }

        pipe.producer_commit();
    }

    // wait for first tile to complete
    pipe.consumer_wait();
    __syncthreads();

    for (int tile = 0; tile < num_tiles; tile++)
    {
        int buf_read = tile & 1;
        buf_write = 1 - buf_read;

        // prefetch next tile while computing current
        if (tile + 1 < num_tiles)
        {
            int next_tile = tile + 1;

            // load input^T tile (transpose during load)
            int a_col0 = next_tile * MLP_TILE + (tx_base + 0);
            int a_col1 = next_tile * MLP_TILE + (tx_base + 1);
            int a_row0 = out_row_base + 0;
            int a_row1 = out_row_base + 1;

            pipe.producer_acquire();

            if (a_col0 < rows_total && a_row0 < input_dim)
            {
                cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 0][tx_base + 0],
                                   &input[a_col0 * input_dim + a_row0], sizeof(float), pipe);
            }
            else
            {
                (*s_input)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
            }
            if (a_col1 < rows_total && a_row0 < input_dim)
            {
                cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 0][tx_base + 1],
                                   &input[a_col1 * input_dim + a_row0], sizeof(float), pipe);
            }
            else
            {
                (*s_input)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
            }
            if (a_col0 < rows_total && a_row1 < input_dim)
            {
                cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 1][tx_base + 0],
                                   &input[a_col0 * input_dim + a_row1], sizeof(float), pipe);
            }
            else
            {
                (*s_input)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
            }
            if (a_col1 < rows_total && a_row1 < input_dim)
            {
                cuda::memcpy_async(&(*s_input)[buf_write][ty_base + 1][tx_base + 1],
                                   &input[a_col1 * input_dim + a_row1], sizeof(float), pipe);
            }
            else
            {
                (*s_input)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
            }

            // load g_out tile
            int b_row0 = next_tile * MLP_TILE + (ty_base + 0);
            int b_row1 = next_tile * MLP_TILE + (ty_base + 1);
            int b_col0 = out_col_base + 0;
            int b_col1 = out_col_base + 1;

            if (b_row0 < rows_total && b_col0 < output_dim)
            {
                cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 0][tx_base + 0],
                                   &g_out[b_row0 * output_dim + b_col0], sizeof(float), pipe);
            }
            else
            {
                (*s_g_out)[buf_write][ty_base + 0][tx_base + 0] = 0.0f;
            }
            if (b_row0 < rows_total && b_col1 < output_dim)
            {
                cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 0][tx_base + 1],
                                   &g_out[b_row0 * output_dim + b_col1], sizeof(float), pipe);
            }
            else
            {
                (*s_g_out)[buf_write][ty_base + 0][tx_base + 1] = 0.0f;
            }
            if (b_row1 < rows_total && b_col0 < output_dim)
            {
                cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 1][tx_base + 0],
                                   &g_out[b_row1 * output_dim + b_col0], sizeof(float), pipe);
            }
            else
            {
                (*s_g_out)[buf_write][ty_base + 1][tx_base + 0] = 0.0f;
            }
            if (b_row1 < rows_total && b_col1 < output_dim)
            {
                cuda::memcpy_async(&(*s_g_out)[buf_write][ty_base + 1][tx_base + 1],
                                   &g_out[b_row1 * output_dim + b_col1], sizeof(float), pipe);
            }
            else
            {
                (*s_g_out)[buf_write][ty_base + 1][tx_base + 1] = 0.0f;
            }

            pipe.producer_commit();
        }

// compute on current tile from buf_read (overlap)
#pragma unroll
        for (int k = 0; k < MLP_TILE; k++)
        {
            float a0 = (*s_input)[buf_read][ty_base + 0][k];
            float a1 = (*s_input)[buf_read][ty_base + 1][k];

            float b0 = (*s_g_out)[buf_read][k][tx_base + 0];
            float b1 = (*s_g_out)[buf_read][k][tx_base + 1];

            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        // release current buffer and wait for next tile to finish loading
        pipe.consumer_release();
        if (tile + 1 < num_tiles)
        {
            pipe.consumer_wait();
        }
        __syncthreads();
    }

    // store weight gradient results
    if (out_row_base + 0 < input_dim)
    {
        int r0 = out_row_base + 0;
        if (out_col_base + 0 < output_dim)
            g_weight[r0 * output_dim + (out_col_base + 0)] = acc00;
        if (out_col_base + 1 < output_dim)
            g_weight[r0 * output_dim + (out_col_base + 1)] = acc01;
    }
    if (out_row_base + 1 < input_dim)
    {
        int r1 = out_row_base + 1;
        if (out_col_base + 0 < output_dim)
            g_weight[r1 * output_dim + (out_col_base + 0)] = acc10;
        if (out_col_base + 1 < output_dim)
            g_weight[r1 * output_dim + (out_col_base + 1)] = acc11;
    }

    // compute bias gradient - only in first row of blocks
    if (ty2 == 0 && blockIdx_y == 0 && g_bias != nullptr)
    {
        int col0 = out_col_base + 0;
        int col1 = out_col_base + 1;

        if (col0 < output_dim)
        {
            float bias_grad = 0.0f;
            for (int bs = 0; bs < rows_total; bs++)
            {
                bias_grad += g_out[bs * output_dim + col0];
            }
            g_bias[col0] = bias_grad;
        }

        if (col1 < output_dim)
        {
            float bias_grad = 0.0f;
            for (int bs = 0; bs < rows_total; bs++)
            {
                bias_grad += g_out[bs * output_dim + col1];
            }
            g_bias[col1] = bias_grad;
        }
    }
}

template <int MLP_TILE>
__global__ void mlp_backward_weight(float *g_weight, float *g_bias, const float *g_out,
                                    const float *input, int batch_size, int seq_len,
                                    int input_dim, int output_dim)
{
    extern __shared__ float shared_mem[];

    mlp_backward_weight_device<MLP_TILE>(
        g_weight,
        g_bias,
        g_out,
        input,
        batch_size,
        seq_len,
        input_dim,
        output_dim,
        blockIdx.x,
        blockIdx.y,
        shared_mem);
}

#endif /* GPT2_LAYERS_MLP_H */
