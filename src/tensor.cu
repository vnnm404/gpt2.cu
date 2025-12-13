/* Minimal Tensor Implementation */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gpt2/tensor.h"

#define CHUNK_SIZE 256

tensor_t tensor_alloc(int ndim, const int *shape)
{
    tensor_t tensor;
    tensor.ndim = ndim;
    tensor.data = NULL;

    int size = 1;
    for (int i = 0; i < ndim; i++)
    {
        tensor.shape[i] = shape[i];
        size *= shape[i];
    }

    // Initialize unused shape dimensions to 0
    for (int i = ndim; i < 4; i++)
    {
        tensor.shape[i] = 0;
    }

    // Allocate data on GPU using CUDA
    cudaError_t err = cudaMalloc(&tensor.data, size * sizeof(float));
    if (err != cudaSuccess)
    {
        tensor.data = NULL;
    }

    return tensor;
}

int tensor_size(const tensor_t tensor)
{
    int size = 1;
    for (int i = 0; i < tensor.ndim; i++)
    {
        size *= tensor.shape[i];
    }
    return size;
}

int tensor_load(tensor_t *tensor, FILE *file)
{
    int size = tensor_size(*tensor);
    float buffer[CHUNK_SIZE]; // Stack buffer for chunked reading
    int remaining = size;
    int offset = 0;

    while (remaining > 0)
    {
        int chunk = (remaining < CHUNK_SIZE) ? remaining : CHUNK_SIZE;
        size_t read = fread(buffer, sizeof(float), chunk, file);
        if (read != chunk)
        {
            return -1; // error reading
        }

        // Copy chunk from CPU buffer to GPU memory
        cudaError_t err = cudaMemcpy(tensor->data + offset, buffer,
                                     chunk * sizeof(float),
                                     cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            return -1; // error copying to GPU
        }

        offset += chunk;
        remaining -= chunk;
    }

    return 0; // success
}

void tensor_free(tensor_t *tensor)
{
    if (tensor->data)
    {
        cudaFree(tensor->data);
        tensor->data = NULL;
    }
}
