/* Minimal Tensor Implementation */

#include <stdio.h>
#include <stdlib.h>
#include "gpt2/tensor.h"

tensor_t *tensor_alloc(int ndim, const int *shape) {
    tensor_t *tensor = (tensor_t *)malloc(sizeof(tensor_t));
    if (tensor == NULL) {
        return NULL;
    }

    tensor->ndim = ndim;
    tensor->shape = (int *)malloc(ndim * sizeof(int));
    if (tensor->shape == NULL) {
        free(tensor);
        return NULL;
    }

    int size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
        size *= shape[i];
    }

    tensor->data = (float *)malloc(size * sizeof(float));
    if (tensor->data == NULL) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    return tensor;
}

int tensor_size(const tensor_t *tensor) {
    int size = 1;
    for (int i = 0; i < tensor->ndim; i++) {
        size *= tensor->shape[i];
    }
    return size;
}

int tensor_load(tensor_t *tensor, FILE *file) {
    int size = tensor_size(tensor);
    size_t read = fread(tensor->data, sizeof(float), size, file);
    if (read != size) {
        return -1; // error reading
    }
    return 0; // success
}

void tensor_free(tensor_t *tensor) {
    if (tensor->data) {
        free(tensor->data);
    }
    if (tensor->shape) {
        free(tensor->shape);
    }
    free(tensor);
}
