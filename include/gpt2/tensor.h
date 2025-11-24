#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

typedef struct {
    float *data;
    int *shape;
    int ndim;
} tensor_t;

tensor_t *tensor_alloc(int ndim, const int *shape);
int tensor_size(const tensor_t *tensor);
void tensor_free(tensor_t *tensor);

#endif /* TENSOR_TENSOR_H */