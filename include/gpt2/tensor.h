#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

typedef struct {
    float *data;
    int shape[4];
    int ndim;
} tensor_t;

tensor_t tensor_alloc(int ndim, const int *shape);
int tensor_size(const tensor_t tensor);
int tensor_load(tensor_t *tensor, FILE *file);
int tensor_dump(const tensor_t *tensor, FILE *file);
void tensor_free(tensor_t *tensor);

#endif /* TENSOR_TENSOR_H */