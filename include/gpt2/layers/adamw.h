#ifndef GPT2_LAYERS_ADAMW_H
#define GPT2_LAYERS_ADAMW_H

#include <cuda_runtime.h>

/**
 * AdamW optimizer kernel for updating model parameters
 * 
 * @param params         [N] model parameters to update (in-place)
 * @param grads          [N] gradients for each parameter
 * @param m              [N] first moment estimates (momentum)
 * @param v              [N] second moment estimates (RMSprop)
 * @param N              number of parameters
 * @param learning_rate  learning rate (alpha)
 * @param beta1          exponential decay rate for first moment
 * @param beta2          exponential decay rate for second moment
 * @param eps            small constant for numerical stability
 * @param weight_decay   weight decay coefficient
 * @param t              current timestep (for bias correction)
 */
__global__ void adamw_kernel(
    float *params,
    const float *grads,
    float *m,
    float *v,
    int N,
    float learning_rate,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int t
);

#endif /* GPT2_LAYERS_ADAMW_H */
