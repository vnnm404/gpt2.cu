/* AdamW Optimizer Implementation */

#include <cuda_runtime.h>
#include <math.h>
#include "gpt2/layers/adamw.h"

/**
 * AdamW optimizer kernel
 * Reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N) return;
    
    float param = params[idx];
    float grad = grads[idx];
    
    // Update first moment (momentum)
    float m_val = beta1 * m[idx] + (1.0f - beta1) * grad;
    
    // Update second moment (RMSprop)
    float v_val = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    
    // Bias-correct both moments
    float m_hat = m_val / (1.0f - powf(beta1, t));
    float v_hat = v_val / (1.0f - powf(beta2, t));
    
    // Store updated moments
    m[idx] = m_val;
    v[idx] = v_val;
    
    // Update parameter: param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
    params[idx] = param - learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
}
