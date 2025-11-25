/* Attention layer implementation - CUDA implementation */

#include "gpt2/layers/attention.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// CUDA kernel for attention forward pass
// Each thread computes attention for one (batch, time, head) position
// out: [B, T, C]
// preatt: [B, NH, T, T]
// att: [B, NH, T, T]
// inp: [B, T, 3*C] where Q,K,V are concatenated
__global__ void attention_forward(float* out, float* preatt, float* att,
                                         const float* inp,
                                         int B, int T, int NH, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * NH;
    
    if (idx >= total_threads) return;
    
    // Decode indices: idx = b * T * NH + t * NH + h
    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (NH * T);
    
    int C3 = C * 3;
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf((float)hs);
    
    // Pointers to query for this head
    const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
    float* preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
    float* att_bth = att + b * NH * T * T + h * T * T + t * T;
    
    // Pass 1: Calculate query dot key and find maxval
    float maxval = -10000.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C for key offset
        
        // Compute dot product: query_t · key_t2
        float val = 0.0f;
        for (int i = 0; i < hs; i++) {
            val += query_t[i] * key_t2[i];
        }
        val *= scale;
        
        if (val > maxval) {
            maxval = val;
        }
        preatt_bth[t2] = val;
    }
    
    // Pass 2: Calculate exp and sum for softmax
    float expsum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        float expv = expf(preatt_bth[t2] - maxval);
        expsum += expv;
        att_bth[t2] = expv;
    }
    float expsum_inv = (expsum == 0.0f) ? 0.0f : 1.0f / expsum;
    
    // Pass 3: Normalize to get softmax (with causal mask)
    for (int t2 = 0; t2 < T; t2++) {
        if (t2 <= t) {
            att_bth[t2] *= expsum_inv;
        } else {
            att_bth[t2] = 0.0f; // Causal mask
        }
    }
    
    // Pass 4: Accumulate weighted values into output
    float* out_bth = out + b * T * C + t * C + h * hs;
    
    // Initialize output to zero
    for (int i = 0; i < hs; i++) {
        out_bth[i] = 0.0f;
    }
    
    // Accumulate weighted values
    for (int t2 = 0; t2 <= t; t2++) {
        const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 for value offset
        float att_btht2 = att_bth[t2];
        for (int i = 0; i < hs; i++) {
            out_bth[i] += att_btht2 * value_t2[i];
        }
    }
}
