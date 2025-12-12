#pragma once

#include <cuda_runtime.h>
#include <math.h>

// Each thread computes attention for one (batch, time, head) position
// out: [B, T, C]
// preatt: [B, NH, T, T]
// att: [B, NH, T, T]
// inp: [B, T, 3*C] where Q,K,V are concatenated
__device__ void attention_forward_device(float* out, float* preatt, float* att,
                                         const float* inp,
                                         int B, int T, int NH, int C,
                                         int blockIdx_x) {
    int idx = blockIdx_x * blockDim.x + threadIdx.x;
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
        float expv = __expf(preatt_bth[t2] - maxval);
        expsum += expv;
        att_bth[t2] = expv;
    }
    float expsum_inv = (expsum == 0.0f) ? 0.0f : 1.0f / expsum;
    
    // Pass 3: Normalize, set causal mask, and accumulate values (fused)
    float* out_bth = out + b * T * C + t * C + h * hs;
    for (int i = 0; i < hs; i++) out_bth[i] = 0.0f;
    
    for (int t2 = 0; t2 <= t; t2++) {
        float att_btht2 = att_bth[t2] * expsum_inv;
        att_bth[t2] = att_btht2;
        const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
        for (int i = 0; i < hs; i++) {
            out_bth[i] += att_btht2 * value_t2[i];
        }
    }
    // Set causal mask for remaining positions
    for (int t2 = t + 1; t2 < T; t2++) att_bth[t2] = 0.0f;
}

// Each thread handles one (batch, time, head) position
// g_inp: [B, T, 3*C] gradients for Q,K,V
// g_preatt: [B, NH, T, T] gradients for pre-attention scores
// g_att: [B, NH, T, T] gradients for attention weights
// g_out: [B, T, C] gradients from output
// inp: [B, T, 3*C] input Q,K,V from forward pass
// att: [B, NH, T, T] attention weights from forward pass
__device__ void attention_backward_device(float* g_inp, float* g_preatt, float* g_att,
                                   const float* g_out, const float* inp, const float* att,
                                   int B, int T, int C, int NH,
                                   int blockIdx_x) {
    int idx = blockIdx_x * blockDim.x + threadIdx.x;
    if (idx >= B * T * NH) return;
    
    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (NH * T);
    int hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);
    
    // Compute g_att[t, t2] = dot(value[t2], g_out[t]) - direct write, no atomic needed
    for (int t2 = 0; t2 <= t; t2++) {
        float acc = 0.0f;
        for (int i = 0; i < hs; i++) {
            acc += inp[(b*T+t2)*C*3 + h*hs + C*2 + i] * g_out[(b*T+t)*C + h*hs + i];
        }
        g_att[(b*NH+h)*T*T + t*T + t2] = acc;
    }
    
    // Compute g_preatt using O(n) softmax backward - direct write, no atomic needed
    float ds = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        ds += att[(b*NH+h)*T*T + t*T + t2] * g_att[(b*NH+h)*T*T + t*T + t2];
    }
    for (int t2 = 0; t2 <= t; t2++) {
        g_preatt[(b*NH+h)*T*T + t*T + t2] = att[(b*NH+h)*T*T + t*T + t2] * (g_att[(b*NH+h)*T*T + t*T + t2] - ds);
    }
    
    // g_query[t]: each thread owns its g_query[t], no atomic needed
    for (int i = 0; i < hs; i++) {
        float g_q = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            g_q += inp[(b*T+t2)*C*3 + h*hs + C + i] * g_preatt[(b*NH+h)*T*T + t*T + t2];
        }
        g_inp[(b*T+t)*C*3 + h*hs + i] = g_q * scale;
    }
    
    // g_key[t] and g_value[t]: need to sum over all tp >= t, use atomics
    for (int t2 = 0; t2 <= t; t2++) {
        float g_preatt_val = g_preatt[(b*NH+h)*T*T + t*T + t2] * scale;
        float att_val = att[(b*NH+h)*T*T + t*T + t2];
        for (int i = 0; i < hs; i++) {
            atomicAdd(&g_inp[(b*T+t2)*C*3 + h*hs + C + i], inp[(b*T+t)*C*3 + h*hs + i] * g_preatt_val);
            atomicAdd(&g_inp[(b*T+t2)*C*3 + h*hs + C*2 + i], att_val * g_out[(b*T+t)*C + h*hs + i]);
        }
    }
}
