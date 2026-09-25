#pragma once
#include "program.cuh"

namespace gpt2 {
__device__ __forceinline__ float gelu_value(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608f * x * (1.f + 0.044715f * x * x)));
}
__device__ __forceinline__ void elementwise(const Operation &o, int tile) {
    int begin = tile * 4096 + threadIdx.x;
    for (int i = begin; i < min(o.m, (tile + 1) * 4096); i += threads) {
        switch (o.code) {
        case Code::clear: o.p[0][i] = 0; break;
        case Code::sum_splits: {
            float value = 0;
            for (int j = 0; j < o.n; ++j) value += o.p[0][j * o.m + i];
            if (o.p[2]) value += o.p[2][i % o.k];
            if (o.flags & 4) value += o.p[1][i];
            o.p[1][i] = value;
            break;
        }
        case Code::add: o.p[2][i] = o.p[0][i] + (o.p[1] ? o.p[1][i] : 0.f); break;
        case Code::advance:
            if (i == 0) {
                float t = ++o.p[0][0];
                o.p[0][1] = 1.f - powf(0.9f, t);
                o.p[0][2] = 1.f - powf(0.999f, t);
            }
            break;
        case Code::gelu: o.p[1][i] = gelu_value(o.p[0][i]); break;
        case Code::gelu_backward: {
            float x = o.p[0][i];
            float t = tanhf(0.7978845608f * x * (1.f + 0.044715f * x * x));
            o.p[2][i] = o.p[1][i] * (0.5f * (1.f + t) +
                0.5f * x * (1.f - t * t) * 0.7978845608f * (1.f + 0.134145f * x * x));
            break;
        }
        case Code::embedding: {
            int r = i / o.n, c = i % o.n;
            int token = reinterpret_cast<const int *>(o.p[0])[r];
            o.p[3][i] = o.p[1][token * o.n + c] + o.p[2][(r % o.k) * o.n + c];
            break;
        }
        case Code::embedding_backward: {
            int r = i / o.n, c = i % o.n;
            int token = reinterpret_cast<const int *>(o.p[0])[r];
            atomicAdd(o.p[2] + token * o.n + c, o.p[1][i]);
            atomicAdd(o.p[3] + (r % o.k) * o.n + c, o.p[1][i]);
            break;
        }
        case Code::adamw: {
            float g = o.p[1][i];
            float m = 0.9f * o.p[2][i] + 0.1f * g;
            float v = 0.999f * o.p[3][i] + 0.001f * g * g;
            o.p[2][i] = m;
            o.p[3][i] = v;
            // scalar: learning rate, weight decay, first/second bias correction.
            o.p[0][i] = o.p[0][i] * (1.f - o.scalar[0] * o.scalar[1]) -
                o.scalar[0] * (m / o.p[4][1]) / (sqrtf(v / o.p[4][2]) + 1e-8f);
            break;
        }
        default: break;
        }
    }
}

__device__ __forceinline__ void norm(const Operation &o, int row, float *s) {
    int t = threadIdx.x, C = o.n;
    const float *x = o.p[0] + row * C;
    float sum = 0;
    for (int c = t; c < C; c += threads) sum += x[c];
    float mean = reduce(sum, s) / C;
    sum = 0;
    for (int c = t; c < C; c += threads) sum += (x[c] - mean) * (x[c] - mean);
    float rstd = rsqrtf(reduce(sum, s) / C + 1e-5f);
    for (int c = t; c < C; c += threads)
        o.p[3][row * C + c] = (x[c] - mean) * rstd * o.p[1][c] + o.p[2][c];
    if (t == 0) { o.p[4][row * 2] = mean; o.p[4][row * 2 + 1] = rstd; }
}

__device__ __forceinline__ void norm_backward(const Operation &o, int row, float *s) {
    int t = threadIdx.x, C = o.n;
    float mean = o.p[3][row * 2], rstd = o.p[3][row * 2 + 1];
    float sum = 0, prod = 0;
    for (int c = t; c < C; c += threads) {
        float d = o.p[1][row * C + c] * o.p[2][c];
        sum += d;
        prod += d * (o.p[0][row * C + c] - mean) * rstd;
    }
    sum = reduce(sum, s) / C;
    prod = reduce(prod, s) / C;
    for (int c = t; c < C; c += threads) {
        int i = row * C + c;
        float dx = rstd * (o.p[1][i] * o.p[2][c] - sum - (o.p[0][i] - mean) * rstd * prod);
        o.p[4][i] = dx + (o.p[5] ? o.p[5][i] : 0.f);
    }
}

__device__ __forceinline__ void sum_rows(const Operation &o, int tile) {
    int c = tile * threads + threadIdx.x;
    if (c >= o.n) return;
    float a = 0, b = 0;
    for (int r = 0; r < o.m; ++r) {
        float g = o.p[0][r * o.n + c];
        a += g;
        if (o.code == Code::norm_parameters)
            b += g * (o.p[1][r * o.n + c] - o.p[2][r * 2]) * o.p[2][r * 2 + 1];
    }
    o.p[3][c] = a * (o.scalar[0] == 0.f ? 1.f : o.scalar[0]);
    if (o.code == Code::norm_parameters) o.p[4][c] = b;
}

// Stable log-sum-exp loss and its logits gradient, one block per token.
__device__ __forceinline__ void cross_entropy(const Operation &o, int row, float *s) {
    float mx = -INFINITY, sum = 0;
    int vocab = o.k ? o.k : o.n;
    const float *x = o.p[0] + row * o.n;
    for (int c = threadIdx.x; c < vocab; c += threads) mx = fmaxf(mx, x[c]);
    mx = reduce<true>(mx, s);
    for (int c = threadIdx.x; c < vocab; c += threads) sum += expf(x[c] - mx);
    sum = reduce(sum, s);
    int target = reinterpret_cast<const int *>(o.p[1])[row];
    if (threadIdx.x == 0) o.p[3][row] = logf(sum) + mx - x[target];
    for (int c = threadIdx.x; c < o.n; c += threads)
        o.p[2][row * o.n + c] = c < vocab ? (expf(x[c] - mx) / sum - float(c == target)) / o.m : 0.f;
}
} // namespace gpt2
