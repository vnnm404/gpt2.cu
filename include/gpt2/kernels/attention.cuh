#pragma once
#include "program.cuh"

namespace gpt2 {
// One warp owns a (batch, head, query) row; lanes own the 64 head channels.
// m = B*T, n = channels, k = T. QKV layout is [B,T,3,C], heads have width 64.
__device__ __forceinline__ void attention(const Operation &o, int task, float *scratch) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int heads = o.n / 64, r = task * 8 + warp;
    if (r >= o.m * heads) return;
    int h = r % heads, token = r / heads, q = token % o.k, batch = token / o.k;
    int base = batch * o.k * 3 * o.n + h * 64;
    float *scores = scratch + warp * o.k;
    const float *query = o.p[0] + base + q * 3 * o.n;
    float q0 = query[lane], q1 = query[lane + 32];
    float mx = -INFINITY;
    for (int j = 0; j <= q; ++j) {
        const float *key = o.p[0] + base + j * 3 * o.n + o.n;
        float dot = warp_sum(q0 * key[lane] + q1 * key[lane + 32]) * 0.125f;
        if (lane == 0) scores[j] = dot;
        mx = fmaxf(mx, dot);
    }
    __syncwarp();
    float sum = 0;
    for (int j = lane; j <= q; j += 32) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
    sum = warp_sum(sum);
    __syncwarp();
    for (int j = lane; j < o.k; j += 32) {
        float p = j <= q ? scores[j] / sum : 0.f;
        scores[j] = p;
        o.p[2][r * o.k + j] = p;
    }
    __syncwarp();
    float a = 0, b = 0;
    for (int j = 0; j <= q; ++j) {
        const float *v = o.p[0] + base + j * 3 * o.n + 2 * o.n;
        a += scores[j] * v[lane]; b += scores[j] * v[lane + 32];
    }
    o.p[1][token * o.n + h * 64 + lane] = a;
    o.p[1][token * o.n + h * 64 + lane + 32] = b;
}

__device__ __forceinline__ void attention_backward(const Operation &o, int task, float *scratch) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int heads = o.n / 64, r = task * 8 + warp;
    if (r >= o.m * heads) return;
    int h = r % heads, token = r / heads, q = token % o.k, batch = token / o.k;
    int base = batch * o.k * 3 * o.n + h * 64;
    float *dp = scratch + warp * o.k;
    const float *dy = o.p[1] + token * o.n + h * 64;
    float d0 = dy[lane], d1 = dy[lane + 32], dot = 0;
    for (int j = 0; j <= q; ++j) {
        const float *v = o.p[0] + base + j * 3 * o.n + 2 * o.n;
        float a = warp_sum(v[lane] * d0 + v[lane + 32] * d1);
        dot += a * o.p[2][r * o.k + j];
        if (lane == 0) dp[j] = a;
    }
    __syncwarp();
    for (int j = lane; j < o.k; j += 32) {
        float d = j <= q ? o.p[2][r * o.k + j] * (dp[j] - dot) * 0.125f : 0.f;
        dp[j] = d; o.p[4][r * o.k + j] = d;
    }
    __syncwarp();
    float a = 0, b = 0;
    for (int j = 0; j <= q; ++j) {
        const float *key = o.p[0] + base + j * 3 * o.n + o.n;
        a += dp[j] * key[lane]; b += dp[j] * key[lane + 32];
    }
    o.p[3][base + q * 3 * o.n + lane] = a;
    o.p[3][base + q * 3 * o.n + lane + 32] = b;
}

// Gather dK/dV after dScores are complete. No float atomics or gradient clearing.
__device__ __forceinline__ void attention_kv_backward(const Operation &o, int task) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int heads = o.n / 64, r = task * 8 + warp;
    if (r >= o.m * heads) return;
    int h = r % heads, token = r / heads, key = token % o.k, batch = token / o.k;
    int base = batch * o.k * 3 * o.n + h * 64;
    float k0 = 0, k1 = 0, v0 = 0, v1 = 0;
    for (int q = key; q < o.k; ++q) {
        int at = ((batch * o.k + q) * heads + h) * o.k + key;
        float ds = o.p[4][at], p = o.p[2][at];
        const float *query = o.p[0] + base + q * 3 * o.n;
        const float *dy = o.p[1] + (batch * o.k + q) * o.n + h * 64;
        k0 += ds * query[lane]; k1 += ds * query[lane + 32];
        v0 += p * dy[lane]; v1 += p * dy[lane + 32];
    }
    float *out = o.p[3] + base + key * 3 * o.n;
    out[o.n + lane] = k0; out[o.n + lane + 32] = k1;
    out[2 * o.n + lane] = v0; out[2 * o.n + lane + 32] = v1;
}
} // namespace gpt2
