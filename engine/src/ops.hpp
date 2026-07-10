// Hand-rolled numerical kernels. fp32 everywhere, row-major, no dependencies.
// Written as clean loops the compiler can autovectorize (native AVX2 / WASM
// SIMD128 with -msimd128); matvec is the hot path (batch-1 generation).
#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace gbx::ops {

// y[o] = dot(w[o, :], x)  with w: [out, in] row-major (torch Linear layout)
inline void matvec(const float* w, const float* x, float* y, size_t out, size_t in) {
    for (size_t o = 0; o < out; ++o) {
        const float* row = w + o * in;
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        size_t i = 0;
        for (; i + 4 <= in; i += 4) {
            acc0 += row[i] * x[i];
            acc1 += row[i + 1] * x[i + 1];
            acc2 += row[i + 2] * x[i + 2];
            acc3 += row[i + 3] * x[i + 3];
        }
        float acc = (acc0 + acc1) + (acc2 + acc3);
        for (; i < in; ++i) acc += row[i] * x[i];
        y[o] = acc;
    }
}

// y += a * x  (axpy)
inline void axpy(float a, const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] += a * x[i];
}

inline float dot(const float* a, const float* b, size_t n) {
    float acc = 0.f;
    for (size_t i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

// x <- x * rsqrt(mean(x^2) + eps) * w   (RMSNorm, out-of-place)
inline void rmsnorm(const float* x, const float* w, float* out, size_t n, float eps) {
    float ss = 0.f;
    for (size_t i = 0; i < n; ++i) ss += x[i] * x[i];
    float scale = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (size_t i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

// RoPE on one head vector (interleaved even/odd pairs), position pos.
inline void rope(float* x, size_t d_head, size_t pos, float base) {
    for (size_t i = 0; i < d_head / 2; ++i) {
        float inv_freq = std::pow(base, -2.0f * static_cast<float>(i) / static_cast<float>(d_head));
        float angle = static_cast<float>(pos) * inv_freq;
        float c = std::cos(angle), s = std::sin(angle);
        float x0 = x[2 * i], x1 = x[2 * i + 1];
        x[2 * i] = x0 * c - x1 * s;
        x[2 * i + 1] = x0 * s + x1 * c;
    }
}

// in-place softmax over n values
inline void softmax(float* x, size_t n) {
    float mx = x[0];
    for (size_t i = 1; i < n; ++i) mx = x[i] > mx ? x[i] : mx;
    float sum = 0.f;
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; ++i) x[i] *= inv;
}

// GELU, tanh approximation — must match torch F.gelu(approximate="tanh")
inline float gelu(float x) {
    constexpr float k = 0.7978845608028654f;  // sqrt(2/pi)
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x3)));
}

inline void gelu_vec(float* x, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] = gelu(x[i]);
}

}  // namespace gbx::ops
