// Top-k SAE runtime: encode (k-sparse codes), decode, and steering deltas.
// Matches sae/model.py:  z = topk(relu(W_enc (x - b_dec) + b_enc), k)
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "gbx.hpp"
#include "ops.hpp"

namespace gbx {

struct FeatureAct {
    int feature;
    float value;
};

class SAE {
public:
    SAE(const GBXFile& file, const SAEConfig& cfg)
        : cfg_(cfg),
          w_enc_(file.tensor("sae.w_enc").data),
          b_enc_(file.tensor("sae.b_enc").data),
          w_dec_(file.tensor("sae.w_dec").data),
          b_dec_(file.tensor("sae.b_dec").data) {}

    const SAEConfig& cfg() const { return cfg_; }

    // returns the k active (feature, value) pairs, sorted by feature id
    std::vector<FeatureAct> encode(const float* x) const {
        const size_t f = static_cast<size_t>(cfg_.n_features);
        const size_t d = static_cast<size_t>(cfg_.d_in);
        std::vector<float> centered(d);
        for (size_t i = 0; i < d; ++i) centered[i] = x[i] - b_dec_[i];
        std::vector<float> pre(f);
        ops::matvec(w_enc_, centered.data(), pre.data(), f, d);
        std::vector<FeatureAct> acts;
        acts.reserve(f / 4);
        for (size_t j = 0; j < f; ++j) {
            float v = pre[j] + b_enc_[j];
            if (v > 0.f) acts.push_back({static_cast<int>(j), v});
        }
        const size_t k = static_cast<size_t>(cfg_.k);
        if (acts.size() > k) {
            std::partial_sort(acts.begin(), acts.begin() + static_cast<std::ptrdiff_t>(k),
                              acts.end(), [](const FeatureAct& a, const FeatureAct& b) {
                                  return a.value > b.value ||
                                         (a.value == b.value && a.feature < b.feature);
                              });
            acts.resize(k);
        }
        std::sort(acts.begin(), acts.end(),
                  [](const FeatureAct& a, const FeatureAct& b) { return a.feature < b.feature; });
        return acts;
    }

    // xhat = W_dec z + b_dec  from sparse codes
    void decode(const std::vector<FeatureAct>& acts, float* out) const {
        const size_t d = static_cast<size_t>(cfg_.d_in);
        for (size_t i = 0; i < d; ++i) out[i] = b_dec_[i];
        for (const auto& a : acts) add_feature(a.feature, a.value, out);
    }

    // x += (mult - 1) * z_f * W_dec[:, f]   — steering with error preservation
    void add_feature(int feature, float coef, float* x) const {
        const size_t d = static_cast<size_t>(cfg_.d_in);
        const size_t f = static_cast<size_t>(cfg_.n_features);
        // w_dec_ is [d_in, n_features] row-major -> column f is strided
        for (size_t i = 0; i < d; ++i) x[i] += coef * w_dec_[i * f + static_cast<size_t>(feature)];
    }

private:
    SAEConfig cfg_;
    const float* w_enc_;
    const float* b_enc_;
    const float* w_dec_;
    const float* b_dec_;
};

}  // namespace gbx
