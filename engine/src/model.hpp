// The GPT forward pass: RoPE, RMSNorm pre-norm, GELU MLP, tied embeddings,
// KV cache for autoregressive generation. Mirrors train/gpt.py exactly.
//
// Steering: if the GBX bundle carries an SAE, the residual stream after
// blocks[hook_layer] can be edited in SAE feature space:
//     z = enc(resid);  resid += sum_f (mult_f - 1) * z_f * W_dec[:, f]
// (error term preserved: only the selected features' contributions change).
// The k active features of the last processed token are always recorded so
// UIs can show live feature activations.
#pragma once

#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "gbx.hpp"
#include "ops.hpp"
#include "sae.hpp"

namespace gbx {

class Model {
public:
    explicit Model(const GBXFile& file)
        : cfg_(file.config()), wte_(file.tensor("wte").data),
          final_norm_(file.tensor("final_norm").data) {
        layers_.reserve(static_cast<size_t>(cfg_.n_layers));
        for (int i = 0; i < cfg_.n_layers; ++i) {
            std::string p = "blocks." + std::to_string(i) + ".";
            layers_.push_back(Layer{
                file.tensor(p + "norm1").data, file.tensor(p + "w_q").data,
                file.tensor(p + "w_k").data, file.tensor(p + "w_v").data,
                file.tensor(p + "w_o").data, file.tensor(p + "norm2").data,
                file.tensor(p + "w_in").data, file.tensor(p + "w_out").data,
            });
        }
        if (file.sae_config().present) sae_.emplace(file, file.sae_config());
        reset();
    }

    const ModelConfig& cfg() const { return cfg_; }
    bool has_sae() const { return sae_.has_value(); }
    const SAE* sae() const { return sae_ ? &*sae_ : nullptr; }

    void reset() {
        pos_ = 0;
        size_t per_layer = static_cast<size_t>(cfg_.ctx_len) * static_cast<size_t>(cfg_.d_model);
        k_cache_.assign(static_cast<size_t>(cfg_.n_layers) * per_layer, 0.f);
        v_cache_.assign(static_cast<size_t>(cfg_.n_layers) * per_layer, 0.f);
        last_features_.clear();
    }

    int pos() const { return pos_; }

    void set_steering(int feature, float multiplier) { steering_[feature] = multiplier; }
    void clear_steering() { steering_.clear(); }
    const std::unordered_map<int, float>& steering() const { return steering_; }

    // features (k-sparse) of the hook layer for the last token processed
    const std::vector<FeatureAct>& last_features() const { return last_features_; }

    // Process one token at the current position; writes vocab_size logits.
    // Returns false if the context window is full.
    bool forward(int token, float* logits) {
        if (pos_ >= cfg_.ctx_len) return false;
        const size_t d = static_cast<size_t>(cfg_.d_model);
        const size_t dh = static_cast<size_t>(cfg_.d_head());
        const size_t h = static_cast<size_t>(cfg_.n_heads);
        const size_t dmlp = static_cast<size_t>(cfg_.d_mlp);

        std::vector<float> x(d), xn(d), q(d), att_out(d), mlp_act(dmlp), scratch(d);
        std::memcpy(x.data(), wte_ + static_cast<size_t>(token) * d, d * sizeof(float));

        for (size_t li = 0; li < layers_.size(); ++li) {
            const Layer& L = layers_[li];
            float* kc = k_cache_.data() + li * static_cast<size_t>(cfg_.ctx_len) * d;
            float* vc = v_cache_.data() + li * static_cast<size_t>(cfg_.ctx_len) * d;

            // attention
            ops::rmsnorm(x.data(), L.norm1, xn.data(), d, cfg_.rms_eps);
            ops::matvec(L.w_q, xn.data(), q.data(), d, d);
            float* k_row = kc + static_cast<size_t>(pos_) * d;
            float* v_row = vc + static_cast<size_t>(pos_) * d;
            ops::matvec(L.w_k, xn.data(), k_row, d, d);
            ops::matvec(L.w_v, xn.data(), v_row, d, d);
            for (size_t hd = 0; hd < h; ++hd) {
                ops::rope(q.data() + hd * dh, dh, static_cast<size_t>(pos_), cfg_.rope_base);
                ops::rope(k_row + hd * dh, dh, static_cast<size_t>(pos_), cfg_.rope_base);
            }

            std::vector<float> scores(static_cast<size_t>(pos_) + 1);
            float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(dh));
            std::fill(att_out.begin(), att_out.end(), 0.f);
            for (size_t hd = 0; hd < h; ++hd) {
                const float* qh = q.data() + hd * dh;
                for (size_t t = 0; t <= static_cast<size_t>(pos_); ++t)
                    scores[t] = ops::dot(qh, kc + t * d + hd * dh, dh) * inv_sqrt;
                ops::softmax(scores.data(), static_cast<size_t>(pos_) + 1);
                float* zh = scratch.data() + hd * dh;  // reuse scratch as z
                std::fill(zh, zh + dh, 0.f);
                for (size_t t = 0; t <= static_cast<size_t>(pos_); ++t)
                    ops::axpy(scores[t], vc + t * d + hd * dh, zh, dh);
            }
            // att_out = w_o @ z ; add to residual
            ops::matvec(L.w_o, scratch.data(), att_out.data(), d, d);
            for (size_t i = 0; i < d; ++i) x[i] += att_out[i];

            // mlp
            ops::rmsnorm(x.data(), L.norm2, xn.data(), d, cfg_.rms_eps);
            ops::matvec(L.w_in, xn.data(), mlp_act.data(), dmlp, d);
            ops::gelu_vec(mlp_act.data(), dmlp);
            ops::matvec(L.w_out, mlp_act.data(), scratch.data(), d, dmlp);
            for (size_t i = 0; i < d; ++i) x[i] += scratch[i];

            // SAE hook on resid_post of hook_layer
            if (sae_ && static_cast<int>(li) == sae_->cfg().hook_layer) {
                last_features_ = sae_->encode(x.data());
                if (!steering_.empty()) {
                    for (const auto& a : last_features_) {
                        auto it = steering_.find(a.feature);
                        if (it != steering_.end())
                            sae_->add_feature(a.feature, (it->second - 1.0f) * a.value, x.data());
                    }
                    // amplifying a feature that did not fire: add it from zero
                    for (const auto& [feat, mult] : steering_) {
                        bool active = false;
                        for (const auto& a : last_features_)
                            if (a.feature == feat) { active = true; break; }
                        if (!active && mult > 1.0f)
                            sae_->add_feature(feat, mult, x.data());
                    }
                }
            }
        }

        ops::rmsnorm(x.data(), final_norm_, xn.data(), d, cfg_.rms_eps);
        // tied unembedding: logits[v] = dot(wte[v], xn)
        ops::matvec(wte_, xn.data(), logits, static_cast<size_t>(cfg_.vocab_size), d);
        ++pos_;
        return true;
    }

private:
    struct Layer {
        const float* norm1;
        const float* w_q;
        const float* w_k;
        const float* w_v;
        const float* w_o;
        const float* norm2;
        const float* w_in;
        const float* w_out;
    };

    ModelConfig cfg_;
    const float* wte_;
    const float* final_norm_;
    std::vector<Layer> layers_;
    std::optional<SAE> sae_;
    std::unordered_map<int, float> steering_;
    std::vector<FeatureAct> last_features_;
    std::vector<float> k_cache_, v_cache_;
    int pos_ = 0;
};

}  // namespace gbx
