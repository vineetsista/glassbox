// Engine: bundles GBX file + model + tokenizer behind one object, with the
// generation loop used by both the CLI and the WASM API.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "gbx.hpp"
#include "model.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp"

namespace gbx {

class Engine {
public:
    explicit Engine(GBXFile file)
        : file_(std::make_unique<GBXFile>(std::move(file))),
          model_(*file_),
          tokenizer_(file_->tokenizer_merges(), file_->tokenizer_vocab_size()) {}

    static Engine from_file(const std::string& path) { return Engine(GBXFile::from_file(path)); }

    Model& model() { return model_; }
    const Tokenizer& tokenizer() const { return tokenizer_; }

    // Feed prompt tokens (no sampling). Returns logits of the last token.
    std::vector<float> prefill(const std::vector<int>& ids) {
        std::vector<float> logits(static_cast<size_t>(model_.cfg().vocab_size));
        model_.reset();
        for (int id : ids)
            if (!model_.forward(id, logits.data())) break;
        return logits;
    }

    // Generate up to max_new tokens; on_token fires after each new token with
    // (token_id, features_of_hook_layer). Stops at EOT or a full context.
    std::string generate(const std::string& prompt, int max_new, const SampleParams& params,
                         uint64_t seed,
                         const std::function<void(int, const std::vector<FeatureAct>&)>& on_token =
                             nullptr) {
        std::vector<int> ids = tokenizer_.encode(prompt);
        if (ids.empty()) ids.push_back(tokenizer_.eot_id());
        std::vector<float> logits = prefill(ids);
        Rng rng(seed);
        for (int i = 0; i < max_new; ++i) {
            if (model_.pos() >= model_.cfg().ctx_len) break;
            int next = sample(logits.data(), model_.cfg().vocab_size, params, rng);
            if (next == tokenizer_.eot_id()) break;
            ids.push_back(next);
            if (on_token) on_token(next, model_.last_features());
            if (!model_.forward(next, logits.data())) break;
        }
        return tokenizer_.decode(ids);
    }

private:
    std::unique_ptr<GBXFile> file_;  // owns tensor memory; must outlive model_
    Model model_;
    Tokenizer tokenizer_;
};

}  // namespace gbx
