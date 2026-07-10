// Deterministic sampler: xorshift128+ RNG, temperature + top-k.
// Not required to match Python sampling (parity contract is on logits);
// determinism per (seed, logits stream) is required and tested.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace gbx {

class Rng {
public:
    explicit Rng(uint64_t seed) {
        // splitmix64 to spread the seed
        auto mix = [](uint64_t& z) {
            z += 0x9e3779b97f4a7c15ULL;
            uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            return x ^ (x >> 31);
        };
        uint64_t z = seed;
        s0_ = mix(z);
        s1_ = mix(z);
        if (s0_ == 0 && s1_ == 0) s1_ = 1;
    }
    // uniform in [0, 1)
    double uniform() {
        uint64_t x = s0_, y = s1_;
        s0_ = y;
        x ^= x << 23;
        s1_ = x ^ y ^ (x >> 17) ^ (y >> 26);
        uint64_t r = s1_ + y;
        return static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);
    }

private:
    uint64_t s0_, s1_;
};

struct SampleParams {
    float temperature = 0.8f;
    int top_k = 40;
};

inline int sample(const float* logits, int vocab, const SampleParams& p, Rng& rng) {
    if (p.temperature <= 0.f) {
        int best = 0;
        for (int i = 1; i < vocab; ++i)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
    std::vector<std::pair<float, int>> cand(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) cand[static_cast<size_t>(i)] = {logits[i], i};
    int k = (p.top_k > 0 && p.top_k < vocab) ? p.top_k : vocab;
    std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    cand.resize(static_cast<size_t>(k));
    float mx = cand[0].first;
    double sum = 0.0;
    std::vector<double> probs(cand.size());
    for (size_t i = 0; i < cand.size(); ++i) {
        probs[i] = std::exp(static_cast<double>((cand[i].first - mx) / p.temperature));
        sum += probs[i];
    }
    double r = rng.uniform() * sum;
    double acc = 0.0;
    for (size_t i = 0; i < cand.size(); ++i) {
        acc += probs[i];
        if (r < acc) return cand[i].second;
    }
    return cand.back().second;
}

}  // namespace gbx
