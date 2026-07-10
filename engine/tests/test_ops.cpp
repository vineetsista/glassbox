// Kernel unit tests against straightforward references.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>
#include <vector>

#include "ops.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("matvec matches naive reference") {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t out = 7, in = 13;
    std::vector<float> w(out * in), x(in), y(out);
    for (auto& v : w) v = dist(rng);
    for (auto& v : x) v = dist(rng);
    gbx::ops::matvec(w.data(), x.data(), y.data(), out, in);
    for (size_t o = 0; o < out; ++o) {
        float ref = 0.f;
        for (size_t i = 0; i < in; ++i) ref += w[o * in + i] * x[i];
        REQUIRE_THAT(y[o], WithinAbs(ref, 1e-5));
    }
}

TEST_CASE("softmax sums to one and is order-preserving") {
    std::vector<float> x = {1.f, 3.f, -2.f, 0.5f};
    gbx::ops::softmax(x.data(), x.size());
    float sum = 0.f;
    for (float v : x) sum += v;
    REQUIRE_THAT(sum, WithinAbs(1.0, 1e-6));
    REQUIRE(x[1] > x[0]);
    REQUIRE(x[0] > x[3]);
    REQUIRE(x[3] > x[2]);
}

TEST_CASE("rmsnorm matches formula") {
    std::vector<float> x = {1.f, -2.f, 3.f, 0.f};
    std::vector<float> w = {1.f, 1.f, 2.f, 1.f};
    std::vector<float> out(4);
    gbx::ops::rmsnorm(x.data(), w.data(), out.data(), 4, 1e-5f);
    float ms = (1.f + 4.f + 9.f + 0.f) / 4.f;
    float scale = 1.f / std::sqrt(ms + 1e-5f);
    REQUIRE_THAT(out[0], WithinAbs(1.f * scale, 1e-6));
    REQUIRE_THAT(out[2], WithinAbs(3.f * scale * 2.f, 1e-6));
}

TEST_CASE("gelu matches tanh approximation at known points") {
    // reference values computed with torch.nn.functional.gelu(x, approximate="tanh")
    REQUIRE_THAT(gbx::ops::gelu(0.f), WithinAbs(0.0, 1e-7));
    REQUIRE_THAT(gbx::ops::gelu(1.f), WithinAbs(0.8411920070648193, 1e-5));
    REQUIRE_THAT(gbx::ops::gelu(-1.f), WithinAbs(-0.15880799293518066, 1e-5));
    REQUIRE_THAT(gbx::ops::gelu(3.f), WithinAbs(2.9963626861572266, 1e-4));
}

TEST_CASE("rope preserves pair norms and position 0 is identity") {
    std::vector<float> x = {0.3f, -0.7f, 1.2f, 0.9f, -0.1f, 0.4f, 0.0f, 1.0f};
    auto orig = x;
    gbx::ops::rope(x.data(), x.size(), 0, 10000.f);
    for (size_t i = 0; i < x.size(); ++i) REQUIRE_THAT(x[i], WithinAbs(orig[i], 1e-6));
    gbx::ops::rope(x.data(), x.size(), 5, 10000.f);
    for (size_t i = 0; i < x.size() / 2; ++i) {
        float n0 = orig[2 * i] * orig[2 * i] + orig[2 * i + 1] * orig[2 * i + 1];
        float n1 = x[2 * i] * x[2 * i] + x[2 * i + 1] * x[2 * i + 1];
        REQUIRE_THAT(n1, WithinAbs(n0, 1e-5));
    }
}
