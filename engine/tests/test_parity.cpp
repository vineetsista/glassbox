// Logit parity against PyTorch (committed fixtures, <=1e-3 abs, fp32) and
// SAE code parity. The contract of the whole engine lives here.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

#include "engine.hpp"

namespace {

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    REQUIRE(f);
    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(size);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
    return buf;
}

const std::string FIX = FIXTURES_DIR;

}  // namespace

TEST_CASE("logits match pytorch within 1e-3") {
    auto engine = gbx::Engine::from_file(FIX + "/tiny_model.gbx");
    auto tok_bytes = read_bytes(FIX + "/tiny_gpt_tokens.u16");
    auto ref_bytes = read_bytes(FIX + "/tiny_gpt_logits.f32");

    const int vocab = engine.model().cfg().vocab_size;
    const size_t seq = 16;
    size_t n_tokens = tok_bytes.size() / 2;
    REQUIRE(n_tokens % seq == 0);
    std::vector<uint16_t> tokens(n_tokens);
    std::memcpy(tokens.data(), tok_bytes.data(), tok_bytes.size());
    const float* ref = reinterpret_cast<const float*>(ref_bytes.data());
    REQUIRE(ref_bytes.size() == n_tokens * static_cast<size_t>(vocab) * sizeof(float));

    float max_diff = 0.f;
    std::vector<float> logits(static_cast<size_t>(vocab));
    for (size_t row = 0; row < n_tokens / seq; ++row) {
        engine.model().reset();
        for (size_t t = 0; t < seq; ++t) {
            REQUIRE(engine.model().forward(tokens[row * seq + t], logits.data()));
            const float* r = ref + (row * seq + t) * static_cast<size_t>(vocab);
            for (int v = 0; v < vocab; ++v)
                max_diff = std::max(max_diff, std::fabs(logits[static_cast<size_t>(v)] - r[v]));
        }
    }
    INFO("max abs logit diff: " << max_diff);
    REQUIRE(max_diff <= 1e-3f);
}

TEST_CASE("sae encode/decode match pytorch fixture") {
    auto file = gbx::GBXFile::from_file(FIX + "/tiny_model.gbx");
    REQUIRE(file.sae_config().present);
    gbx::SAE sae(file, file.sae_config());

    auto x_bytes = read_bytes(FIX + "/tiny_sae_input.f32");
    auto z_bytes = read_bytes(FIX + "/tiny_sae_z.f32");
    auto xhat_bytes = read_bytes(FIX + "/tiny_sae_xhat.f32");
    const float* x = reinterpret_cast<const float*>(x_bytes.data());
    const float* z_ref = reinterpret_cast<const float*>(z_bytes.data());
    const float* xhat_ref = reinterpret_cast<const float*>(xhat_bytes.data());

    const size_t d = static_cast<size_t>(file.sae_config().d_in);
    const size_t nf = static_cast<size_t>(file.sae_config().n_features);
    const size_t rows = x_bytes.size() / sizeof(float) / d;
    REQUIRE(rows == 4);

    for (size_t r = 0; r < rows; ++r) {
        auto acts = sae.encode(x + r * d);
        // sparse acts must match the dense fixture row
        std::vector<float> dense(nf, 0.f);
        for (const auto& a : acts) dense[static_cast<size_t>(a.feature)] = a.value;
        for (size_t j = 0; j < nf; ++j) REQUIRE(std::fabs(dense[j] - z_ref[r * nf + j]) <= 1e-4f);

        std::vector<float> xhat(d);
        sae.decode(acts, xhat.data());
        for (size_t i = 0; i < d; ++i) REQUIRE(std::fabs(xhat[i] - xhat_ref[r * d + i]) <= 1e-4f);
    }
}

TEST_CASE("steering changes logits and clears cleanly") {
    auto engine = gbx::Engine::from_file(FIX + "/tiny_model.gbx");
    auto ids = engine.tokenizer().encode("Once upon a time");
    auto base = engine.prefill(ids);

    engine.model().set_steering(3, 8.0f);
    auto steered = engine.prefill(ids);
    engine.model().clear_steering();
    auto restored = engine.prefill(ids);

    float diff = 0.f, restore_diff = 0.f;
    for (size_t i = 0; i < base.size(); ++i) {
        diff = std::max(diff, std::fabs(base[i] - steered[i]));
        restore_diff = std::max(restore_diff, std::fabs(base[i] - restored[i]));
    }
    REQUIRE(diff > 1e-4f);           // steering must do something
    REQUIRE(restore_diff <= 1e-6f);  // and be fully reversible
}

TEST_CASE("generation is deterministic per seed") {
    auto engine = gbx::Engine::from_file(FIX + "/tiny_model.gbx");
    gbx::SampleParams p;
    auto a = engine.generate("Once", 20, p, 42);
    auto b = engine.generate("Once", 20, p, 42);
    auto c = engine.generate("Once", 20, p, 43);
    REQUIRE(a == b);
    // different seed will usually differ; do not REQUIRE (tiny model may tie)
    (void)c;
}
