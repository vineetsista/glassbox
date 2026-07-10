// Native CLI for the glassbox engine.
//
//   engine_cli generate --gbx model.gbx --prompt "Once upon a time"
//       [--steps 120] [--temp 0.8] [--top-k 40] [--seed 1] [--steer f:mult ...]
//   engine_cli parity --gbx tiny_model.gbx --tokens t.u16 --logits l.f32
//   engine_cli tokparity --gbx tiny_model.gbx --encodings tiny_encodings.json
//   engine_cli bench --gbx model.gbx [--steps 200]
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "engine.hpp"
#include "json.hpp"

namespace {

std::string arg_value(int argc, char** argv, const std::string& name, const std::string& dflt) {
    for (int i = 0; i + 1 < argc; ++i)
        if (name == argv[i]) return argv[i + 1];
    return dflt;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(size);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
    return buf;
}

int cmd_parity(int argc, char** argv) {
    auto engine = gbx::Engine::from_file(arg_value(argc, argv, "--gbx", ""));
    auto tok_bytes = read_file(arg_value(argc, argv, "--tokens", ""));
    auto ref_bytes = read_file(arg_value(argc, argv, "--logits", ""));

    const int vocab = engine.model().cfg().vocab_size;
    size_t n_tokens = tok_bytes.size() / 2;
    std::vector<uint16_t> tokens(n_tokens);
    std::memcpy(tokens.data(), tok_bytes.data(), tok_bytes.size());
    const float* ref = reinterpret_cast<const float*>(ref_bytes.data());
    size_t ref_rows = ref_bytes.size() / sizeof(float) / static_cast<size_t>(vocab);
    if (ref_rows != n_tokens) {
        std::cerr << "fixture shape mismatch: " << ref_rows << " rows vs " << n_tokens
                  << " tokens\n";
        return 2;
    }

    // fixture tensors are [batch, seq]; each batch row is an independent sequence
    size_t seq = static_cast<size_t>(engine.model().cfg().ctx_len);
    if (n_tokens % seq != 0) seq = n_tokens;  // single sequence fallback
    float max_diff = 0.f;
    std::vector<float> logits(static_cast<size_t>(vocab));
    for (size_t row = 0; row < n_tokens / seq; ++row) {
        engine.model().reset();
        for (size_t t = 0; t < seq; ++t) {
            engine.model().forward(tokens[row * seq + t], logits.data());
            const float* r = ref + (row * seq + t) * static_cast<size_t>(vocab);
            for (int v = 0; v < vocab; ++v) {
                float d = std::fabs(logits[static_cast<size_t>(v)] - r[v]);
                if (d > max_diff) max_diff = d;
            }
        }
    }
    std::cout << "max_abs_logit_diff " << max_diff << "\n";
    return max_diff <= 1e-3f ? 0 : 1;
}

int cmd_tokparity(int argc, char** argv) {
    auto engine = gbx::Engine::from_file(arg_value(argc, argv, "--gbx", ""));
    auto enc_bytes = read_file(arg_value(argc, argv, "--encodings", ""));
    auto spec = gbx::json::parse(
        std::string(reinterpret_cast<const char*>(enc_bytes.data()), enc_bytes.size()));
    int failures = 0;
    for (const auto& [text, ids_json] : spec.as_object()) {
        std::vector<int> expect;
        for (const auto& v : ids_json.as_array()) expect.push_back(static_cast<int>(v.as_int()));
        auto got = engine.tokenizer().encode(text);
        bool ok = got == expect && engine.tokenizer().decode(got) == text;
        if (!ok) {
            ++failures;
            std::cerr << "MISMATCH on " << text.substr(0, 40) << " (got " << got.size()
                      << " ids, want " << expect.size() << ")\n";
        }
    }
    std::cout << (failures == 0 ? "tokenizer parity OK" : "tokenizer parity FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}

int cmd_generate(int argc, char** argv) {
    auto engine = gbx::Engine::from_file(arg_value(argc, argv, "--gbx", ""));
    gbx::SampleParams params;
    params.temperature = std::stof(arg_value(argc, argv, "--temp", "0.8"));
    params.top_k = std::stoi(arg_value(argc, argv, "--top-k", "40"));
    int steps = std::stoi(arg_value(argc, argv, "--steps", "120"));
    uint64_t seed = std::stoull(arg_value(argc, argv, "--seed", "1"));
    std::string prompt = arg_value(argc, argv, "--prompt", "Once upon a time");

    for (int i = 0; i + 1 < argc; ++i) {
        if (std::string("--steer") == argv[i]) {
            std::string s = argv[i + 1];
            auto colon = s.find(':');
            if (colon != std::string::npos)
                engine.model().set_steering(std::stoi(s.substr(0, colon)),
                                            std::stof(s.substr(colon + 1)));
        }
    }

    std::string out = engine.generate(prompt, steps, params, seed);
    std::cout << out << "\n";
    return 0;
}

int cmd_bench(int argc, char** argv) {
    auto engine = gbx::Engine::from_file(arg_value(argc, argv, "--gbx", ""));
    int steps = std::stoi(arg_value(argc, argv, "--steps", "200"));
    gbx::SampleParams params;
    auto t0 = std::chrono::steady_clock::now();
    engine.generate("Once upon a time", steps, params, 1);
    auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "tokens_per_sec " << static_cast<double>(steps) / dt << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: engine_cli {generate|parity|tokparity|bench} [options]\n";
        return 2;
    }
    std::string cmd = argv[1];
    try {
        if (cmd == "parity") return cmd_parity(argc, argv);
        if (cmd == "tokparity") return cmd_tokparity(argc, argv);
        if (cmd == "generate") return cmd_generate(argc, argv);
        if (cmd == "bench") return cmd_bench(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
    std::cerr << "unknown command " << cmd << "\n";
    return 2;
}
