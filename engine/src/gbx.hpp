// GBX weight-bundle loader. See docs/GBX_FORMAT.md and scripts/export_gbx.py.
// The loader owns the raw byte buffer; tensor views point into it (fp32,
// 64-byte aligned, little-endian — we assert a little-endian host).
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"

namespace gbx {

static_assert(std::endian::native == std::endian::little, "GBX requires a little-endian host");

struct TensorView {
    const float* data = nullptr;
    std::vector<size_t> shape;
    size_t numel = 0;
};

struct ModelConfig {
    int vocab_size, d_model, n_layers, n_heads, d_mlp, ctx_len;
    float rope_base, rms_eps;
    int d_head() const { return d_model / n_heads; }
};

struct SAEConfig {
    int d_in = 0, k = 0, n_features = 0, hook_layer = -1;
    bool present = false;
};

class GBXFile {
public:
    static GBXFile from_file(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("cannot open " + path);
        auto size = static_cast<size_t>(f.tellg());
        f.seekg(0);
        std::vector<uint8_t> buf(size);
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
        return GBXFile(std::move(buf));
    }

    explicit GBXFile(std::vector<uint8_t> buf) : buf_(std::move(buf)) {
        if (buf_.size() < 8 || std::memcmp(buf_.data(), "GBX1", 4) != 0)
            throw std::runtime_error("not a GBX1 file");
        uint32_t hlen;
        std::memcpy(&hlen, buf_.data() + 4, 4);
        if (8 + hlen > buf_.size()) throw std::runtime_error("truncated GBX header");
        std::string header_text(reinterpret_cast<const char*>(buf_.data()) + 8, hlen);
        header_ = json::parse(header_text);

        size_t base = 8 + hlen;
        base += (64 - (base % 64)) % 64;

        const auto& cfg = header_["config"];
        config_ = ModelConfig{
            static_cast<int>(cfg["vocab_size"].as_int()),
            static_cast<int>(cfg["d_model"].as_int()),
            static_cast<int>(cfg["n_layers"].as_int()),
            static_cast<int>(cfg["n_heads"].as_int()),
            static_cast<int>(cfg["d_mlp"].as_int()),
            static_cast<int>(cfg["ctx_len"].as_int()),
            static_cast<float>(cfg["rope_base"].as_double()),
            static_cast<float>(cfg["rms_eps"].as_double()),
        };

        if (header_.has("sae") && !header_["sae"].is_null()) {
            const auto& s = header_["sae"];
            sae_ = SAEConfig{
                static_cast<int>(s["d_in"].as_int()),
                static_cast<int>(s["k"].as_int()),
                static_cast<int>(s["n_features"].as_int()),
                static_cast<int>(s["hook_layer"].as_int()),
                true,
            };
        }

        for (const auto& t : header_["tensors"].as_array()) {
            TensorView v;
            size_t offset = static_cast<size_t>(t["offset"].as_int());
            size_t nbytes = static_cast<size_t>(t["nbytes"].as_int());
            if (base + offset + nbytes > buf_.size())
                throw std::runtime_error("tensor out of bounds: " + t["name"].as_string());
            v.data = reinterpret_cast<const float*>(buf_.data() + base + offset);
            v.numel = nbytes / sizeof(float);
            for (const auto& d : t["shape"].as_array())
                v.shape.push_back(static_cast<size_t>(d.as_int()));
            tensors_.emplace(t["name"].as_string(), std::move(v));
        }
    }

    const TensorView& tensor(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::runtime_error("missing tensor: " + name);
        return it->second;
    }
    bool has_tensor(const std::string& name) const { return tensors_.count(name) > 0; }

    const ModelConfig& config() const { return config_; }
    const SAEConfig& sae_config() const { return sae_; }

    std::vector<std::pair<int, int>> tokenizer_merges() const {
        std::vector<std::pair<int, int>> merges;
        for (const auto& m : header_["tokenizer"]["merges"].as_array()) {
            const auto& pair = m.as_array();
            merges.emplace_back(static_cast<int>(pair[0].as_int()),
                                static_cast<int>(pair[1].as_int()));
        }
        return merges;
    }
    int tokenizer_vocab_size() const {
        return static_cast<int>(header_["tokenizer"]["vocab_size"].as_int());
    }

private:
    std::vector<uint8_t> buf_;
    json::Value header_;
    ModelConfig config_{};
    SAEConfig sae_{};
    std::unordered_map<std::string, TensorView> tensors_;
};

}  // namespace gbx
