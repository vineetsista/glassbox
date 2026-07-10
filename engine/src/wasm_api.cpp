// C API exported to JavaScript (Emscripten). One global engine instance —
// the web app loads one model bundle. All strings are utf-8; bulk data moves
// through caller-allocated buffers on the WASM heap.
//
// Flow: gbx_load(bytes) once; then per generation: gbx_reset(); gbx_feed(id)
// for each prompt token; loop { gbx_sample() -> id; gbx_feed(id) }. After any
// gbx_feed, gbx_feature_count/gbx_feature_id/gbx_feature_val expose the SAE
// features of the hook layer at that token.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "engine.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define GBX_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define GBX_EXPORT
#endif

namespace {

std::unique_ptr<gbx::Engine> g_engine;
std::vector<int> g_ids;  // full token sequence (prompt + generated)
std::vector<float> g_logits;
std::vector<int> g_tokenize_buf;
std::string g_text_buf;
std::unique_ptr<gbx::Rng> g_rng;

}  // namespace

extern "C" {

// ---- lifecycle ----------------------------------------------------------

GBX_EXPORT int gbx_load(const uint8_t* data, int len) {
    try {
        std::vector<uint8_t> buf(data, data + len);
        g_engine = std::make_unique<gbx::Engine>(gbx::GBXFile(std::move(buf)));
        g_logits.assign(static_cast<size_t>(g_engine->model().cfg().vocab_size), 0.f);
        g_rng = std::make_unique<gbx::Rng>(1);
        g_ids.clear();
        return 0;
    } catch (const std::exception&) {
        g_engine.reset();
        return -1;
    }
}

GBX_EXPORT int gbx_vocab_size() {
    return g_engine ? g_engine->model().cfg().vocab_size : -1;
}
GBX_EXPORT int gbx_ctx_len() {
    return g_engine ? g_engine->model().cfg().ctx_len : -1;
}
GBX_EXPORT int gbx_has_sae() {
    return g_engine && g_engine->model().has_sae() ? 1 : 0;
}
GBX_EXPORT int gbx_n_features() {
    return g_engine && g_engine->model().has_sae() ? g_engine->model().sae()->cfg().n_features : 0;
}

// ---- tokenizer ----------------------------------------------------------

GBX_EXPORT int gbx_tokenize(const char* text) {
    if (!g_engine) return -1;
    g_tokenize_buf = g_engine->tokenizer().encode(text);
    return static_cast<int>(g_tokenize_buf.size());
}
GBX_EXPORT int gbx_token_at(int i) {
    if (i < 0 || static_cast<size_t>(i) >= g_tokenize_buf.size()) return -1;
    return g_tokenize_buf[static_cast<size_t>(i)];
}
// decoded text of the current sequence (valid until next call)
GBX_EXPORT const char* gbx_text() {
    if (!g_engine) return "";
    g_text_buf = g_engine->tokenizer().decode(g_ids);
    return g_text_buf.c_str();
}
// display string for one token id (may be a partial utf-8 sequence)
GBX_EXPORT const char* gbx_token_str(int id) {
    if (!g_engine || id < 0 || id >= g_engine->tokenizer().vocab_size()) return "";
    g_text_buf = g_engine->tokenizer().token_bytes(id);
    return g_text_buf.c_str();
}

// ---- generation ---------------------------------------------------------

GBX_EXPORT void gbx_seed(uint32_t hi, uint32_t lo) {
    g_rng = std::make_unique<gbx::Rng>((static_cast<uint64_t>(hi) << 32) | lo);
}

GBX_EXPORT void gbx_reset() {
    if (!g_engine) return;
    g_engine->model().reset();
    g_ids.clear();
}

// feed one token; returns new position count, or -1 when context is full
GBX_EXPORT int gbx_feed(int token) {
    if (!g_engine) return -1;
    if (!g_engine->model().forward(token, g_logits.data())) return -1;
    g_ids.push_back(token);
    return g_engine->model().pos();
}

// sample the next token from the last logits (does NOT feed it)
GBX_EXPORT int gbx_sample(float temperature, int top_k) {
    if (!g_engine || !g_rng) return -1;
    gbx::SampleParams p;
    p.temperature = temperature;
    p.top_k = top_k;
    return gbx::sample(g_logits.data(), g_engine->model().cfg().vocab_size, p, *g_rng);
}

GBX_EXPORT int gbx_eot_id() {
    return g_engine ? g_engine->tokenizer().eot_id() : -1;
}
GBX_EXPORT int gbx_seq_len() {
    return static_cast<int>(g_ids.size());
}
GBX_EXPORT int gbx_seq_at(int i) {
    return (i >= 0 && static_cast<size_t>(i) < g_ids.size()) ? g_ids[static_cast<size_t>(i)] : -1;
}

// top-k logits of the last fed token: fills ids/vals (caller buffers)
GBX_EXPORT int gbx_top_logits(int* ids, float* vals, int n) {
    if (!g_engine) return 0;
    int vocab = g_engine->model().cfg().vocab_size;
    n = n < vocab ? n : vocab;
    std::vector<int> order(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) order[static_cast<size_t>(i)] = i;
    std::partial_sort(order.begin(), order.begin() + n, order.end(), [&](int a, int b) {
        return g_logits[static_cast<size_t>(a)] > g_logits[static_cast<size_t>(b)];
    });
    for (int i = 0; i < n; ++i) {
        ids[i] = order[static_cast<size_t>(i)];
        vals[i] = g_logits[static_cast<size_t>(order[static_cast<size_t>(i)])];
    }
    return n;
}

// ---- SAE features + steering --------------------------------------------

GBX_EXPORT int gbx_feature_count() {
    return g_engine ? static_cast<int>(g_engine->model().last_features().size()) : 0;
}
GBX_EXPORT int gbx_feature_id(int i) {
    const auto& f = g_engine->model().last_features();
    return (i >= 0 && static_cast<size_t>(i) < f.size()) ? f[static_cast<size_t>(i)].feature : -1;
}
GBX_EXPORT float gbx_feature_val(int i) {
    const auto& f = g_engine->model().last_features();
    return (i >= 0 && static_cast<size_t>(i) < f.size()) ? f[static_cast<size_t>(i)].value : 0.f;
}

GBX_EXPORT void gbx_set_steering(int feature, float multiplier) {
    if (g_engine) g_engine->model().set_steering(feature, multiplier);
}
GBX_EXPORT void gbx_clear_steering() {
    if (g_engine) g_engine->model().clear_steering();
}

}  // extern "C"
