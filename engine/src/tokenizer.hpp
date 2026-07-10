// Byte-level BPE runtime matching train/bpe.py.
//
// The pre-tokenizer is a hand-written scanner replicating the Python regex
//     '(?:[sdmt]|ll|ve|re) | ?[A-Za-z]+ | ?[0-9]+ | ?[^\sA-Za-z0-9]+
//     | \s+(?!\S) | \s+
// on ASCII input. Known divergence (documented in docs/METHODS.md): Python's
// \s matches Unicode whitespace (e.g. U+00A0); this scanner only treats ASCII
// whitespace as \s, so exotic whitespace lands in the punctuation class.
// Round-trip stays exact either way; only token boundaries could differ.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gbx {

class Tokenizer {
public:
    // merges[i] = {a, b} producing token id 256 + i
    Tokenizer(std::vector<std::pair<int, int>> merges, int vocab_size)
        : vocab_size_(vocab_size), merges_(std::move(merges)) {
        token_bytes_.reserve(static_cast<size_t>(vocab_size_));
        for (int b = 0; b < 256; ++b) token_bytes_.push_back(std::string(1, static_cast<char>(b)));
        for (size_t r = 0; r < merges_.size(); ++r) {
            const auto& [a, b] = merges_[r];
            token_bytes_.push_back(token_bytes_[static_cast<size_t>(a)] +
                                   token_bytes_[static_cast<size_t>(b)]);
            ranks_[key(a, b)] = static_cast<int>(r);
        }
        token_bytes_.push_back("<|endoftext|>");  // display only
    }

    int eot_id() const { return vocab_size_ - 1; }
    int vocab_size() const { return vocab_size_; }

    std::vector<int> encode(std::string_view text) const {
        std::vector<int> out;
        size_t i = 0;
        while (i < text.size()) {
            size_t j = next_chunk_end(text, i);
            encode_chunk(text.substr(i, j - i), out);
            i = j;
        }
        return out;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        for (int id : ids) {
            if (id == eot_id() || id < 0 || id >= vocab_size_) continue;
            out += token_bytes_[static_cast<size_t>(id)];
        }
        return out;
    }

    const std::string& token_bytes(int id) const { return token_bytes_[static_cast<size_t>(id)]; }

private:
    static uint32_t key(int a, int b) {
        return (static_cast<uint32_t>(a) << 16) | static_cast<uint32_t>(b);
    }
    static bool is_ws(char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    }
    static bool is_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
    static bool is_digit(char c) { return c >= '0' && c <= '9'; }
    static bool is_punct(char c) { return !is_ws(c) && !is_alpha(c) && !is_digit(c); }

    // returns the end (exclusive) of the pre-token starting at i
    static size_t next_chunk_end(std::string_view s, size_t i) {
        const size_t n = s.size();
        char c = s[i];
        // contraction: '(?:[sdmt]|ll|ve|re)
        if (c == '\'' && i + 1 < n) {
            char d = s[i + 1];
            if (d == 's' || d == 'd' || d == 'm' || d == 't') return i + 2;
            if (i + 2 < n) {
                char e = s[i + 2];
                if ((d == 'l' && e == 'l') || (d == 'v' && e == 'e') || (d == 'r' && e == 'e'))
                    return i + 3;
            }
        }
        // ' ?[A-Za-z]+ | ' ?[0-9]+ | ' ?[^\sA-Za-z0-9]+
        size_t j = i;
        if (c == ' ' && j + 1 < n) {
            char d = s[j + 1];
            if (is_alpha(d)) {
                j += 1;
                while (j < n && is_alpha(s[j])) ++j;
                return j;
            }
            if (is_digit(d)) {
                j += 1;
                while (j < n && is_digit(s[j])) ++j;
                return j;
            }
            if (is_punct(d)) {
                j += 1;
                while (j < n && is_punct(s[j])) ++j;
                return j;
            }
        }
        if (is_alpha(c)) {
            while (j < n && is_alpha(s[j])) ++j;
            return j;
        }
        if (is_digit(c)) {
            while (j < n && is_digit(s[j])) ++j;
            return j;
        }
        if (is_punct(c)) {
            while (j < n && is_punct(s[j])) ++j;
            return j;
        }
        // whitespace run: \s+(?!\S) | \s+ with the GPT-2 backtracking effect —
        // if the run is followed by a non-space, the last space attaches to it
        size_t end = i;
        while (end < n && is_ws(s[end])) ++end;
        if (end < n && end - i >= 2) return end - 1;  // leave one space for the next chunk
        if (end < n && end - i == 1 && s[i] == ' ') {
            // single space followed by non-space was handled above only for
            // space+alnum/punct; other single-space cases (e.g. "\t") fall here
            return end;
        }
        return end;
    }

    void encode_chunk(std::string_view chunk, std::vector<int>& out) const {
        auto it = cache_.find(std::string(chunk));
        if (it != cache_.end()) {
            out.insert(out.end(), it->second.begin(), it->second.end());
            return;
        }
        std::vector<int> ids;
        ids.reserve(chunk.size());
        for (unsigned char b : chunk) ids.push_back(static_cast<int>(b));
        while (ids.size() >= 2) {
            int best_rank = -1;
            size_t best_i = 0;
            for (size_t k = 0; k + 1 < ids.size(); ++k) {
                auto r = ranks_.find(key(ids[k], ids[k + 1]));
                if (r != ranks_.end() && (best_rank < 0 || r->second < best_rank)) {
                    best_rank = r->second;
                    best_i = k;
                }
            }
            if (best_rank < 0) break;
            ids[best_i] = 256 + best_rank;
            ids.erase(ids.begin() + static_cast<std::ptrdiff_t>(best_i) + 1);
        }
        if (cache_.size() < 500000) cache_.emplace(std::string(chunk), ids);
        out.insert(out.end(), ids.begin(), ids.end());
    }

    int vocab_size_;
    std::vector<std::pair<int, int>> merges_;
    std::vector<std::string> token_bytes_;
    std::unordered_map<uint32_t, int> ranks_;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

}  // namespace gbx
