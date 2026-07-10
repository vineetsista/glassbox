// Tokenizer parity with the Python trainer via committed fixtures.
#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gbx.hpp"
#include "json.hpp"
#include "tokenizer.hpp"

namespace {

std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

gbx::Tokenizer load_fixture_tokenizer() {
    auto file = gbx::GBXFile::from_file(std::string(FIXTURES_DIR) + "/tiny_model.gbx");
    return gbx::Tokenizer(file.tokenizer_merges(), file.tokenizer_vocab_size());
}

}  // namespace

TEST_CASE("encodings match python fixture exactly") {
    auto tok = load_fixture_tokenizer();
    auto spec = gbx::json::parse(read_text(std::string(FIXTURES_DIR) + "/tiny_encodings.json"));
    for (const auto& [text, ids_json] : spec.as_object()) {
        std::vector<int> expect;
        for (const auto& v : ids_json.as_array()) expect.push_back(static_cast<int>(v.as_int()));
        INFO("text: " << text);
        REQUIRE(tok.encode(text) == expect);
    }
}

TEST_CASE("roundtrip on assorted strings") {
    auto tok = load_fixture_tokenizer();
    for (const std::string s : {"hello world", "", "a", "  spaces", "tab\ttab", "line\nline",
                                "punct!?...", "it's 42", "caf\xc3\xa9 \xe4\xb8\xad\xe6\x96\x87"}) {
        REQUIRE(tok.decode(tok.encode(s)) == s);
    }
}

TEST_CASE("eot id is vocab_size - 1 and never encoded") {
    auto tok = load_fixture_tokenizer();
    REQUIRE(tok.eot_id() == tok.vocab_size() - 1);
    auto ids = tok.encode("some <|endoftext|> text");
    for (int id : ids) REQUIRE(id != tok.eot_id());
}
