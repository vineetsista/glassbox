// Minimal recursive-descent JSON parser. Zero dependencies, C++20.
// Supports exactly what GBX headers need: objects, arrays, strings (with
// \uXXXX escapes), doubles, ints, bools, null. Throws std::runtime_error on
// malformed input. Not a general-purpose library; not optimized.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace gbx::json {

class Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value>;

class Value {
public:
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> v;

    Value() : v(nullptr) {}
    explicit Value(std::nullptr_t) : v(nullptr) {}
    explicit Value(bool b) : v(b) {}
    explicit Value(double d) : v(d) {}
    explicit Value(std::string s) : v(std::move(s)) {}
    explicit Value(Array a) : v(std::move(a)) {}
    explicit Value(Object o) : v(std::move(o)) {}

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(v); }
    bool as_bool() const { return std::get<bool>(v); }
    double as_double() const { return std::get<double>(v); }
    int64_t as_int() const { return static_cast<int64_t>(std::get<double>(v)); }
    const std::string& as_string() const { return std::get<std::string>(v); }
    const Array& as_array() const { return std::get<Array>(v); }
    const Object& as_object() const { return std::get<Object>(v); }

    const Value& operator[](const std::string& key) const {
        const auto& obj = as_object();
        auto it = obj.find(key);
        if (it == obj.end()) throw std::runtime_error("json: missing key " + key);
        return it->second;
    }
    bool has(const std::string& key) const {
        const auto* obj = std::get_if<Object>(&v);
        return obj && obj->count(key) > 0;
    }
};

namespace detail {

struct Parser {
    std::string_view s;
    size_t i = 0;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("json parse error at byte " + std::to_string(i) + ": " + msg);
    }
    void skip_ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }
    char peek() {
        if (i >= s.size()) fail("unexpected end");
        return s[i];
    }
    char next() {
        char c = peek();
        ++i;
        return c;
    }
    void expect(char c) {
        if (next() != c) fail(std::string("expected '") + c + "'");
    }

    Value parse_value() {
        skip_ws();
        char c = peek();
        switch (c) {
            case '{':
                return parse_object();
            case '[':
                return parse_array();
            case '"':
                return Value(parse_string());
            case 't':
                literal("true");
                return Value(true);
            case 'f':
                literal("false");
                return Value(false);
            case 'n':
                literal("null");
                return Value(nullptr);
            default:
                return parse_number();
        }
    }

    void literal(std::string_view lit) {
        if (s.substr(i, lit.size()) != lit) fail("bad literal");
        i += lit.size();
    }

    Value parse_object() {
        expect('{');
        Object obj;
        skip_ws();
        if (peek() == '}') {
            ++i;
            return Value(std::move(obj));
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            obj.emplace(std::move(key), parse_value());
            skip_ws();
            char c = next();
            if (c == '}') break;
            if (c != ',') fail("expected ',' or '}'");
        }
        return Value(std::move(obj));
    }

    Value parse_array() {
        expect('[');
        Array arr;
        skip_ws();
        if (peek() == ']') {
            ++i;
            return Value(std::move(arr));
        }
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            char c = next();
            if (c == ']') break;
            if (c != ',') fail("expected ',' or ']'");
        }
        return Value(std::move(arr));
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            char c = next();
            if (c == '"') break;
            if (c == '\\') {
                char e = next();
                switch (e) {
                    case '"':
                        out += '"';
                        break;
                    case '\\':
                        out += '\\';
                        break;
                    case '/':
                        out += '/';
                        break;
                    case 'b':
                        out += '\b';
                        break;
                    case 'f':
                        out += '\f';
                        break;
                    case 'n':
                        out += '\n';
                        break;
                    case 'r':
                        out += '\r';
                        break;
                    case 't':
                        out += '\t';
                        break;
                    case 'u':
                        out += parse_unicode_escape();
                        break;
                    default:
                        fail("bad escape");
                }
            } else {
                out += c;
            }
        }
        return out;
    }

    std::string parse_unicode_escape() {
        auto hex4 = [&]() -> uint32_t {
            uint32_t cp = 0;
            for (int k = 0; k < 4; ++k) {
                char c = next();
                cp <<= 4;
                if (c >= '0' && c <= '9')
                    cp |= static_cast<uint32_t>(c - '0');
                else if (c >= 'a' && c <= 'f')
                    cp |= static_cast<uint32_t>(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F')
                    cp |= static_cast<uint32_t>(c - 'A' + 10);
                else
                    fail("bad \\u escape");
            }
            return cp;
        };
        uint32_t cp = hex4();
        if (cp >= 0xD800 && cp <= 0xDBFF) {  // surrogate pair
            expect('\\');
            expect('u');
            uint32_t lo = hex4();
            if (lo < 0xDC00 || lo > 0xDFFF) fail("bad surrogate pair");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
        }
        // encode as utf-8
        std::string out;
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return out;
    }

    Value parse_number() {
        size_t start = i;
        if (peek() == '-') ++i;
        while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' ||
                                s[i] == 'E' || s[i] == '+' || s[i] == '-'))
            ++i;
        if (i == start) fail("bad number");
        return Value(std::stod(std::string(s.substr(start, i - start))));
    }
};

}  // namespace detail

inline Value parse(std::string_view text) {
    detail::Parser p{text};
    Value v = p.parse_value();
    p.skip_ws();
    if (p.i != text.size()) p.fail("trailing garbage");
    return v;
}

}  // namespace gbx::json
