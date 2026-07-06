#include "wck/common/mini_json.hpp"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace wck::json {

namespace {

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Value parse_root() {
        skip_whitespace();
        Value v = parse_value();
        skip_whitespace();
        if (!eof()) {
            throw error("trailing characters after JSON value");
        }
        return v;
    }

private:
    [[nodiscard]] bool eof() const {
        return pos_ >= text_.size();
    }

    [[nodiscard]] char peek() const {
        if (eof()) {
            return '\0';
        }
        return text_[pos_];
    }

    char take() {
        if (eof()) {
            throw error("unexpected end of input");
        }
        return text_[pos_++];
    }

    void skip_whitespace() {
        while (!eof()) {
            const unsigned char ch = static_cast<unsigned char>(text_[pos_]);
            if (!std::isspace(ch)) {
                return;
            }
            ++pos_;
        }
    }

    [[nodiscard]] ParseError error(const std::string& message) const {
        return ParseError(message, pos_);
    }

    [[nodiscard]] Value parse_value() {
        if (eof()) {
            throw error("expected JSON value");
        }

        const char ch = peek();
        if (ch == '{') {
            return parse_object();
        }
        if (ch == '[') {
            return parse_array();
        }
        if (ch == '"') {
            return Value(parse_string());
        }
        if (ch == 't') {
            parse_literal("true");
            return Value(true);
        }
        if (ch == 'f') {
            parse_literal("false");
            return Value(false);
        }
        if (ch == 'n') {
            parse_literal("null");
            return Value(nullptr);
        }
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
            return Value(parse_number());
        }

        throw error("invalid JSON value");
    }

    void parse_literal(std::string_view literal) {
        for (char expected : literal) {
            if (take() != expected) {
                throw error("invalid literal token");
            }
        }
    }

    static void append_utf8(std::string& out, std::uint32_t code_point) {
        if (code_point <= 0x7FU) {
            out.push_back(static_cast<char>(code_point));
            return;
        }
        if (code_point <= 0x7FFU) {
            out.push_back(static_cast<char>(0xC0U | ((code_point >> 6U) & 0x1FU)));
            out.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
            return;
        }
        if (code_point <= 0xFFFFU) {
            out.push_back(static_cast<char>(0xE0U | ((code_point >> 12U) & 0x0FU)));
            out.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
            out.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
            return;
        }
        out.push_back(static_cast<char>(0xF0U | ((code_point >> 18U) & 0x07U)));
        out.push_back(static_cast<char>(0x80U | ((code_point >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
    }

    [[nodiscard]] std::uint32_t parse_hex4() {
        std::uint32_t value = 0U;
        for (int i = 0; i < 4; ++i) {
            if (eof()) {
                throw error("incomplete unicode escape");
            }
            const char ch = take();
            value <<= 4U;
            if (ch >= '0' && ch <= '9') {
                value += static_cast<std::uint32_t>(ch - '0');
            } else if (ch >= 'a' && ch <= 'f') {
                value += 10U + static_cast<std::uint32_t>(ch - 'a');
            } else if (ch >= 'A' && ch <= 'F') {
                value += 10U + static_cast<std::uint32_t>(ch - 'A');
            } else {
                throw error("invalid unicode escape sequence");
            }
        }
        return value;
    }

    [[nodiscard]] std::string parse_string() {
        if (take() != '"') {
            throw error("expected opening quote");
        }

        std::string out;
        while (!eof()) {
            const char ch = take();
            if (ch == '"') {
                return out;
            }
            if (ch == '\\') {
                if (eof()) {
                    throw error("incomplete escape sequence");
                }
                const char esc = take();
                switch (esc) {
                case '"':
                case '\\':
                case '/':
                    out.push_back(esc);
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u': {
                    const std::uint32_t cp = parse_hex4();
                    append_utf8(out, cp);
                    break;
                }
                default:
                    throw error("unsupported escape sequence");
                }
                continue;
            }

            if (static_cast<unsigned char>(ch) < 0x20U) {
                throw error("control character in JSON string");
            }
            out.push_back(ch);
        }

        throw error("unterminated JSON string");
    }

    [[nodiscard]] double parse_number() {
        const std::size_t begin = pos_;

        if (peek() == '-') {
            ++pos_;
        }

        if (eof()) {
            throw error("incomplete number");
        }

        if (peek() == '0') {
            ++pos_;
        } else {
            if (!std::isdigit(static_cast<unsigned char>(peek()))) {
                throw error("invalid number token");
            }
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
            }
        }

        if (!eof() && peek() == '.') {
            ++pos_;
            if (eof() || !std::isdigit(static_cast<unsigned char>(peek()))) {
                throw error("invalid fractional part in number");
            }
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
            }
        }

        if (!eof() && (peek() == 'e' || peek() == 'E')) {
            ++pos_;
            if (!eof() && (peek() == '+' || peek() == '-')) {
                ++pos_;
            }
            if (eof() || !std::isdigit(static_cast<unsigned char>(peek()))) {
                throw error("invalid exponent in number");
            }
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
            }
        }

        const std::string token(text_.substr(begin, pos_ - begin));
        char* end_ptr = nullptr;
        errno = 0;
        const double value = std::strtod(token.c_str(), &end_ptr);
        if (end_ptr == nullptr || *end_ptr != '\0') {
            throw error("failed to parse number");
        }
        if (errno == ERANGE && !std::isfinite(value)) {
            throw error("number out of range");
        }
        return value;
    }

    [[nodiscard]] Value parse_array() {
        if (take() != '[') {
            throw error("expected '['");
        }

        Value::Array values;
        skip_whitespace();
        if (!eof() && peek() == ']') {
            take();
            return Value(std::move(values));
        }

        while (true) {
            skip_whitespace();
            values.push_back(parse_value());
            skip_whitespace();
            if (eof()) {
                throw error("unterminated array");
            }
            const char ch = take();
            if (ch == ']') {
                break;
            }
            if (ch != ',') {
                throw error("expected ',' or ']' in array");
            }
        }

        return Value(std::move(values));
    }

    [[nodiscard]] Value parse_object() {
        if (take() != '{') {
            throw error("expected '{'");
        }

        Value::Object obj;
        skip_whitespace();
        if (!eof() && peek() == '}') {
            take();
            return Value(std::move(obj));
        }

        while (true) {
            skip_whitespace();
            if (eof() || peek() != '"') {
                throw error("expected string key in object");
            }
            const std::string key = parse_string();
            skip_whitespace();
            if (eof() || take() != ':') {
                throw error("expected ':' after object key");
            }
            skip_whitespace();
            auto [it, inserted] = obj.emplace(key, parse_value());
            if (!inserted) {
                throw error("duplicate object key: " + key);
            }
            skip_whitespace();
            if (eof()) {
                throw error("unterminated object");
            }
            const char ch = take();
            if (ch == '}') {
                break;
            }
            if (ch != ',') {
                throw error("expected ',' or '}' in object");
            }
        }

        return Value(std::move(obj));
    }

    std::string_view text_;
    std::size_t pos_ = 0U;
};

}  // namespace

Value::Value() : data_(nullptr) {}

Value::Value(std::nullptr_t) : data_(nullptr) {}

Value::Value(bool value) : data_(value) {}

Value::Value(double value) : data_(value) {}

Value::Value(std::string value) : data_(std::move(value)) {}

Value::Value(const char* value) : data_(std::string(value)) {}

Value::Value(Array value) : data_(std::move(value)) {}

Value::Value(Object value) : data_(std::move(value)) {}

Value::Type Value::type() const noexcept {
    switch (data_.index()) {
    case 0:
        return Type::kNull;
    case 1:
        return Type::kBool;
    case 2:
        return Type::kNumber;
    case 3:
        return Type::kString;
    case 4:
        return Type::kArray;
    case 5:
        return Type::kObject;
    default:
        return Type::kNull;
    }
}

bool Value::is_null() const noexcept {
    return std::holds_alternative<std::nullptr_t>(data_);
}

bool Value::is_bool() const noexcept {
    return std::holds_alternative<bool>(data_);
}

bool Value::is_number() const noexcept {
    return std::holds_alternative<double>(data_);
}

bool Value::is_string() const noexcept {
    return std::holds_alternative<std::string>(data_);
}

bool Value::is_array() const noexcept {
    return std::holds_alternative<Array>(data_);
}

bool Value::is_object() const noexcept {
    return std::holds_alternative<Object>(data_);
}

bool Value::as_bool() const {
    if (!is_bool()) {
        throw std::invalid_argument("JSON value is not a bool");
    }
    return std::get<bool>(data_);
}

double Value::as_number() const {
    if (!is_number()) {
        throw std::invalid_argument("JSON value is not a number");
    }
    return std::get<double>(data_);
}

const std::string& Value::as_string() const {
    if (!is_string()) {
        throw std::invalid_argument("JSON value is not a string");
    }
    return std::get<std::string>(data_);
}

const Value::Array& Value::as_array() const {
    if (!is_array()) {
        throw std::invalid_argument("JSON value is not an array");
    }
    return std::get<Array>(data_);
}

const Value::Object& Value::as_object() const {
    if (!is_object()) {
        throw std::invalid_argument("JSON value is not an object");
    }
    return std::get<Object>(data_);
}

const Value* Value::get(const std::string& key) const {
    if (!is_object()) {
        throw std::invalid_argument("JSON value is not an object");
    }
    const auto& object = std::get<Object>(data_);
    const auto it = object.find(key);
    if (it == object.end()) {
        return nullptr;
    }
    return &it->second;
}

std::string type_name(Value::Type type) {
    switch (type) {
    case Value::Type::kNull:
        return "null";
    case Value::Type::kBool:
        return "bool";
    case Value::Type::kNumber:
        return "number";
    case Value::Type::kString:
        return "string";
    case Value::Type::kArray:
        return "array";
    case Value::Type::kObject:
        return "object";
    default:
        return "unknown";
    }
}

ParseError::ParseError(std::string message, std::size_t offset_in)
    : std::runtime_error(std::move(message)), offset(offset_in) {}

Value parse(std::string_view text) {
    Parser parser(text);
    return parser.parse_root();
}

Value parse_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open JSON file: " + path.string());
    }

    std::ostringstream oss;
    oss << in.rdbuf();
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("failed while reading JSON file: " + path.string());
    }

    try {
        return parse(oss.str());
    } catch (const ParseError& ex) {
        std::ostringstream msg;
        msg << "JSON parse error at offset " << ex.offset << " in " << path << ": "
            << ex.what();
        throw std::runtime_error(msg.str());
    }
}

}  // namespace wck::json
