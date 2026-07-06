#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace wck::json {

class Value {
public:
    using Object = std::map<std::string, Value>;
    using Array = std::vector<Value>;
    enum class Type {
        kNull,
        kBool,
        kNumber,
        kString,
        kArray,
        kObject,
    };

    Value();
    explicit Value(std::nullptr_t);
    explicit Value(bool value);
    explicit Value(double value);
    explicit Value(std::string value);
    explicit Value(const char* value);
    explicit Value(Array value);
    explicit Value(Object value);

    [[nodiscard]] Type type() const noexcept;
    [[nodiscard]] bool is_null() const noexcept;
    [[nodiscard]] bool is_bool() const noexcept;
    [[nodiscard]] bool is_number() const noexcept;
    [[nodiscard]] bool is_string() const noexcept;
    [[nodiscard]] bool is_array() const noexcept;
    [[nodiscard]] bool is_object() const noexcept;

    [[nodiscard]] bool as_bool() const;
    [[nodiscard]] double as_number() const;
    [[nodiscard]] const std::string& as_string() const;
    [[nodiscard]] const Array& as_array() const;
    [[nodiscard]] const Object& as_object() const;

    [[nodiscard]] const Value* get(const std::string& key) const;

private:
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> data_;
};

[[nodiscard]] std::string type_name(Value::Type type);

class ParseError : public std::runtime_error {
public:
    ParseError(std::string message, std::size_t offset);
    std::size_t offset = 0U;
};

[[nodiscard]] Value parse(std::string_view text);
[[nodiscard]] Value parse_file(const std::filesystem::path& path);

}  // namespace wck::json
