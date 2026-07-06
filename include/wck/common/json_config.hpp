#pragma once

#include "wck/common/distributions.hpp"
#include "wck/common/mini_json.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace wck::json_config {

// Shared JSON-config helper kit. Every helper takes the dotted config `path`
// of the value being inspected so error messages stay informative, e.g.
// "simulation.sample_time: must be > 0".

[[noreturn]] void fail(const std::string& path, const std::string& message);

const json::Value& require_key(
    const json::Value::Object& object,
    const std::string& key,
    const std::string& path);

const json::Value* optional_key(const json::Value::Object& object, const std::string& key);

const json::Value::Object& require_object(const json::Value& value, const std::string& path);

const json::Value::Array& require_array(const json::Value& value, const std::string& path);

std::string require_string(const json::Value& value, const std::string& path);

bool require_bool(const json::Value& value, const std::string& path);

double require_number(const json::Value& value, const std::string& path);

int double_to_int(double value, const std::string& path);

// Parses a nonnegative integer-valued number representable as uint64 (used
// for "seed" fields).
std::uint64_t require_uint64(const json::Value& value, const std::string& path);

void reject_unknown_keys(
    const json::Value::Object& object,
    const std::vector<std::string>& allowed,
    const std::string& path);

std::string lowercase(std::string text);

DistributionFamily parse_distribution_family(const std::string& family, const std::string& path);

// Parses and validates a {"distribution": {"family": ..., "params": {...}}}
// wrapper object located at `path`.
DistributionSpec parse_distribution_spec(const json::Value::Object& wrapper, const std::string& path);

}  // namespace wck::json_config
