#pragma once

#include <cstdint>
#include <string>

namespace wck {

// FNV-1a and splitmix64 are part of the frozen seed-derivation machinery.
// Same-seed reproducibility of published results depends on these exact
// constants and operations; do not modify.

inline std::uint64_t fnv1a_64(const std::string& text) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const char ch : text) {
        hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(ch));
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline std::uint64_t splitmix64(std::uint64_t x) {
    std::uint64_t z = x + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

}  // namespace wck
