#pragma once

// Shared helpers for the C++ tests.

#include "wck/common/distributions.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

inline void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void expect_close(double lhs, double rhs, double tol, const std::string& message) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs) || std::abs(lhs - rhs) > tol) {
        throw std::runtime_error(message + " (lhs=" + std::to_string(lhs) + ", rhs=" + std::to_string(rhs) + ")");
    }
}

template <typename Fn>
void expect_throw(Fn&& fn, const std::string& message) {
    bool threw = false;
    try {
        fn();
    } catch (...) {
        threw = true;
    }
    if (!threw) {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void expect_throw_contains(Fn&& fn, const std::string& needle, const std::string& message) {
    try {
        fn();
    } catch (const std::exception& ex) {
        const std::string text(ex.what());
        if (text.find(needle) != std::string::npos) {
            return;
        }
        throw std::runtime_error(
            message + " (expected error containing '" + needle + "', got '" + text + "')");
    } catch (...) {
        throw std::runtime_error(message + " (unexpected non-std exception)");
    }
    throw std::runtime_error(message + " (did not throw)");
}

class TempDir {
public:
    explicit TempDir(const std::string& tag) {
        std::mt19937_64 rng(static_cast<std::uint64_t>(std::hash<std::string>{}(tag)));
        path_ = std::filesystem::temp_directory_path() / (tag + "_" + std::to_string(rng()));

        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
        if (ec) {
            throw std::runtime_error("failed to clear temp dir: " + path_.string());
        }
        std::filesystem::create_directories(path_, ec);
        if (ec) {
            throw std::runtime_error("failed to create temp dir: " + path_.string());
        }
    }

    ~TempDir() {
        if (!path_.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(path_, ec);
        }
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    TempDir(TempDir&& other) noexcept : path_(std::move(other.path_)) {
        other.path_.clear();
    }

    TempDir& operator=(TempDir&&) = delete;

    operator const std::filesystem::path&() const noexcept {
        return path_;
    }

    std::string string() const {
        return path_.string();
    }

    friend std::filesystem::path operator/(const TempDir& dir, const std::filesystem::path& child) {
        return dir.path_ / child;
    }

private:
    std::filesystem::path path_{};
};

inline TempDir make_temp_dir(const std::string& tag) {
    return TempDir(tag);
}

inline std::filesystem::path write_text_file(
    const std::filesystem::path& dir,
    const std::string& filename,
    const std::string& content) {
    const std::filesystem::path path = dir / filename;
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open file for write: " + path.string());
    }
    out << content;
    if (!out.good()) {
        throw std::runtime_error("failed to write file: " + path.string());
    }
    return path;
}

inline std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open file: " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline wck::DistributionSpec make_exp(double rate) {
    wck::DistributionSpec d{};
    d.family = wck::DistributionFamily::kExponential;
    d.exponential.rate = rate;
    return d;
}

inline wck::DistributionSpec make_erlang(int k, double rate) {
    wck::DistributionSpec d{};
    d.family = wck::DistributionFamily::kErlangK;
    d.erlang_k.k = k;
    d.erlang_k.rate = rate;
    return d;
}

inline wck::DistributionSpec make_lognormal(double mean, double scv) {
    wck::DistributionSpec d{};
    d.family = wck::DistributionFamily::kLognormal;
    d.lognormal.mean = mean;
    d.lognormal.scv = scv;
    return d;
}

inline wck::DistributionSpec make_h2(double p, double rate1, double rate2) {
    wck::DistributionSpec d{};
    d.family = wck::DistributionFamily::kHyperexponential2;
    d.hyperexponential2.p = p;
    d.hyperexponential2.rate1 = rate1;
    d.hyperexponential2.rate2 = rate2;
    return d;
}
