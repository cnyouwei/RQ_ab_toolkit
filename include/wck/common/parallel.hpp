#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace wck {

// Runs fn(index) for index in [0, n_items) on up to `threads` workers pulling
// indices from a shared atomic counter. With one worker the loop runs
// sequentially in ascending index order. The first exception thrown by fn
// stops the pool and is rethrown after all workers have joined.
//
// Determinism contract: per-item results must be written to pre-sized,
// index-addressed storage inside fn, and any reductions (mean/std/...) must
// remain sequential post-join loops in ascending index order. Under that
// contract outputs are bit-identical for any thread count.
template <typename Fn>
void parallel_for_index(std::size_t n_items, std::size_t threads, Fn&& fn) {
    const std::size_t workers = std::max<std::size_t>(1U, std::min(threads, n_items));

    if (workers == 1U) {
        for (std::size_t idx = 0U; idx < n_items; ++idx) {
            fn(idx);
        }
        return;
    }

    std::atomic<std::size_t> next_index{0U};
    std::atomic<bool> stop_requested{false};
    std::mutex error_mutex{};
    std::exception_ptr first_error{};

    std::vector<std::jthread> pool{};
    pool.reserve(workers);
    for (std::size_t worker = 0U; worker < workers; ++worker) {
        pool.emplace_back([&](std::stop_token /*token*/) {
            while (!stop_requested.load(std::memory_order_relaxed)) {
                const std::size_t idx = next_index.fetch_add(1U, std::memory_order_relaxed);
                if (idx >= n_items) {
                    break;
                }
                try {
                    fn(idx);
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (first_error == nullptr) {
                            first_error = std::current_exception();
                        }
                    }
                    stop_requested.store(true, std::memory_order_relaxed);
                    break;
                }
            }
        });
    }

    for (auto& worker : pool) {
        worker.join();
    }
    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }
}

}  // namespace wck
