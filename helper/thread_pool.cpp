#include "thread_pool.h"

#include <atomic>

namespace engine {
namespace helper {

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;  // very conservative fallback
    }
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this]{ workerLoop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
}

void ThreadPool::workerLoop() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this]{ return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

void ThreadPool::parallelFor(
    size_t n,
    const std::function<void(size_t)>& body) {

    if (n == 0) return;

    const size_t num_threads = workers_.size();
    const size_t num_chunks = std::min(n, num_threads);
    if (num_chunks == 0) {
        // Fallback: run on calling thread.
        for (size_t i = 0; i < n; ++i) body(i);
        return;
    }

    // Split [0, n) into num_chunks roughly-equal sub-ranges.  Each
    // chunk is enqueued as a single task; the task loops over its
    // sub-range internally (avoiding per-iteration enqueue overhead).
    std::atomic<size_t> remaining{num_chunks};
    std::mutex          done_mtx;
    std::condition_variable done_cv;

    const size_t base_chunk = n / num_chunks;
    const size_t extra      = n % num_chunks;

    size_t cursor = 0;
    for (size_t c = 0; c < num_chunks; ++c) {
        const size_t chunk_size = base_chunk + (c < extra ? 1 : 0);
        const size_t first      = cursor;
        const size_t last       = cursor + chunk_size;
        cursor = last;

        enqueue([&body, &remaining, &done_mtx, &done_cv, first, last]() {
            for (size_t i = first; i < last; ++i) {
                body(i);
            }
            if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(done_mtx);
                done_cv.notify_one();
            }
        });
    }

    // Wait for all chunks to signal completion.
    std::unique_lock<std::mutex> lk(done_mtx);
    done_cv.wait(lk, [&remaining]{
        return remaining.load(std::memory_order_acquire) == 0;
    });
}

}  // namespace helper
}  // namespace engine
