#pragma once
//
// thread_pool.h — minimal fixed-size CPU thread pool.
//
// Used for embarrassingly-parallel CPU work (e.g., BC7 block encoding
// for the runtime virtual texture).  Not optimized for low-latency
// fine-grained tasks — pick std::async or a lock-free queue if those
// are your needs.  This pool exists so we can saturate cores when we
// have hundreds-of-millis of CPU work to do at scene-load time.
//
// Usage:
//   ThreadPool pool;                 // sized to hardware_concurrency()
//   pool.parallelFor(N, [&](size_t i) { ... });   // blocks until done
//
// parallelFor splits [0, N) into chunks of roughly N/num_threads each,
// dispatches one chunk per thread, and waits for all to finish.  Inner
// functor must be safe to call from any thread.
//

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace engine {
namespace helper {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    // Submit a single task.  Returns immediately; work runs on a
    // worker thread.  No future / handle — caller must synchronise
    // externally if it needs to know when the task finished.
    void enqueue(std::function<void()> task);

    // Run `body(i)` for i in [0, n) split across worker threads.
    // Blocks until all chunks complete.  Safe to call from outside
    // the pool's worker threads only — re-entry from inside a
    // worker would deadlock.
    void parallelFor(
        size_t n,
        const std::function<void(size_t)>& body);

    size_t numThreads() const { return workers_.size(); }

private:
    void workerLoop();

    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex                        mtx_;
    std::condition_variable           cv_;
    std::atomic<bool>                 stop_{false};
};

}  // namespace helper
}  // namespace engine
