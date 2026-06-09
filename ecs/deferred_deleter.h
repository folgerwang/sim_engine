#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// deferred_deleter.h — GC-safe, frame-delayed resource reclamation.
//
// The GPU runs behind the CPU by up to `frames_in_flight` frames. If we freed a
// Vulkan buffer / image / descriptor set the instant an entity was destroyed,
// the GPU might still be reading it from a command buffer recorded two frames
// ago → use-after-free / device-lost.
//
// DeferredDeleter solves this with a ring of `frames_in_flight + 1` buckets.
// schedule(fn) drops a deletion closure into the CURRENT frame's bucket. Each
// beginFrame() advances the ring and flushes the bucket we're about to reuse —
// which holds closures scheduled exactly `frames_in_flight` frames ago, by
// which point the GPU has provably finished those frames. shutdown() flushes
// everything (call after Device::waitIdle()).
//
// It is renderer-agnostic: it just runs std::function<void()> closures, so the
// timing logic is unit-testable. The engine schedules closures that capture a
// shared_ptr<DrawableObject> + Device and call drawable->destroy(device).
// ─────────────────────────────────────────────────────────────────────────────
#include <cstddef>
#include <functional>
#include <vector>

namespace engine {
namespace ecs {

class DeferredDeleter {
public:
    using Deleter = std::function<void()>;

    // frames_in_flight = how many frames the GPU may lag the CPU (e.g. 2).
    // We keep one extra bucket so a closure scheduled this frame is never
    // flushed before `frames_in_flight` further beginFrame() calls.
    explicit DeferredDeleter(size_t frames_in_flight = 2)
        : buckets_(frames_in_flight + 1) {}

    ~DeferredDeleter() { shutdown(); }

    DeferredDeleter(const DeferredDeleter&)            = delete;
    DeferredDeleter& operator=(const DeferredDeleter&) = delete;

    // Queue a resource-release closure to run after frames_in_flight frames.
    void schedule(Deleter fn) {
        if (fn) buckets_[head_].push_back(std::move(fn));
    }

    // Advance the ring one frame and flush closures that have aged out.
    // Returns how many closures were run (handy for the HUD / tests).
    size_t beginFrame() {
        head_ = (head_ + 1) % buckets_.size();
        return flushBucket(head_);
    }

    // Run every pending closure immediately (post-waitIdle / shutdown).
    void shutdown() {
        for (size_t i = 0; i < buckets_.size(); ++i) flushBucket(i);
    }

    // Total closures still waiting across all buckets (diagnostics).
    size_t pending() const {
        size_t n = 0;
        for (const auto& b : buckets_) n += b.size();
        return n;
    }

    size_t framesInFlight() const { return buckets_.size() - 1; }

private:
    size_t flushBucket(size_t idx) {
        auto& bucket = buckets_[idx];
        const size_t n = bucket.size();
        // Move out first so a closure that schedules more deletions can't
        // mutate the bucket we're iterating.
        std::vector<Deleter> local;
        local.swap(bucket);
        for (auto& fn : local) fn();
        return n;
    }

    std::vector<std::vector<Deleter>> buckets_;
    size_t head_ = 0;
};

}  // namespace ecs
}  // namespace engine
