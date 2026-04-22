#pragma once
//
// mesh_load_task_manager.h — Layer 2 of the async mesh loader.
//
// A small task queue + worker thread that drives the three-phase async
// mesh load:
//
//   Phase 1 (main thread):     construct task shell, enqueue.
//   Phase 2 (worker thread):   parse file / build CPU buffers /
//                              create GPU buffers+images / record copies
//                              / submit to the loader queue with a fence.
//   Phase 3 (main thread):     after the fence signals, allocate
//                              descriptor sets + create pipelines +
//                              publish the finished object.
//
// The main thread polls via poll() once per frame; tasks whose fence has
// signaled invoke their user-supplied phase3_fn on the polling thread.
//
// If the device does not expose a loader queue (single-queue hardware),
// submit() falls back to running phase2+phase3 synchronously on the
// calling thread. Existing synchronous call sites therefore continue to
// work without modification.
//
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "renderer/renderer.h"

namespace engine {
namespace game_object {

// Lifecycle status of a single async mesh load. Atomically updated so the
// main thread can observe it without locking the manager.
enum class MeshLoadStatus : uint32_t {
    kPending      = 0,   // sitting in the queue
    kRunning      = 1,   // worker is running phase2
    kGpuSubmitted = 2,   // phase2 finished, fence submitted, waiting
    kFinalized    = 3,   // phase3 ran, task complete
    kError        = 4    // phase2 failed; see error_message
};

struct MeshLoadTask {
    // Human-readable label for the HUD spinner / logging.
    std::string filename;

    std::atomic<MeshLoadStatus> status{MeshLoadStatus::kPending};

    // GPU sync: set by the worker before moving the task to "in-flight".
    // The main thread polls fence via Device::isFenceSignaled.
    std::shared_ptr<renderer::Fence>         fence;
    std::shared_ptr<renderer::CommandBuffer> cmd_buf;

    // Populated on error; readable once status == kError.
    std::string error_message;

    // Phase 2 runs on the worker. It receives the device (for buffer /
    // image creation), a recording command buffer pre-begun with the
    // one-time-submit flag, and an error-string out param. Return true on
    // success (the manager will endCommandBuffer + submit with fence).
    using Phase2Fn = std::function<bool(
        const std::shared_ptr<renderer::Device>& /*device*/,
        const std::shared_ptr<renderer::CommandBuffer>& /*cmd_buf*/,
        std::string& /*error_out*/)>;

    // Phase 3 runs on the main thread after the fence has signaled.
    // All descriptor-pool / pipeline work belongs here because those
    // paths are not guaranteed thread-safe by the current engine.
    using Phase3Fn = std::function<void()>;

    Phase2Fn phase2_fn;
    Phase3Fn phase3_fn;
};

// Manages a single worker thread that drains a task queue. One manager
// instance is owned by the application; drawable_object and friends
// submit() to it from the main thread.
class MeshLoadTaskManager {
public:
    // `device` must outlive the manager. If device->hasLoaderQueue() is
    // false, the manager still works but submit() runs synchronously.
    explicit MeshLoadTaskManager(
        const std::shared_ptr<renderer::Device>& device);
    ~MeshLoadTaskManager();

    MeshLoadTaskManager(const MeshLoadTaskManager&)            = delete;
    MeshLoadTaskManager& operator=(const MeshLoadTaskManager&) = delete;

    // Submit a new async load. Returns a shared_ptr so the main thread
    // can observe status / error without holding manager locks. If the
    // async path is disabled, phase2+phase3 run inline before return.
    std::shared_ptr<MeshLoadTask> submit(
        const std::string&          filename,
        MeshLoadTask::Phase2Fn      phase2_fn,
        MeshLoadTask::Phase3Fn      phase3_fn);

    // Main-thread tick. Runs phase3_fn for any in-flight tasks whose
    // fence has signaled. Cheap when nothing is ready.
    void poll();

    // Block until every submitted task has either finalized or errored.
    // Useful at shutdown or for the startup-asset barrier.
    void waitAll();

    // HUD-friendly query: count of tasks not yet finalized.
    size_t inFlightCount() const;

    // Snapshot of filenames still in flight, in submission order. The
    // returned vector is a copy and safe to read across threads.
    std::vector<std::string> inFlightFilenames() const;

    bool hasAsyncPath() const { return async_enabled_; }

private:
    void workerLoop();

    // Runs phase2 on the CURRENT thread (worker normally, calling thread
    // if async is disabled). On success the task is pushed to in_flight_
    // with a submitted fence; on failure status becomes kError.
    void runPhase2(const std::shared_ptr<MeshLoadTask>& task);

    // Shared across construction / destruction.
    std::shared_ptr<renderer::Device> device_;
    bool                              async_enabled_ = false;

    // Pending queue (main -> worker).
    mutable std::mutex                            pending_mutex_;
    std::condition_variable                       pending_cv_;
    std::queue<std::shared_ptr<MeshLoadTask>>     pending_tasks_;
    std::atomic<bool>                             shutdown_{false};

    // Tasks whose phase2 has been submitted to the GPU and are waiting
    // on their fence. Polled and drained by the main thread.
    mutable std::mutex                            in_flight_mutex_;
    std::vector<std::shared_ptr<MeshLoadTask>>    in_flight_tasks_;

    std::thread worker_;
};

}  // namespace game_object
}  // namespace engine
