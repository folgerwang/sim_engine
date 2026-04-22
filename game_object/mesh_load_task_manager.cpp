//
// mesh_load_task_manager.cpp — worker thread + phase orchestration for the
// async mesh-load pipeline. See the header for the three-phase model.
//
#include "mesh_load_task_manager.h"

#include <algorithm>
#include <chrono>
#include <iostream>

#include "renderer/renderer.h"

namespace engine {
namespace game_object {

namespace {
// Chosen to avoid the 16ms frame budget: if we do find a task whose fence
// signaled but phase3 takes longer than this, log it as a hitch warning.
constexpr double kPhase3HitchMs = 4.0;
}  // namespace

MeshLoadTaskManager::MeshLoadTaskManager(
    const std::shared_ptr<renderer::Device>& device)
    : device_(device) {

    async_enabled_ =
        device_ && device_->hasLoaderQueue() && device_->getLoaderQueue() != nullptr;

    if (async_enabled_) {
        std::cout
            << "[MESHLOAD] async path enabled (worker thread + loader queue)"
            << std::endl;
        worker_ = std::thread([this]() { workerLoop(); });
    } else {
        std::cout
            << "[MESHLOAD] async path disabled; submit() will run inline"
            << std::endl;
    }
}

MeshLoadTaskManager::~MeshLoadTaskManager() {
    // Flag the worker to stop, wake it up, and join. If no worker was
    // started (sync fallback), just return — there's nothing to clean up.
    shutdown_.store(true, std::memory_order_release);
    pending_cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }

    // Any in-flight tasks still have GPU work waiting on a fence. Rather
    // than aborting mid-copy (which risks use-after-free on buffers the
    // mesh holds shared_ptrs to), block until everything drains.
    waitAll();
}

std::shared_ptr<MeshLoadTask> MeshLoadTaskManager::submit(
    const std::string&          filename,
    MeshLoadTask::Phase2Fn      phase2_fn,
    MeshLoadTask::Phase3Fn      phase3_fn) {

    auto task       = std::make_shared<MeshLoadTask>();
    task->filename  = filename;
    task->phase2_fn = std::move(phase2_fn);
    task->phase3_fn = std::move(phase3_fn);

    if (!async_enabled_) {
        // Single-queue hardware path: do everything right here. Matches
        // the historical synchronous behaviour so existing call sites
        // can switch to submit() without worrying about the fallback.
        runPhase2(task);

        // With async disabled we used the transient compute queue which
        // blocks until the fence signals inside submitAndWaitTransientCommandBuffer,
        // so phase3 can run immediately.
        if (task->status.load(std::memory_order_acquire) !=
            MeshLoadStatus::kError &&
            task->phase3_fn) {
            task->phase3_fn();
        }
        task->status.store(
            task->status.load() == MeshLoadStatus::kError
                ? MeshLoadStatus::kError
                : MeshLoadStatus::kFinalized,
            std::memory_order_release);
        return task;
    }

    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_tasks_.push(task);
    }
    pending_cv_.notify_one();

    return task;
}

void MeshLoadTaskManager::poll() {
    if (!async_enabled_) {
        // Sync path finalizes inside submit(); nothing to poll.
        return;
    }

    // Collect tasks whose fence has signaled under the in-flight lock, but
    // drop the lock before calling phase3_fn — phase3 may re-enter the
    // manager (creating sub-uploads, logging, etc.) and we don't want the
    // lock held across user code.
    std::vector<std::shared_ptr<MeshLoadTask>> ready;
    {
        std::lock_guard<std::mutex> lock(in_flight_mutex_);
        auto it = in_flight_tasks_.begin();
        while (it != in_flight_tasks_.end()) {
            auto& task = *it;
            if (task->fence && device_->isFenceSignaled(task->fence)) {
                ready.push_back(task);
                it = in_flight_tasks_.erase(it);
            } else {
                ++it;
            }
        }
    }

    for (auto& task : ready) {
        auto start = std::chrono::high_resolution_clock::now();
        try {
            if (task->phase3_fn) {
                task->phase3_fn();
            }
            task->status.store(MeshLoadStatus::kFinalized,
                std::memory_order_release);
        } catch (const std::exception& e) {
            task->error_message = std::string("phase3 threw: ") + e.what();
            task->status.store(MeshLoadStatus::kError,
                std::memory_order_release);
            std::cerr
                << "[MESHLOAD] phase3 error for '" << task->filename
                << "': " << e.what() << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (ms > kPhase3HitchMs) {
            std::cout
                << "[MESHLOAD] phase3 for '" << task->filename
                << "' took " << ms << " ms (> " << kPhase3HitchMs
                << " ms budget)" << std::endl;
        }
    }
}

size_t MeshLoadTaskManager::inFlightCount() const {
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        n += pending_tasks_.size();
    }
    {
        std::lock_guard<std::mutex> lock(in_flight_mutex_);
        n += in_flight_tasks_.size();
    }
    return n;
}

std::vector<std::string> MeshLoadTaskManager::inFlightFilenames() const {
    std::vector<std::string> out;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        // std::queue doesn't expose iteration; copy.
        auto copy = pending_tasks_;
        while (!copy.empty()) {
            out.push_back(copy.front()->filename);
            copy.pop();
        }
    }
    {
        std::lock_guard<std::mutex> lock(in_flight_mutex_);
        for (const auto& t : in_flight_tasks_) {
            out.push_back(t->filename);
        }
    }
    return out;
}

void MeshLoadTaskManager::waitAll() {
    // Spin-wait on the atomic counts. We can't use a CV here because the
    // worker pushes into in_flight_ *after* popping from pending_, so a
    // simple "pending empty + worker idle" predicate isn't sufficient —
    // we also need in-flight to be drained by poll(). Tiny sleep keeps
    // CPU idle between poll attempts.
    while (true) {
        if (async_enabled_) {
            poll();
        }
        if (inFlightCount() == 0) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void MeshLoadTaskManager::workerLoop() {
    // Register ourselves with the device so the transient command-buffer
    // dispatch in VulkanDevice routes our setup/submit calls to the
    // loader channel instead of the main-thread channel. This is what
    // lets unmodified helper code (Helper::createBuffer etc.) run safely
    // from the worker thread.
    if (device_) {
        device_->registerLoaderThread(std::this_thread::get_id());
    }

    // Single-threaded drain loop. One command buffer per task, one fence
    // per task — simple and safe. If we later see measurable per-task
    // allocator overhead, we can pool command buffers by length bucket.
    while (true) {
        std::shared_ptr<MeshLoadTask> task;
        {
            std::unique_lock<std::mutex> lock(pending_mutex_);
            pending_cv_.wait(lock, [this]() {
                return shutdown_.load(std::memory_order_acquire) ||
                       !pending_tasks_.empty();
            });
            if (shutdown_.load(std::memory_order_acquire) &&
                pending_tasks_.empty()) {
                return;
            }
            task = pending_tasks_.front();
            pending_tasks_.pop();
        }
        runPhase2(task);
    }
}

void MeshLoadTaskManager::runPhase2(
    const std::shared_ptr<MeshLoadTask>& task) {
    task->status.store(MeshLoadStatus::kRunning, std::memory_order_release);

    try {
        if (async_enabled_) {
            // Async path: use the loader command pool + loader queue.
            auto cmd_pool = device_->getLoaderCommandPool();
            if (!cmd_pool) {
                throw std::runtime_error("loader command pool is null");
            }

            auto cmd_bufs = device_->allocateCommandBuffers(cmd_pool, 1, true);
            if (cmd_bufs.empty()) {
                throw std::runtime_error("failed to allocate loader command buffer");
            }
            task->cmd_buf = cmd_bufs[0];

            task->cmd_buf->beginCommandBuffer(
                SET_FLAG_BIT(CommandBufferUsage, ONE_TIME_SUBMIT_BIT));

            std::string err;
            bool ok = false;
            if (task->phase2_fn) {
                ok = task->phase2_fn(device_, task->cmd_buf, err);
            }

            if (!ok) {
                task->error_message =
                    err.empty() ? "phase2_fn returned false" : err;
                task->status.store(MeshLoadStatus::kError,
                    std::memory_order_release);
                // End the buffer so Vulkan validation doesn't complain
                // about a left-open buffer, then let it drop with the task.
                task->cmd_buf->endCommandBuffer();
                std::cerr
                    << "[MESHLOAD] phase2 error for '" << task->filename
                    << "': " << task->error_message << std::endl;
                return;
            }

            task->cmd_buf->endCommandBuffer();

            task->fence = device_->createFence(std::source_location::current());
            device_->getLoaderQueue()->submit({ task->cmd_buf }, task->fence);

            task->status.store(MeshLoadStatus::kGpuSubmitted,
                std::memory_order_release);

            {
                std::lock_guard<std::mutex> lock(in_flight_mutex_);
                in_flight_tasks_.push_back(task);
            }
        } else {
            // Sync fallback: reuse the existing transient command buffer
            // + blocking submit. We don't track a fence; the transient
            // path waits internally.
            auto cmd_buf = device_->setupTransientCommandBuffer();
            task->cmd_buf = cmd_buf;

            std::string err;
            bool ok = false;
            if (task->phase2_fn) {
                ok = task->phase2_fn(device_, cmd_buf, err);
            }

            if (!ok) {
                task->error_message =
                    err.empty() ? "phase2_fn returned false" : err;
                task->status.store(MeshLoadStatus::kError,
                    std::memory_order_release);
                // endCommandBuffer + submitAndWait are driven by the
                // transient API; we have to close it out to keep the
                // shared buffer usable for the next call.
                device_->submitAndWaitTransientCommandBuffer();
                std::cerr
                    << "[MESHLOAD] phase2 error for '" << task->filename
                    << "' (sync path): " << task->error_message << std::endl;
                return;
            }

            device_->submitAndWaitTransientCommandBuffer();
            task->status.store(MeshLoadStatus::kGpuSubmitted,
                std::memory_order_release);
        }
    } catch (const std::exception& e) {
        task->error_message = std::string("phase2 threw: ") + e.what();
        task->status.store(MeshLoadStatus::kError,
            std::memory_order_release);
        std::cerr
            << "[MESHLOAD] phase2 exception for '" << task->filename
            << "': " << e.what() << std::endl;
    }
}

}  // namespace game_object
}  // namespace engine
