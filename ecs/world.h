#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// world.h — the ECS facade the engine talks to.
//
// World owns the EnTT registry, the deferred deleter (GC), and orchestrates the
// per-frame system order. It is the only ECS type the rest of the engine needs
// to hold. Renderer-agnostic: streaming is driven through an injected
// IAssetStreamer*, and resource cleanup through an injectable hook, so World
// compiles and ticks with zero Vulkan dependency (the unit tests do exactly
// that).
//
// Per-frame contract (call once per rendered frame, in this order):
//
//     world.beginFrame();                 // advance GC ring, flush aged frees
//     world.updateTransforms();           // dirty subtrees → world matrices
//     world.updateStreaming(camera_pos);  // load/unload by distance
//     world.collectGarbage();             // destroy PendingDestroy entities
//     // ... engine records draw calls via the render bridge ...
//
// Entity creation helpers set DirtyTransform so the first frame computes a
// correct world matrix. destroy() is deferred (tags PendingDestroy); the actual
// teardown + generational invalidation happens in collectGarbage().
// ─────────────────────────────────────────────────────────────────────────────
#include <cstddef>

#include <entt/entt.hpp>
#include <glm/glm.hpp>

#include "ecs/components.h"
#include "ecs/deferred_deleter.h"
#include "ecs/lifetime_system.h"

namespace engine {
namespace ecs {

class IAssetStreamer;

class World {
public:
    explicit World(size_t frames_in_flight = 2) : deleter_(frames_in_flight) {}

    // Streaming + cleanup are optional; install them when the renderer is up.
    void setStreamer(IAssetStreamer* s) { streamer_ = s; }
    void setCleanupHook(LifetimeSystem::CleanupHook hook) {
        cleanup_ = std::move(hook);
    }

    entt::registry&       registry()       { return reg_; }
    const entt::registry& registry() const { return reg_; }
    DeferredDeleter&      deleter()         { return deleter_; }

    // ── Entity lifecycle ─────────────────────────────────────────────────--
    Entity create() {
        Entity e = reg_.create();
        reg_.emplace<Active>(e);
        return e;
    }

    // Create an entity with a transform (and mark it dirty for frame 0).
    Entity createAt(const LocalTransform& xform, Entity parent = kNull) {
        Entity e = create();
        reg_.emplace<LocalTransform>(e, xform);
        reg_.emplace<WorldTransform>(e);
        if (parent != kNull) reg_.emplace<Parent>(e, Parent{parent});
        reg_.emplace<DirtyTransform>(e);
        return e;
    }

    bool valid(Entity e) const { return reg_.valid(e); }

    // Mark the entity's transform dirty so the next updateTransforms()
    // recomputes it (and its descendants). Call after editing LocalTransform.
    void markDirty(Entity e) {
        if (reg_.valid(e) && !reg_.all_of<DirtyTransform>(e))
            reg_.emplace<DirtyTransform>(e);
    }

    // Deferred destroy: tag now, tear down in collectGarbage(). Safe to call
    // from inside a system iterating other entities.
    void destroy(Entity e) {
        if (reg_.valid(e) && !reg_.all_of<PendingDestroy>(e))
            reg_.emplace<PendingDestroy>(e);
    }

    // ── Per-frame systems ────────────────────────────────────────────────--
    size_t beginFrame()        { ++frame_; return deleter_.beginFrame(); }
    size_t updateTransforms();
    size_t updateStreaming(const glm::vec3& focus);
    size_t collectGarbage();

    // Flush all pending GC immediately. Call after Device::waitIdle() on
    // shutdown / scene teardown so every GPU resource is released.
    void flushAll() {
        collectGarbage();
        deleter_.shutdown();
    }

    uint64_t frameIndex() const { return frame_; }

private:
    entt::registry             reg_;
    DeferredDeleter            deleter_;
    IAssetStreamer*            streamer_ = nullptr;
    LifetimeSystem::CleanupHook cleanup_;
    uint64_t                   frame_ = 0;
};

}  // namespace ecs
}  // namespace engine
