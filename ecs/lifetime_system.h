#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// lifetime_system.h — deferred, GC-safe entity destruction.
//
// Destruction is two-phase. World::destroy(e) only TAGS the entity with
// PendingDestroy (so a system mid-iteration never has the rug pulled out).
// Once per frame, LifetimeSystem::collect() walks the PendingDestroy set and,
// for each entity:
//   1. lets a cleanup hook release per-component resources — streamed assets
//      are unloaded (which itself schedules a GC-safe GPU free), etc.;
//   2. calls registry.destroy(e), recycling the slot with a bumped version so
//      every stale Entity copy is now invalid (the generational GC guarantee).
//
// The cleanup hook keeps this file renderer-agnostic: the engine installs a
// hook that knows about Renderable/Streaming components; the core just invokes
// it. Tests install a counting hook.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstddef>
#include <functional>

#include "ecs/entity.h"

namespace engine {
namespace ecs {

class LifetimeSystem {
public:
    // Called once per pending entity, just before it is destroyed, to release
    // any resources its components hold (GPU buffers via the deferred deleter,
    // streamed assets, etc.).
    using CleanupHook = std::function<void(entt::registry&, Entity)>;

    // Destroy every entity tagged PendingDestroy. Returns the count destroyed.
    static size_t collect(entt::registry& reg, const CleanupHook& on_destroy);
};

}  // namespace ecs
}  // namespace engine
