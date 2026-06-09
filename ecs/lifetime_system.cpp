// lifetime_system.cpp — see lifetime_system.h.
#include "ecs/lifetime_system.h"

#include <vector>

#include <entt/entt.hpp>

#include "ecs/components.h"

namespace engine {
namespace ecs {

size_t LifetimeSystem::collect(entt::registry& reg,
                               const CleanupHook& on_destroy) {
    // Snapshot first: destroying while iterating the same view is undefined,
    // and a cleanup hook may itself touch components.
    std::vector<Entity> doomed;
    for (auto e : reg.view<PendingDestroy>()) doomed.push_back(e);

    for (Entity e : doomed) {
        if (!reg.valid(e)) continue;          // already gone (double-tag)
        if (on_destroy) on_destroy(reg, e);   // release resources
        reg.destroy(e);                        // bumps version → handles invalid
    }
    return doomed.size();
}

}  // namespace ecs
}  // namespace engine
