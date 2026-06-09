#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// streaming_system.h — distance-based load/unload of streamed entities.
//
// Drives each StreamingComponent through Unloaded → Loading → Resident →
// (back to) Unloaded based on the entity's world-space distance to a focus
// point (usually the camera). Hysteresis (load_radius < unload_radius)
// prevents thrash at the boundary. All actual loading goes through the
// injected IAssetStreamer, so this file has no renderer dependency and is
// driven by a mock in the unit tests.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstddef>

#include <glm/glm.hpp>
#include "ecs/entity.h"

namespace engine {
namespace ecs {

class IAssetStreamer;

struct StreamingStats {
    size_t loads_started   = 0;
    size_t became_resident = 0;
    size_t unloads         = 0;
    size_t resident_count  = 0;
    size_t loading_count   = 0;
};

class StreamingSystem {
public:
    // Tick the streaming state machine for all entities with a
    // StreamingComponent + WorldTransform. `focus` is the world-space point
    // distances are measured from. Returns per-tick statistics.
    static StreamingStats update(entt::registry& reg,
                                 IAssetStreamer& streamer,
                                 const glm::vec3& focus);
};

}  // namespace ecs
}  // namespace engine
