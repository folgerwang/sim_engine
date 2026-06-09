// world.cpp — see world.h.
#include "ecs/world.h"

#include "ecs/asset_streamer.h"
#include "ecs/streaming_system.h"
#include "ecs/transform_system.h"

namespace engine {
namespace ecs {

size_t World::updateTransforms() {
    return TransformSystem::update(reg_);
}

size_t World::updateStreaming(const glm::vec3& focus) {
    if (!streamer_) return 0;
    return StreamingSystem::update(reg_, *streamer_, focus).resident_count;
}

size_t World::collectGarbage() {
    return LifetimeSystem::collect(reg_, cleanup_);
}

}  // namespace ecs
}  // namespace engine
