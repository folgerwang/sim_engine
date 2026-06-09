// streaming_system.cpp — see streaming_system.h.
#include "ecs/streaming_system.h"

#include <entt/entt.hpp>

#include "ecs/asset_streamer.h"
#include "ecs/components.h"

namespace engine {
namespace ecs {

StreamingStats StreamingSystem::update(entt::registry& reg,
                                       IAssetStreamer& streamer,
                                       const glm::vec3& focus) {
    StreamingStats stats;

    for (auto [e, sc, wt] :
         reg.view<StreamingComponent, WorldTransform>().each()) {
        const glm::vec3 pos = glm::vec3(wt.matrix[3]);
        const float dist    = glm::distance(pos, focus);

        switch (sc.state) {
        case AssetState::kUnloaded:
            if (dist <= sc.load_radius) {
                sc.handle = streamer.beginLoad(e, sc.asset_path, wt.matrix);
                if (sc.handle != kInvalidStream) {
                    sc.state = AssetState::kLoading;
                    ++stats.loads_started;
                }
            }
            break;

        case AssetState::kLoading: {
            const AssetState s = streamer.poll(sc.handle);
            if (s == AssetState::kResident) {
                sc.state = AssetState::kResident;
                ++stats.became_resident;
            } else if (dist > sc.unload_radius) {
                // Moved away before the load finished — cancel.
                streamer.unload(sc.handle);
                sc.handle = kInvalidStream;
                sc.state  = AssetState::kUnloaded;
                ++stats.unloads;
            }
            break;
        }

        case AssetState::kResident:
            if (dist > sc.unload_radius) {
                streamer.unload(sc.handle);
                sc.handle = kInvalidStream;
                sc.state  = AssetState::kUnloaded;
                ++stats.unloads;
            } else {
                // Keep the renderable's world transform in sync.
                streamer.setWorld(sc.handle, wt.matrix);
            }
            break;
        }

        if (sc.state == AssetState::kResident) ++stats.resident_count;
        else if (sc.state == AssetState::kLoading) ++stats.loading_count;
    }

    return stats;
}

}  // namespace ecs
}  // namespace engine
