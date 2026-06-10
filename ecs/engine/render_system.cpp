// render_system.cpp — see render_system.h.
#include "ecs/engine/render_system.h"

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/gtx/matrix_decompose.hpp>

#include <entt/entt.hpp>

#include "ecs/components.h"
#include "ecs/engine/render_components.h"
#include "game_object/drawable_object.h"

namespace engine {
namespace ecs {

std::vector<std::shared_ptr<game_object::DrawableObject>>
RenderSystem::gather(entt::registry& reg) {
    std::vector<std::shared_ptr<game_object::DrawableObject>> out;

    // Renderable + Visible; Active is implied (created with it).
    for (auto [e, r] : reg.view<Renderable, Visible>().each()) {
        if (!r.drawable || !r.drawable->isReady()) continue;

        // Sync world transform from ECS -> drawable (covers parent moves and
        // editor edits; streamed loads already set it on attach).
        // EXCEPTION: controller-driven drawables (the adopted scene player
        // in play mode) own their world placement via PlayerController's
        // root-node writes — pushing the authored scene transform on top
        // would double-transform the rig.  Still rendered, just not synced.
        if (!r.drawable->isControllerDriven()) {
            if (const auto* wt = reg.try_get<WorldTransform>(e)) {
                glm::vec3 t, s, skew;
                glm::quat q;
                glm::vec4 persp;
                if (glm::decompose(wt->matrix, s, q, t, skew, persp)) {
                    r.drawable->setInstanceRootTransform(t, glm::normalize(q), s);
                }
            }
        }
        out.push_back(r.drawable);
    }
    return out;
}

}  // namespace ecs
}  // namespace engine
