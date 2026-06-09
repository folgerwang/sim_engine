#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// render_system.h — bridge from ECS entities to the existing draw path.
//
// The engine's renderer already knows how to draw a list of DrawableObjects
// (ObjectSceneView / application.cpp's drawScene). Rather than rewrite the
// renderer, the render system simply GATHERS the drawables of all entities that
// are Renderable + Visible + Active, syncing each drawable's world transform
// from the entity's WorldTransform first. The returned list is fed straight
// into the existing ObjectSceneView, so the ECS slots in under the renderer
// with no shader/pipeline changes.
//
// Engine side only (touches DrawableObject + Vulkan).
// ─────────────────────────────────────────────────────────────────────────────
#include <memory>
#include <vector>
#include "ecs/entity.h"

namespace engine {
namespace game_object { class DrawableObject; }

namespace ecs {

class RenderSystem {
public:
    // Collect drawables for all visible, ready entities, pushing each entity's
    // WorldTransform into its DrawableObject (so a parent move / animation is
    // reflected). Returns the list to hand to ObjectSceneView.
    static std::vector<std::shared_ptr<game_object::DrawableObject>>
    gather(entt::registry& reg);
};

}  // namespace ecs
}  // namespace engine
