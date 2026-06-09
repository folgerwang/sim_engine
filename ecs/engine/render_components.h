#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// render_components.h — Vulkan-bearing components (engine side only).
//
// Kept OUT of ecs/components.h so the core ECS stays renderer-free and
// unit-testable. These components hold engine objects (the existing
// DrawableObject, etc.) by shared_ptr, so an entity is the owner and the GC /
// deferred-deleter controls when the underlying GPU resources are freed.
//
// Note we deliberately reuse the existing DrawableObject rather than shredding
// it into Transform/Mesh/Material sub-components on day one. DrawableObject is
// already a self-contained "render package" (nodes, meshes, materials, skins);
// wrapping it in a component gives us the ECS lifetime/streaming/iteration wins
// immediately, and the deeper SoA decomposition can follow incrementally
// (see ECS_DESIGN.md, phase 3).
// ─────────────────────────────────────────────────────────────────────────────
#include <memory>

#include "game_object/drawable_object.h"
#include "game_object/camera_object.h"

namespace engine {
namespace ecs {

// An entity that renders via the existing DrawableObject pipeline.
struct Renderable {
    std::shared_ptr<game_object::DrawableObject> drawable;
};

// An entity that owns a camera (view + projection driver).
struct CameraRef {
    std::shared_ptr<game_object::CameraObject> camera;
};

}  // namespace ecs
}  // namespace engine
