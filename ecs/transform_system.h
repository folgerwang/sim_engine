#pragma once
// transform_system.h — hierarchy propagation + world-matrix/bounds caching.
//
// Recomputes WorldTransform (and WorldBounds, when LocalBounds is present) for
// every entity flagged DirtyTransform, plus all of their descendants (a dirty
// parent dirties its whole subtree). Processing is ordered shallow->deep so a
// child always sees its parent's freshly computed world matrix.
//
// Renderer-agnostic and allocation-light: the only per-pass scratch is a
// parent->children adjacency built from the Parent pool. For very large scenes
// this can be replaced by EnTT relationship storage + a sorted pool (see
// ECS_DESIGN.md, "Scaling the transform system"); the public API stays the same.
#include <cstddef>

#include <glm/glm.hpp>

#include "ecs/entity.h"

namespace engine {
namespace ecs {

class TransformSystem {
public:
    // Recompute world transforms/bounds for dirty subtrees, then clear the
    // DirtyTransform tags. Returns the number of entities updated.
    static size_t update(entt::registry& reg);

    // Compute a single entity's world matrix on demand (walks up to the root).
    static glm::mat4 computeWorld(entt::registry& reg, Entity e);
};

}  // namespace ecs
}  // namespace engine
