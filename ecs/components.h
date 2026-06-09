#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// components.h — core, renderer-agnostic components for the ECS.
//
// These are plain-old-data structs stored in EnTT's per-type dense pools, so
// iterating one component type touches contiguous memory (the "data-oriented"
// win). Anything that needs Vulkan (the actual DrawableObject, descriptor
// sets, etc.) lives in ecs/engine/render_components.h instead, so this header
// stays buildable and testable without the renderer.
//
// Design notes:
//   • Transform is split into LocalTransform (authored TRS, the source of
//     truth) and WorldTransform (cached world matrix recomputed by the
//     transform system). This is the standard SoA split: systems that only
//     read world matrices never touch the TRS pool and vice-versa.
//   • Hierarchy is expressed with a Parent component (an Entity handle), which
//     mirrors the existing scene::Object::parent_index model but in entity
//     space. Children are derived each pass by the transform system; we don't
//     store a Children vector to avoid the bookkeeping of keeping it in sync.
//   • Tags (Active / Visible / Static / DirtyTransform / PendingDestroy) are
//     empty structs — EnTT stores them as a membership set with no payload.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstdint>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "ecs/entity.h"

namespace engine {
namespace ecs {

// ── Identity ────────────────────────────────────────────────────────────────
struct Name {
    std::string value;
};

// Stable, serialization-friendly id (distinct from the volatile Entity handle).
// Lets a saved scene re-establish parent/child links after a reload, when
// EnTT will hand out different raw handles.
struct PersistentId {
    uint64_t value = 0;
};

// ── Transform ─────────────────────────────────────────────────────────────--
// Authored local transform (relative to Parent, or world when no Parent).
// Matches engine::scene::Transform's TRS convention (T * R * S).
struct LocalTransform {
    glm::vec3 translation = glm::vec3(0.0f);
    glm::quat rotation    = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);  // (w,x,y,z)
    glm::vec3 scale       = glm::vec3(1.0f);

    glm::mat4 toMatrix() const {
        const glm::mat4 t = glm::translate(glm::mat4(1.0f), translation);
        const glm::mat4 r = glm::mat4_cast(rotation);
        const glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }
};

// Cached world-space matrix, recomputed by transform_system from LocalTransform
// (+ Parent's WorldTransform). Read by render / streaming / culling.
struct WorldTransform {
    glm::mat4 matrix = glm::mat4(1.0f);

    glm::vec3 position() const { return glm::vec3(matrix[3]); }
};

// Parent link. parent must be a valid Entity that also has a WorldTransform.
struct Parent {
    Entity parent = kNull;
};

// ── Bounds ───────────────────────────────────────────────────────────────--
// Object-space AABB (center + half-extents). Filled in when an asset's mesh
// bounds become known; used to derive WorldBounds for streaming + culling.
struct LocalBounds {
    glm::vec3 center  = glm::vec3(0.0f);
    glm::vec3 extents = glm::vec3(0.0f);  // half-size
};

// World-space AABB recomputed alongside WorldTransform.
struct WorldBounds {
    glm::vec3 center  = glm::vec3(0.0f);
    glm::vec3 extents = glm::vec3(0.0f);
};

// ── Tags (empty payload) ─────────────────────────────────────────────────--
struct Active {};          // entity participates in systems (gate)
struct Visible {};         // entity should be drawn (render gather gate)
struct Static {};          // transform never changes after spawn (skip in dirty pass)
struct DirtyTransform {};   // LocalTransform changed; world needs recompute
struct PendingDestroy {};   // queued for deferred destruction by lifetime system
struct Culled {};          // set by CullingSystem when outside the view frustum

}  // namespace ecs
}  // namespace engine
