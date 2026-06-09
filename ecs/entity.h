#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// entity.h — core entity type for the ECS layer.
//
// The engine's ECS is built on EnTT (third_parties/entt). An Entity is just
// EnTT's generational handle: a 32-bit value packing an index + a version.
// When an entity is destroyed the index is recycled with a bumped version, so
// any stale copy of the old handle compares != the live one and World::valid()
// returns false. That generational check is the backbone of the GC-safe
// lifetime model — no dangling raw pointers to game objects anywhere.
//
// This header is renderer-agnostic on purpose: the whole core ECS (entity,
// components, transform / streaming / lifetime systems, World) compiles with
// no Vulkan dependency, which keeps it unit-testable in isolation.
// ─────────────────────────────────────────────────────────────────────────────
#include <entt/entt.hpp>

namespace engine {
namespace ecs {

// The one entity handle type used throughout the ECS.
using Entity = entt::entity;

// A null/invalid entity. Compare with `e == kNull` or use World::valid(e).
inline constexpr Entity kNull = entt::null;

}  // namespace ecs
}  // namespace engine
