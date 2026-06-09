#pragma once
// culling_system.h — frustum culling over WorldBounds (renderer-agnostic).
//
// Tags entities whose world-space AABB lies fully outside the view frustum with
// the Culled component, and clears it when they re-enter. Pure data-oriented:
// it reads WorldBounds (produced by the transform system) and toggles a tag,
// with no renderer dependency, so it is unit-tested in isolation. The engine
// extracts FrustumPlanes from the camera view-projection and decides how to act
// on Culled (skip the draw, drop from a render list, etc.).
#include <cstddef>

#include <glm/glm.hpp>

#include "ecs/entity.h"

namespace engine {
namespace ecs {

// Six frustum planes, each (nx,ny,nz,d) with the normal pointing INWARD: a
// point p is inside the half-space when dot(n, p) + d >= 0.
struct FrustumPlanes {
    glm::vec4 planes[6];

    // Gribb–Hartmann extraction from a view-projection matrix (row-major access
    // via glm's column-major storage). Normalized so distances are metric.
    static FrustumPlanes fromViewProj(const glm::mat4& vp);
};

struct CullingStats {
    size_t tested  = 0;
    size_t culled  = 0;
    size_t visible = 0;
};

class CullingSystem {
public:
    // Toggle the Culled tag for every entity with WorldBounds against `frustum`.
    static CullingStats update(entt::registry& reg, const FrustumPlanes& frustum);

    // True when the center/half-extents AABB is fully outside the frustum.
    static bool aabbOutside(const glm::vec3& center, const glm::vec3& extents,
                            const FrustumPlanes& frustum);
};

}  // namespace ecs
}  // namespace engine
