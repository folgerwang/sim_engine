// culling_system.cpp — see culling_system.h.
#include "ecs/culling_system.h"

#include <entt/entt.hpp>

#include "ecs/components.h"

namespace engine {
namespace ecs {

FrustumPlanes FrustumPlanes::fromViewProj(const glm::mat4& vp) {
    // glm is column-major: vp[col][row]. Build rows for Gribb–Hartmann.
    const glm::vec4 r0(vp[0][0], vp[1][0], vp[2][0], vp[3][0]);
    const glm::vec4 r1(vp[0][1], vp[1][1], vp[2][1], vp[3][1]);
    const glm::vec4 r2(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);
    const glm::vec4 r3(vp[0][3], vp[1][3], vp[2][3], vp[3][3]);

    FrustumPlanes f;
    f.planes[0] = r3 + r0;  // left
    f.planes[1] = r3 - r0;  // right
    f.planes[2] = r3 + r1;  // bottom
    f.planes[3] = r3 - r1;  // top
    f.planes[4] = r3 + r2;  // near
    f.planes[5] = r3 - r2;  // far
    for (auto& p : f.planes) {
        const float len = glm::length(glm::vec3(p));
        if (len > 1e-8f) p /= len;
    }
    return f;
}

bool CullingSystem::aabbOutside(const glm::vec3& center, const glm::vec3& extents,
                                const FrustumPlanes& frustum) {
    for (const auto& p : frustum.planes) {
        const glm::vec3 n(p);
        const float dist   = glm::dot(n, center) + p.w;       // signed dist to plane
        const float radius = glm::dot(glm::abs(n), extents);  // AABB projection radius
        if (dist + radius < 0.0f) return true;                // fully behind a plane
    }
    return false;
}

CullingStats CullingSystem::update(entt::registry& reg,
                                                  const FrustumPlanes& frustum) {
    CullingStats stats;
    for (auto [e, wb] : reg.view<WorldBounds>().each()) {
        ++stats.tested;
        const bool culled = aabbOutside(wb.center, wb.extents, frustum);
        if (culled) {
            ++stats.culled;
            if (!reg.all_of<Culled>(e)) reg.emplace<Culled>(e);
        } else {
            ++stats.visible;
            if (reg.all_of<Culled>(e)) reg.remove<Culled>(e);
        }
    }
    return stats;
}

}  // namespace ecs
}  // namespace engine
