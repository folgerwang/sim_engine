#pragma once
#include <glm/glm.hpp>

namespace engine::helper {
// XZ bounds, packed as xmin/zmin/xmax/zmax. Shared by visible geometry
// and RT casters so altitude and wrapper transforms cannot select different LODs.
inline glm::vec4 pcgLodTileRect(const glm::mat4& m, float x, float z, float size) {
    glm::vec4 rect(0.0f);
    for (int i = 0; i < 4; ++i) {
        const glm::vec3 p(m * glm::vec4(x + (i & 1) * size, 0.0f,
                                      z + ((i >> 1) & 1) * size, 1.0f));
        if (i == 0) rect = glm::vec4(p.x, p.z, p.x, p.z);
        else {
            rect.x = glm::min(rect.x, p.x); rect.y = glm::min(rect.y, p.z);
            rect.z = glm::max(rect.z, p.x); rect.w = glm::max(rect.w, p.z);
        }
    }
    return rect;
}
inline float pcgLodDistanceSquared(const glm::vec4& rect, const glm::vec3& eye) {
    const float dx = glm::max(glm::max(rect.x - eye.x, 0.0f), eye.x - rect.z);
    const float dz = glm::max(glm::max(rect.y - eye.z, 0.0f), eye.z - rect.w);
    return dx * dx + dz * dz;
}
inline bool pcgLodOwnsDistance(float d2, float near_m, float far_m) {
    return d2 >= near_m * near_m && d2 < far_m * far_m;
}
}
