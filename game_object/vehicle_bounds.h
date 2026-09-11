#pragma once
#include <cmath>
#include <algorithm>
#include "glm/glm.hpp"

namespace engine::game_object {
struct VehicleBounds {
    glm::vec3 centre{0}, half{0};
    glm::vec3 axis[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
};
inline VehicleBounds vehicleBounds(glm::vec3 pos, float yaw, glm::vec3 up,
                                    glm::vec3 lo, glm::vec3 hi) {
    VehicleBounds b;
    b.axis[1] = glm::normalize(up);
    glm::vec3 heading(std::sin(yaw), 0, std::cos(yaw));
    b.axis[0] = glm::normalize(heading - b.axis[1]*glm::dot(heading,b.axis[1]));
    b.axis[2] = glm::normalize(glm::cross(b.axis[0], b.axis[1]));
    const auto local = (lo+hi)*0.5f;
    b.centre = pos;
    for (int k=0;k<3;++k) b.centre += b.axis[k]*local[k];
    b.half = (hi-lo)*0.5f + glm::vec3(0.15f); // 30 cm total clearance
    return b;
}
inline bool vehicleBoundsOverlap(const VehicleBounds& a, const VehicleBounds& b) {
    const auto d = b.centre-a.centre;
    // Separating axis test: faces and all nine edge cross-products.
    auto separated = [&](glm::vec3 n) {
        if (glm::dot(n,n)<1e-10f) return false;
        float radius=0;
        for(int k=0;k<3;++k)
            radius += a.half[k]*std::abs(glm::dot(a.axis[k],n)) +
                      b.half[k]*std::abs(glm::dot(b.axis[k],n));
        return std::abs(glm::dot(d,n)) >= radius;
    };
    for(int k=0;k<3;++k) if(separated(a.axis[k]) || separated(b.axis[k])) return false;
    for(int k=0;k<3;++k) for(int j=0;j<3;++j)
        if(separated(glm::cross(a.axis[k],b.axis[j]))) return false;
    return true;
}
inline VehicleBounds sweptVehicleBounds(const VehicleBounds& a, const VehicleBounds& b) {
    VehicleBounds result=a;
    const auto motion=b.centre-a.centre;
    result.centre += motion*0.5f;
    float turn=0;
    for(int k=0;k<3;++k) turn += glm::length(b.axis[k]-a.axis[k]);
    const float margin=glm::length(glm::max(a.half,b.half))*turn;
    for(int k=0;k<3;++k)
        result.half[k] += std::abs(glm::dot(motion,a.axis[k]))*0.5f+margin;
    return result;
}
} // namespace engine::game_object
