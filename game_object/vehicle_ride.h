#pragma once
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>

namespace engine::game_object {
// Exact critically damped response: stable even after a long render frame.
inline void dampRide(float target, float dt, float frequency, float& value, float& velocity) {
    dt = std::clamp(dt, 0.0f, 0.1f);
    const float omega = 6.2831853f * frequency;
    const float error = value - target;
    const float c = velocity + omega * error;
    const float decay = std::exp(-omega * dt);
    value = target + (error + c * dt) * decay;
    velocity = (velocity - omega * c * dt) * decay;
}
// Bicycle steering from actual distance/yaw, then Ackermann per hub.
inline float roadSteer(float yawDelta, float distance, float wheelbase) {
    return distance > 0.001f ? std::clamp(std::atan(wheelbase * yawDelta / distance), -0.55f, 0.55f) : 0.0f;
}
inline float hubSteer(float steer, float wheelbase, float lateral) {
    if (std::abs(steer) < 1e-5f) return 0.0f;
    const float radius = wheelbase / std::tan(steer);
    return std::clamp(std::atan(wheelbase / (radius + lateral)), -0.7f, 0.7f);
}
// Legacy dual is the total tyre envelope in units of one tyre width.
// Keep that envelope while adding a visible gap between the two tyres.
inline float dualTyreWidth(float width, float dual) {
    return dual > 1.05f ? width * (dual - 0.10f) * 0.5f : width;
}
inline float dualTyreCentre(float track, float width, float dual, int tyre) {
    if (dual <= 1.05f) return track;
    const float outer = track + width * 0.5f;
    const float single = dualTyreWidth(width, dual);
    return outer - single * 0.5f - tyre * (single + width * 0.10f);
}
} // namespace engine::game_object
