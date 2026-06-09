// animation_system.cpp — see animation_system.h.
#include "ecs/animation_system.h"

#include <algorithm>
#include <cmath>

#include <entt/entt.hpp>

namespace engine {
namespace ecs {

namespace {

// Locate the keyframe segment for `t`: returns index i such that
// times[i] <= t <= times[i+1], plus the normalized factor in [0,1]. Clamps to
// the ends when t is outside the range. Assumes times is non-empty + sorted.
struct Seg { size_t i0, i1; float f; };

Seg findSeg(const std::vector<float>& times, float t) {
    const size_t n = times.size();
    if (n == 1 || t <= times.front()) return {0, 0, 0.0f};
    if (t >= times.back())            return {n - 1, n - 1, 0.0f};
    // upper_bound: first key strictly greater than t.
    size_t hi = static_cast<size_t>(
        std::upper_bound(times.begin(), times.end(), t) - times.begin());
    const size_t i1 = hi;
    const size_t i0 = hi - 1;
    const float span = times[i1] - times[i0];
    const float f = span > 1e-8f ? (t - times[i0]) / span : 0.0f;
    return {i0, i1, f};
}

}  // namespace

void AnimationSystem::sample(const AnimationClip& clip, float time, AnimPose& out) {
    // Grow the pose to cover every target node referenced by the clip.
    int32_t max_node = -1;
    for (const auto& ch : clip.channels)
        max_node = std::max(max_node, ch.target_node);
    if (max_node < 0) return;
    if (out.nodes.size() < static_cast<size_t>(max_node + 1))
        out.nodes.resize(max_node + 1);

    for (const auto& ch : clip.channels) {
        if (ch.target_node < 0 || ch.times.empty()) continue;
        const Seg seg = findSeg(ch.times, time);
        NodeTRS& trs = out.nodes[ch.target_node];

        switch (ch.path) {
        case AnimPath::kTranslation:
        case AnimPath::kScale: {
            if (ch.vec.empty()) break;
            glm::vec3 v = ch.vec[std::min(seg.i0, ch.vec.size() - 1)];
            if (ch.interp == AnimInterp::kLinear && seg.i1 < ch.vec.size() &&
                seg.i1 != seg.i0) {
                v = glm::mix(ch.vec[seg.i0], ch.vec[seg.i1], seg.f);
            }
            if (ch.path == AnimPath::kTranslation) { trs.translation = v; trs.has_t = true; }
            else                                   { trs.scale = v;       trs.has_s = true; }
            break;
        }
        case AnimPath::kRotation: {
            if (ch.quat.empty()) break;
            glm::quat q = ch.quat[std::min(seg.i0, ch.quat.size() - 1)];
            if (ch.interp == AnimInterp::kLinear && seg.i1 < ch.quat.size() &&
                seg.i1 != seg.i0) {
                q = glm::slerp(ch.quat[seg.i0], ch.quat[seg.i1], seg.f);
            }
            trs.rotation = glm::normalize(q);
            trs.has_r = true;
            break;
        }
        }
    }
}

size_t AnimationSystem::update(entt::registry& reg, float dt) {
    size_t n = 0;
    for (auto [e, player] : reg.view<AnimationPlayer>().each()) {
        if (!player.clip || player.clip->duration <= 0.0f) continue;

        const float dur = player.clip->duration;
        if (player.playing) {
            player.time += dt * player.speed;
            if (player.loop) {
                player.time = std::fmod(player.time, dur);
                if (player.time < 0.0f) player.time += dur;   // wrap negatives
            } else {
                player.time = std::clamp(player.time, 0.0f, dur);
            }
        }

        AnimPose& pose = reg.all_of<AnimPose>(e) ? reg.get<AnimPose>(e)
                                                 : reg.emplace<AnimPose>(e);
        sample(*player.clip, player.time, pose);
        ++n;
    }
    return n;
}

}  // namespace ecs
}  // namespace engine
