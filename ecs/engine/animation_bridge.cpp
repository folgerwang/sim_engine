// animation_bridge.cpp — see animation_bridge.h.
#include "ecs/engine/animation_bridge.h"

#include <algorithm>

#include "game_object/drawable_object.h"

namespace engine {
namespace ecs {

std::vector<AnimationClip>
AnimationBridge::extractClips(const game_object::DrawableObject& drawable) {
    std::vector<AnimationClip> clips;
    if (!drawable.isReady()) return clips;

    const auto& data = drawable.getDrawableData();
    clips.reserve(data.animations_.size());

    for (const auto& anim : data.animations_) {
        AnimationClip clip;
        float duration = 0.0f;

        for (const auto& ch_ptr : anim.channels_) {
            if (!ch_ptr) continue;
            const auto& ch = *ch_ptr;

            AnimChannel oc;
            oc.target_node = static_cast<int32_t>(ch.node_idx_);
            oc.interp      = AnimInterp::kLinear;  // glTF default (engine uses linear)
            switch (ch.type_) {
            case game_object::AnimChannelInfo::kTranslation:
                oc.path = AnimPath::kTranslation; break;
            case game_object::AnimChannelInfo::kScale:
                oc.path = AnimPath::kScale; break;
            case game_object::AnimChannelInfo::kRotation:
            default:
                oc.path = AnimPath::kRotation; break;
            }

            oc.times.reserve(ch.samples_.size());
            const bool is_rot = (oc.path == AnimPath::kRotation);
            for (const auto& [t, v] : ch.samples_) {
                oc.times.push_back(t);
                if (is_rot) {
                    // glTF stores rotation keyframes as (x,y,z,w); glm::quat
                    // is constructed (w,x,y,z).
                    oc.quat.emplace_back(v.w, v.x, v.y, v.z);
                } else {
                    oc.vec.emplace_back(v.x, v.y, v.z);
                }
                duration = std::max(duration, t);
            }
            clip.channels.push_back(std::move(oc));
        }

        clip.duration = duration;
        clips.push_back(std::move(clip));
    }
    return clips;
}

void AnimationBridge::applyPose(game_object::DrawableObject& drawable,
                                const AnimPose& pose) {
    const uint32_t node_count = drawable.getNodeCount();
    const uint32_t n =
        std::min<uint32_t>(node_count, static_cast<uint32_t>(pose.nodes.size()));
    for (uint32_t i = 0; i < n; ++i) {
        const NodeTRS& s = pose.nodes[i];
        if (!s.has_t && !s.has_r && !s.has_s) continue;

        glm::vec3 t, scale;
        glm::quat r;
        if (!drawable.getNodeLocalTRS(i, t, r, scale)) continue;  // bind defaults
        if (s.has_t) t     = s.translation;
        if (s.has_r) r     = s.rotation;
        if (s.has_s) scale = s.scale;
        drawable.setNodeLocalTRS(i, t, r, scale);
    }
}

}  // namespace ecs
}  // namespace engine
