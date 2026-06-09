#pragma once
// animation_bridge.h — connects the renderer-agnostic AnimationSystem to the
// engine's DrawableObject (engine side only).
//
//   extractClips(drawable)  reads the parsed glTF animation channels exposed by
//                           DrawableObject::getDrawableData() and converts them
//                           into ecs::AnimationClip(s) the AnimationSystem can
//                           sample. Call once when the drawable becomes ready;
//                           cache the result (clips are reused across frames and
//                           shared if the same asset is instanced).
//   applyPose(drawable,pose) writes an AnimPose (sampled by AnimationSystem)
//                           back onto the drawable's node hierarchy via
//                           setNodeLocalTRS, preserving the bind value of any
//                           channel the clip did not drive. Pair with
//                           DrawableObject::setExternalAnimation(true) so the
//                           imported channel evaluation does not overwrite it.
#include <vector>

#include "ecs/animation_system.h"

namespace engine {
namespace game_object { class DrawableObject; }

namespace ecs {

struct AnimationBridge {
    // One AnimationClip per imported animation (empty if the drawable has none
    // or is not ready). target_node indices match the drawable's node order.
    static std::vector<AnimationClip>
    extractClips(const game_object::DrawableObject& drawable);

    // Write the sampled pose onto the drawable's nodes. Unspecified channels
    // keep their current (bind) value.
    static void applyPose(game_object::DrawableObject& drawable,
                          const AnimPose& pose);
};

}  // namespace ecs
}  // namespace engine
