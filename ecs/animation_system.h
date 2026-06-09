#pragma once
// animation_system.h — renderer-agnostic skeletal/node animation playback.
//
// Decouples animation evaluation from DrawableObject::update. An AnimationClip
// is a set of keyframed channels (translation / rotation / scale per target
// node), exactly the glTF animation model. An entity carries an AnimationPlayer
// (clip + time + speed + loop) and the system advances it each frame and samples
// the clip into an AnimPose (node index -> sampled TRS). The engine then applies
// that pose onto the matching DrawableObject node hierarchy (the wiring step).
//
// Pure and unit-testable: no Vulkan, no DrawableObject. Clips are owned by an
// asset cache elsewhere; the player references one by const pointer.
#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "ecs/entity.h"

namespace engine {
namespace ecs {

enum class AnimPath   : uint8_t { kTranslation, kRotation, kScale };
enum class AnimInterp : uint8_t { kStep, kLinear };

// One animated property of one node. `times` is sorted ascending. For
// translation/scale the samples live in `vec`; for rotation in `quat`. The two
// sample arrays are parallel to `times` (same length).
struct AnimChannel {
    int32_t                 target_node = -1;
    AnimPath                path        = AnimPath::kTranslation;
    AnimInterp              interp      = AnimInterp::kLinear;
    std::vector<float>      times;
    std::vector<glm::vec3>  vec;    // translation / scale keyframes
    std::vector<glm::quat>  quat;   // rotation keyframes
};

struct AnimationClip {
    std::string             name;
    float                   duration = 0.0f;   // seconds (max keyframe time)
    std::vector<AnimChannel> channels;
};

// Sampled transform for one node. has_* flags tell the engine which fields the
// clip actually drove (others should keep the node's authored bind value).
struct NodeTRS {
    glm::vec3 translation = glm::vec3(0.0f);
    glm::quat rotation    = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale       = glm::vec3(1.0f);
    bool      has_t = false, has_r = false, has_s = false;
};

// Per-entity sampled pose, indexed by node. Recomputed each frame.
struct AnimPose {
    std::vector<NodeTRS> nodes;
};

// Playback state. `clip` is non-owning (asset cache owns it).
struct AnimationPlayer {
    const AnimationClip* clip    = nullptr;
    float                time    = 0.0f;
    float                speed   = 1.0f;
    bool                 loop    = true;
    bool                 playing = true;
};

class AnimationSystem {
public:
    // Advance every AnimationPlayer by dt and sample its clip into the entity's
    // AnimPose (created if absent). Returns the number of players updated.
    static size_t update(entt::registry& reg, float dt);

    // Sample `clip` at absolute `time` (seconds) into `out`. out.nodes is grown
    // to cover the highest target node. Pure; used directly by tests.
    static void sample(const AnimationClip& clip, float time, AnimPose& out);
};

}  // namespace ecs
}  // namespace engine
