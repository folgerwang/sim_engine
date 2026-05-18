#include "player_controller.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>

#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "drawable_object.h"
#include "terrain.h"
#include "helper/collision_mesh.h"

namespace engine {
namespace game_object {

namespace {

float wrapDeg(float a) {
    a = std::fmod(a + 180.0f, 360.0f);
    if (a < 0.0f) a += 360.0f;
    return a - 180.0f;
}

float chaseAngleDeg(float current, float target, float max_step) {
    float diff = wrapDeg(target - current);
    if (std::fabs(diff) <= max_step) return target;
    return current + (diff > 0.0f ? max_step : -max_step);
}

glm::quat axisAngleDeg(const glm::vec3& axis, float deg) {
    return glm::angleAxis(glm::radians(deg), axis);
}

} // namespace

PlayerController::PlayerController() = default;

void PlayerController::update(
    GLFWwindow* window,
    float delta_t,
    float camera_yaw_deg,
    const std::shared_ptr<DrawableObject>& player,
    const std::vector<std::shared_ptr<DrawableObject>>& obstacles,
    const helper::CollisionWorld* world) {

    if (!player || !player->isReady()) return;

    // Wait for the application to call spawnAt() once the level
    // finishes loading.  Until then, leave the rig at its glTF rest
    // pose instead of teleporting it to the world origin.
    if (!initialized_) return;

    // ── Stationary-but-animated mode ─────────────────────────────────
    // We still don't own movement here — the application's per-frame
    // follow block writes position_ + yaw_deg_ via setPositionAndYaw
    // each frame.  But we DO drive the skeleton's procedural pose:
    //   • idle_phase_ advances every frame so the spine sway plays
    //     even when standing still (otherwise the character looks
    //     posed-but-frozen, like a mannequin).
    //   • anim_phase_ advances proportional to horizontal walking
    //     speed.  We measure speed by diffing position_ against the
    //     previous frame's position_ (set by the external follow
    //     block), so any caller — keyboard, AI pathing, camera-glue —
    //     that moves us will produce a matching limb swing.
    //   • walking_ is latched on when the speed exceeds a small
    //     threshold (0.05 m/s) and off otherwise.  applyPose() reads
    //     it to gate the arm / leg / knee swing amplitudes; idle sway
    //     plays in both cases.
    // The other params (window / camera_yaw / obstacles / world)
    // remain unused while the controller stays in this "external
    // motion source" mode — they're kept on the signature for the
    // future WASD-self-driven version.
    (void)window;
    (void)camera_yaw_deg;
    (void)obstacles;
    (void)world;

    // ── Movement detection (XZ only — vertical jitter from the
    // bistro-floor pivot calc shouldn't count as walking) ──────────
    float speed_mps = 0.0f;
    if (last_position_valid_ && delta_t > 1e-5f) {
        glm::vec2 dxz =
            glm::vec2(position_.x, position_.z) -
            glm::vec2(last_position_.x, last_position_.z);
        speed_mps = glm::length(dxz) / delta_t;
    }
    last_position_       = position_;
    last_position_valid_ = true;

    // 0.05 m/s threshold filters out the sub-mm-per-frame jitter the
    // pivot-compensation math produces while the camera is still.
    // Anything above that (walking, running, glued-to-camera-pan)
    // engages the walking animation.
    constexpr float kWalkingSpeedThreshold = 0.05f;
    walking_ = speed_mps > kWalkingSpeedThreshold;

    // ── Phase advance ───────────────────────────────────────────────
    // idle_phase_: slow constant rate.  ~1 rad/s gives a 6.28 s
    // breathing cycle, fast enough to read as alive, slow enough not
    // to look like she's swaying drunk.
    constexpr float kIdleRateRad = 1.0f;
    idle_phase_ += kIdleRateRad * delta_t;
    if (idle_phase_ > 6.28318530718f) {
        idle_phase_ -= 6.28318530718f;
    }

    // anim_phase_: cadence proportional to walking speed.  A normal
    // human walking gait is roughly 2 steps/second at ~1.4 m/s, which
    // is about 4.5 rad/s of swing phase (one full cycle = two steps).
    // We scale by speed so a slow shuffle has a slow swing and a
    // sprint has a brisk one.  Capped to a sane max so a teleport
    // doesn't produce a one-frame helicopter-arm spin.
    if (walking_) {
        constexpr float kSwingRadPerMps = 4.5f / 1.4f; // ≈3.21 rad/s per m/s
        constexpr float kMaxSwingRad    = 12.0f;       // ~2 cycles/sec ceiling
        float swing_rate =
            std::min(speed_mps * kSwingRadPerMps, kMaxSwingRad);
        anim_phase_ += swing_rate * delta_t;
        if (anim_phase_ > 6.28318530718f) {
            anim_phase_ -= 6.28318530718f;
        }
    } else {
        // Standing still: decay the swing toward 0 so arms come
        // back to rest instead of freezing mid-stride.  τ ≈ 1/6 s.
        constexpr float kAnimDecay = 6.0f;
        anim_phase_ *= std::max(0.0f, 1.0f - kAnimDecay * delta_t);
    }

    applyPose(player);
}

glm::vec2 PlayerController::resolveXzCollisions(
    const glm::vec2& current_xz,
    const glm::vec2& desired_xz,
    const std::vector<std::shared_ptr<DrawableObject>>& obstacles) const {

    glm::vec2 result = desired_xz;
    for (const auto& obj : obstacles) {
        if (!obj || !obj->isReady()) continue;
        glm::vec3 mn = obj->getModelBboxMin();
        glm::vec3 mx = obj->getModelBboxMax();
        if (mn.x >= mx.x || mn.z >= mx.z) continue;

        const glm::mat4& M = obj->getLocation();
        glm::vec3 cs[4] = {
            glm::vec3(M * glm::vec4(mn.x, 0.0f, mn.z, 1.0f)),
            glm::vec3(M * glm::vec4(mx.x, 0.0f, mn.z, 1.0f)),
            glm::vec3(M * glm::vec4(mn.x, 0.0f, mx.z, 1.0f)),
            glm::vec3(M * glm::vec4(mx.x, 0.0f, mx.z, 1.0f)),
        };
        glm::vec2 amn(cs[0].x, cs[0].z), amx = amn;
        for (int i = 1; i < 4; ++i) {
            amn = glm::min(amn, glm::vec2(cs[i].x, cs[i].z));
            amx = glm::max(amx, glm::vec2(cs[i].x, cs[i].z));
        }
        amn -= glm::vec2(player_radius_);
        amx += glm::vec2(player_radius_);

        if (result.x <= amn.x || result.x >= amx.x ||
            result.y <= amn.y || result.y >= amx.y) continue;

        float dx_neg = result.x - amn.x;
        float dx_pos = amx.x - result.x;
        float dz_neg = result.y - amn.y;
        float dz_pos = amx.y - result.y;
        float min_x = std::min(dx_neg, dx_pos);
        float min_z = std::min(dz_neg, dz_pos);

        if (min_x < min_z) {
            result.x = (dx_neg < dx_pos) ? amn.x : amx.x;
        } else {
            result.y = (dz_neg < dz_pos) ? amn.y : amx.y;
        }
    }
    return result;
}

void PlayerController::applyPose(
    const std::shared_ptr<DrawableObject>& player) {

    glm::quat root_rot = axisAngleDeg(glm::vec3(0, 1, 0), yaw_deg_);
    player->setRootNodeTransform(position_, root_rot);

    // ── First-frame bind-pose capture ────────────────────────────────
    // The previous version did `setNodeRotationByName(bone, axisAngle(X, deg))`,
    // which REPLACED the asset's authored bone rotation (the bind pose).
    // For rigs where the bone's local axes don't align with world XYZ,
    // that meant rotating around the WRONG axis — every animation
    // frame looked broken (twist where it should be swing, etc.) AND
    // the bind orientation was destroyed in the process.
    //
    // Fix: snapshot each bone's asset-authored rotation once (the
    // first time we have a ready drawable to read from), then on
    // every subsequent frame write `bind_rot * delta_rot_in_local_frame`.
    // The delta is the procedural swing we author each frame; the
    // bind preserves the bone's orientation.
    if (!bind_rots_captured_) {
        static const char* kBoneNames[] = {
            "left_upper_arm",  "right_upper_arm",
            "left_lower_arm",  "right_lower_arm",
            "left_upper_leg",  "right_upper_leg",
            "left_lower_leg",  "right_lower_leg",
            "spine",
        };
        bool any_resolved = false;
        for (const char* nm : kBoneNames) {
            int idx = player->findNodeIndexByName(nm);
            if (idx < 0) continue; // bone missing on this rig — skip
            BoneBindRot b;
            b.rot      = player->getNodeRotationByName(nm);
            b.node_idx = idx;
            bind_rots_[nm] = b;
            any_resolved = true;
        }
        // Only mark captured if at least one bone resolved — otherwise
        // we're called before isReady() really populated the rig and
        // would lock in identity rotations.  Re-tries on every frame
        // until at least one bone name lands.
        if (any_resolved) {
            bind_rots_captured_ = true;
        }
    }

    // ── Procedural swing deltas (in bone-local space) ────────────────
    // amp_arm/amp_leg ramp linearly with anim_phase' amplitude so when
    // the controller's update() decays anim_phase_ back to 0 the limbs
    // come gracefully to rest instead of snapping.
    const float walk_arm_amp_deg = walking_ ? 35.0f : 0.0f;
    const float walk_leg_amp_deg = walking_ ? 30.0f : 0.0f;
    const float idle_sway_deg    = std::sin(idle_phase_) * 1.5f;

    const float swing     = std::sin(anim_phase_);
    const float swing_opp = -swing;

    // Swing axis in BONE-LOCAL frame.  For most humanoid rigs exported
    // from Blender/Mixamo/Maya, the bone's local X axis runs down the
    // bone length and the local Z axis is "forward swing".  We rotate
    // around local-X here as a starting point — that matches the
    // existing applyPose code's intent and gives a reasonable swing
    // for the scene-skinned.gltf auto-rig.  If a future rig needs a
    // different axis, the bind-rot composition (below) means the swing
    // remains in the bone's LOCAL frame regardless of how the bind
    // rotation orients that frame in world space.
    const glm::vec3 swing_axis_local(1, 0, 0);

    const glm::quat dq_l_arm   = axisAngleDeg(swing_axis_local, swing     * walk_arm_amp_deg);
    const glm::quat dq_r_arm   = axisAngleDeg(swing_axis_local, swing_opp * walk_arm_amp_deg);
    const glm::quat dq_l_leg   = axisAngleDeg(swing_axis_local, swing_opp * walk_leg_amp_deg);
    const glm::quat dq_r_leg   = axisAngleDeg(swing_axis_local, swing     * walk_leg_amp_deg);
    const glm::quat dq_l_knee  = axisAngleDeg(swing_axis_local,
                                              std::max(0.0f, swing_opp) * 25.0f * (walking_ ? 1.0f : 0.0f));
    const glm::quat dq_r_knee  = axisAngleDeg(swing_axis_local,
                                              std::max(0.0f, swing)     * 25.0f * (walking_ ? 1.0f : 0.0f));
    const glm::quat dq_l_elbow = axisAngleDeg(swing_axis_local,
                                              std::max(0.0f, swing)     * 20.0f * (walking_ ? 1.0f : 0.0f));
    const glm::quat dq_r_elbow = axisAngleDeg(swing_axis_local,
                                              std::max(0.0f, swing_opp) * 20.0f * (walking_ ? 1.0f : 0.0f));
    const glm::quat dq_spine   = axisAngleDeg(glm::vec3(0, 0, 1), idle_sway_deg);

    // ── Compose bind * delta and write ───────────────────────────────
    // bind * delta means: start at the bone's authored orientation,
    // then apply the swing in the bone's LOCAL frame.  This is the
    // standard "additive animation on top of bind pose" formula.
    auto applyBoneDelta = [&](const char* name, const glm::quat& dq) {
        auto it = bind_rots_.find(name);
        if (it == bind_rots_.end()) return; // bone missing on this rig
        const glm::quat& bind = it->second.rot;
        player->setNodeRotationByName(name, bind * dq);
    };

    applyBoneDelta("left_upper_arm",  dq_l_arm);
    applyBoneDelta("right_upper_arm", dq_r_arm);
    applyBoneDelta("left_lower_arm",  dq_l_elbow);
    applyBoneDelta("right_lower_arm", dq_r_elbow);
    applyBoneDelta("left_upper_leg",  dq_l_leg);
    applyBoneDelta("right_upper_leg", dq_r_leg);
    applyBoneDelta("left_lower_leg",  dq_l_knee);
    applyBoneDelta("right_lower_leg", dq_r_knee);
    applyBoneDelta("spine",           dq_spine);
}

} // namespace game_object
} // namespace engine
