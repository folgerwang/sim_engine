#include "player_controller.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

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

// Extract a pure rotation quaternion from a (possibly scaled) 4x4 by
// normalising the basis columns.  The rig bakes a uniform 0.1 scale into
// the root, so the world matrices carry that scale; dividing each column
// by its length removes it and leaves the bone's world orientation.
glm::quat matRotation(const glm::mat4& m) {
    glm::vec3 c0(m[0]), c1(m[1]), c2(m[2]);
    const float l0 = glm::length(c0);
    const float l1 = glm::length(c1);
    const float l2 = glm::length(c2);
    if (l0 < 1e-8f || l1 < 1e-8f || l2 < 1e-8f)
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    // Gram-Schmidt orthonormalisation rather than a naive per-column
    // normalise: robust to NON-uniform bone scale and slight shear in
    // the cached matrix (either of which makes a raw quat_cast return
    // garbage and throw the IK into a wild pose).  Handedness of the
    // original basis is preserved so a mirrored rig is not flipped.
    const glm::vec3 x = c0 / l0;
    glm::vec3 y = c1 - x * glm::dot(x, c1);
    const float ly = glm::length(y);
    if (ly < 1e-8f) return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    y /= ly;
    glm::vec3 z = glm::cross(x, y);
    if (glm::dot(z, c2) < 0.0f) z = -z;
    const glm::mat3 r(x, y, z);
    return glm::normalize(glm::quat_cast(r));
}

// Shortest-arc axis+angle that rotates `a` onto `b` (both treated as
// directions).  Returns false on degenerate (zero-length) input; for the
// antiparallel case it fabricates a stable perpendicular axis.
bool axisAngleBetween(glm::vec3 a, glm::vec3 b,
                      glm::vec3& out_axis, float& out_angle) {
    const float la = glm::length(a), lb = glm::length(b);
    if (la < 1e-6f || lb < 1e-6f) return false;
    a /= la; b /= lb;
    const float d = glm::clamp(glm::dot(a, b), -1.0f, 1.0f);
    if (d > 0.999999f) {
        out_axis = glm::vec3(0.0f, 1.0f, 0.0f); out_angle = 0.0f; return true;
    }
    if (d < -0.999999f) {
        glm::vec3 c = glm::cross(a, glm::vec3(1.0f, 0.0f, 0.0f));
        if (glm::length(c) < 1e-6f) c = glm::cross(a, glm::vec3(0.0f, 0.0f, 1.0f));
        out_axis = glm::normalize(c);
        out_angle = 3.14159265358979f;
        return true;
    }
    out_axis  = glm::normalize(glm::cross(a, b));
    out_angle = std::acos(d);
    return true;
}

// Shortest-arc quaternion rotating direction `from` onto `to`.
// Identity when (anti)parallel; reuses axisAngleBetween for the axis.
glm::quat fromTo(const glm::vec3& from, const glm::vec3& to) {
    glm::vec3 ax; float ang;
    if (!axisAngleBetween(from, to, ax, ang))
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    return glm::angleAxis(ang, ax);
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
        // Latch the travel direction for the foot-IK gait step
        // placement (kept from the last real step while standing).
        const float dlen = glm::length(dxz);
        if (dlen > 1e-4f) gait_fwd_xz_ = dxz / dlen;
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
    // Apply the IK pelvis drop computed by applyFootIk on the
    // PREVIOUS frame as a negative-Y offset, so a foot that could not
    // reach its ground target pulls the whole body down instead of
    // leaving the leg dangling.  0 in normal terrain.
    glm::vec3 root_pos = position_;
    root_pos.y -= pelvis_drop_;
    player->setRootNodeTransform(root_pos, root_rot);

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
    applyBoneDelta("spine",           dq_spine);

    // Legs: two-bone foot IK owns them when active (it plants each
    // foot on the ground, tilts to the surface, and drops the pelvis
    // when a foot cannot reach).  Fall back to the procedural
    // leg/knee swing only when IK is unavailable (disabled, no
    // ground-query wired, or the rig has no leg bones) so behaviour
    // never regresses to the old floating-feet look.
    if (!applyFootIk(player)) {
        applyBoneDelta("left_upper_leg",  dq_l_leg);
        applyBoneDelta("right_upper_leg", dq_r_leg);
        applyBoneDelta("left_lower_leg",  dq_l_knee);
        applyBoneDelta("right_lower_leg", dq_r_knee);
    }
}

bool PlayerController::applyFootIk(
    const std::shared_ptr<DrawableObject>& player) {
    if (!foot_ik_enabled_ || !ground_query_) return false;
    if (player->findNodeIndexByName("left_upper_leg")  < 0 &&
        player->findNodeIndexByName("right_upper_leg") < 0) {
        return false;
    }

    // Snapshot the initialized (bind) leg pose ONCE — the stateless solver
    // needs its frozen bone lengths + reference axes.  Retry until the rig's
    // leg world matrices are actually populated (a valid capture).
    if (!leg_bind_captured_) {
        captureLegBind(player);
        if (leg_bind_[0].valid || leg_bind_[1].valid) leg_bind_captured_ = true;
        else return false;
    }

    // Direction of travel (world XZ) for the walking step offsets.
    glm::vec3 fwd(gait_fwd_xz_.x, 0.0f, gait_fwd_xz_.y);
    if (glm::length(fwd) < 1e-4f) fwd = glm::vec3(0.0f, 0.0f, 1.0f);
    fwd = glm::normalize(fwd);
    const glm::vec3 up(0.0f, 1.0f, 0.0f);

    struct Leg {
        int idx; const char* hip; const char* knee; const char* foot;
        float phase;
    };
    const Leg legs[2] = {
        { 0, "left_upper_leg",  "left_lower_leg",  "left_foot",
          anim_phase_ },
        { 1, "right_upper_leg", "right_lower_leg", "right_foot",
          anim_phase_ + 3.14159265358979f },
    };

    float worst_deficit = 0.0f;
    bool  any = false;
    glm::vec3 dbg_hip(0.0f), dbg_foot(0.0f), dbg_target(0.0f);
    float dbg_gy = 0.0f, dbg_def = 0.0f; bool dbg_have = false;

    for (const Leg& L : legs) {
        if (!leg_bind_[L.idx].valid) continue;
        const glm::vec3 hip_w(player->getNodeWorldMatrixByName(L.hip)[3]);

        // Gait: while walking the foot steps along travel + lifts mid-swing;
        // standing keeps both feet grounded.  Left/right 180 deg out of phase.
        float step = 0.0f, lift = 0.0f;
        if (walking_) {
            step = -std::cos(L.phase) * foot_stride_amp_;
            const float sn = std::sin(L.phase);
            lift = (sn > 0.0f ? sn : 0.0f) * foot_lift_amp_;
        }
        // Foot rests under its own hip, displaced along travel by the step.
        const float gx = hip_w.x + fwd.x * step;
        const float gz = hip_w.z + fwd.z * step;

        float gy = hip_w.y;
        glm::vec3 gn(0.0f, 1.0f, 0.0f);
        // Probe ~0.9 m below the hip so an upper storey above is ignored.
        if (!ground_query_(gx, gz, hip_w.y - 0.9f, gy, gn)) continue;
        if (glm::length(gn) < 0.1f) gn = up;

        // Anti-collapse guard: a bad ground probe (furniture, a mis-classified
        // surface) must never put the target up near the hip and fold the leg
        // into a heap -- keep the foot target at least 0.35 m below the hip.
        const float foot_target_y =
            std::min(gy + foot_sole_drop_ + lift, hip_w.y - 0.35f);
        const glm::vec3 target(gx, foot_target_y, gz);

        // Only tilt a PLANTED foot to the surface; fade tilt out while lifted.
        float plant = 1.0f;
        if (walking_ && foot_lift_amp_ > 1e-4f) {
            plant = 1.0f - glm::clamp(lift / foot_lift_amp_, 0.0f, 1.0f);
        }

        float deficit = 0.0f;
        solveLegIk(player, L.idx, L.hip, L.knee, L.foot, target, gn,
                   foot_tilt_weight_ * plant, deficit);
        if (deficit > worst_deficit) worst_deficit = deficit;
        any = true;

        if (L.idx == 0) {
            dbg_hip = hip_w;
            dbg_foot = glm::vec3(player->getNodeWorldMatrixByName(L.foot)[3]);
            dbg_gy = gy; dbg_target = target; dbg_def = deficit;
            dbg_have = true;
        }
    }

    // Pelvis drop: when a foot couldn't reach its ground target the body sits
    // too high -> lower the root (next frame) by the worst deficit, smoothed
    // and clamped.  ~0 in normal terrain now that the solve is exact.
    const float target_drop =
        glm::clamp(worst_deficit, 0.0f, pelvis_drop_max_);
    pelvis_drop_ += (target_drop - pelvis_drop_) * 0.2f;
    if (pelvis_drop_ < 1e-4f) pelvis_drop_ = 0.0f;

    // Once/sec ground-truth dump (left leg).  If target.y ~= hip.y the ground
    // probe is feeding a near-hip surface (guard then clamps it); otherwise
    // the foot should sit right on target.y.
    if (dbg_have) {
        static unsigned s_ik_log = 0;
        if ((s_ik_log++ % 60u) == 0u) {
            std::printf(
                "[foot_ik] hip=(%.2f,%.2f,%.2f) foot=(%.2f,%.2f,%.2f) "
                "gy=%.2f target=(%.2f,%.2f,%.2f) L1=%.3f L2=%.3f "
                "def=%.3f drop=%.3f walk=%d\n",
                dbg_hip.x, dbg_hip.y, dbg_hip.z,
                dbg_foot.x, dbg_foot.y, dbg_foot.z, dbg_gy,
                dbg_target.x, dbg_target.y, dbg_target.z,
                leg_bind_[0].L1, leg_bind_[0].L2, dbg_def, pelvis_drop_,
                walking_ ? 1 : 0);
        }
    }
    return any;
}

void PlayerController::captureLegBind(
    const std::shared_ptr<DrawableObject>& player) {
    struct LegN { int idx; const char* hip; const char* knee; const char* foot; };
    const LegN legs[2] = {
        { 0, "left_upper_leg",  "left_lower_leg",  "left_foot"  },
        { 1, "right_upper_leg", "right_lower_leg", "right_foot" },
    };
    for (const LegN& L : legs) {
        leg_bind_[L.idx].valid = false;
        if (player->findNodeIndexByName(L.hip)  < 0 ||
            player->findNodeIndexByName(L.knee) < 0 ||
            player->findNodeIndexByName(L.foot) < 0) continue;

        const glm::mat4 H = player->getNodeWorldMatrixByName(L.hip);
        const glm::mat4 K = player->getNodeWorldMatrixByName(L.knee);
        const glm::mat4 F = player->getNodeWorldMatrixByName(L.foot);
        const glm::vec3 a0(H[3]), b0(K[3]), c0(F[3]);
        const float L1 = glm::length(b0 - a0);
        const float L2 = glm::length(c0 - b0);
        if (L1 < 1e-4f || L2 < 1e-4f) continue;   // chain not populated yet

        const glm::quat Rhip0  = matRotation(H);
        const glm::quat Rknee0 = matRotation(K);
        const glm::vec3 u0 = (b0 - a0) / L1;       // bind upper-leg dir (world)
        const glm::vec3 v0 = (c0 - b0) / L2;       // bind lower-leg dir (world)

        LegIkBind& B = leg_bind_[L.idx];
        B.L1  = L1; B.L2 = L2;
        B.qh0 = player->getNodeRotationByName(L.hip);
        B.qk0 = player->getNodeRotationByName(L.knee);
        B.qf0 = player->getNodeRotationByName(L.foot);
        B.ahx = glm::inverse(Rhip0)  * u0;          // bone axis, hip-local
        B.akx = glm::inverse(Rknee0) * v0;          // bone axis, knee-local

        glm::vec3 bendW0 = glm::cross(u0, v0);      // bind knee hinge (world)
        if (glm::length(bendW0) < 1e-4f) {          // near-straight bind leg
            bendW0 = glm::cross(u0, glm::vec3(0.0f, 0.0f, 1.0f));
            if (glm::length(bendW0) < 1e-4f)
                bendW0 = glm::cross(u0, glm::vec3(1.0f, 0.0f, 0.0f));
        }
        bendW0 = glm::normalize(bendW0);
        const glm::quat RhipParent0 = Rhip0 * glm::inverse(B.qh0);
        B.bend_hp = glm::inverse(RhipParent0) * bendW0;  // hinge, hip-parent
        B.valid = true;
    }
}

void PlayerController::solveLegIk(
    const std::shared_ptr<DrawableObject>& player,
    int leg_idx, const char* hip_name, const char* knee_name,
    const char* foot_name, const glm::vec3& target,
    const glm::vec3& ground_normal, float foot_tilt_w, float& out_deficit) {
    out_deficit = 0.0f;
    const LegIkBind& B = leg_bind_[leg_idx];
    if (!B.valid) return;

    // Hip world position + parent rotation.  Neither depends on this leg's
    // own IK, so reading the 1-frame-old cached matrix is safe -- the solve
    // cannot feed back on itself, which is what made the old approach drift
    // into a folded pose.
    const glm::mat4 H = player->getNodeWorldMatrixByName(hip_name);
    const glm::vec3 a(H[3]);
    const glm::quat RhipW       = matRotation(H);
    const glm::quat qh_cur      = player->getNodeRotationByName(hip_name);
    const glm::quat RhipParentW = RhipW * glm::inverse(qh_cur);

    glm::vec3 d3 = target - a;
    float dist = glm::length(d3);
    if (dist < 1e-5f) return;                       // target on the hip
    const glm::vec3 dir = d3 / dist;

    const float eps = 0.002f;
    const float hi = B.L1 + B.L2 - eps;
    const float lo = std::fabs(B.L1 - B.L2) + eps;
    if (dist > hi) { out_deficit = dist - hi; dist = hi; }
    else if (dist < lo) { dist = lo; }

    // Law of cosines: angle at the hip between (hip->target) and upper leg.
    const float cosA =
        (B.L1 * B.L1 + dist * dist - B.L2 * B.L2) / (2.0f * B.L1 * dist);
    const float alpha = std::acos(glm::clamp(cosA, -1.0f, 1.0f));

    // Knee hinge axis follows the body (bind axis rotated by the live
    // hip-parent), forced perpendicular to the reach direction.
    glm::vec3 bendW = RhipParentW * B.bend_hp;
    bendW = bendW - dir * glm::dot(bendW, dir);
    if (glm::length(bendW) < 1e-5f) {
        bendW = glm::cross(dir, glm::vec3(0.0f, 0.0f, 1.0f));
        if (glm::length(bendW) < 1e-5f)
            bendW = glm::cross(dir, glm::vec3(1.0f, 0.0f, 0.0f));
    }
    bendW = glm::normalize(bendW);

    // Analytic knee + foot placement (exact, single pass).
    const glm::vec3 u1    = glm::angleAxis(alpha, bendW) * dir;
    const glm::vec3 knee  = a + u1 * B.L1;
    const glm::vec3 footP = a + dir * dist;
    glm::vec3 v1 = footP - knee;
    const float v1len = glm::length(v1);
    if (v1len < 1e-6f) return;
    v1 /= v1len;

    // Hip: swing the bind upper-leg direction onto u1, written as LOCAL rot.
    const glm::quat RhipW_bind = RhipParentW * B.qh0;
    const glm::vec3 u_bind_now = RhipW_bind * B.ahx;
    const glm::quat RhipW_new  = fromTo(u_bind_now, u1) * RhipW_bind;
    player->setNodeRotationByName(
        hip_name, glm::normalize(glm::inverse(RhipParentW) * RhipW_new));

    // Knee: swing the bind lower-leg direction onto v1 under the NEW hip.
    const glm::quat RkneeW_bind = RhipW_new * B.qk0;
    const glm::vec3 v_bind_now  = RkneeW_bind * B.akx;
    const glm::quat RkneeW_new  = fromTo(v_bind_now, v1) * RkneeW_bind;
    player->setNodeRotationByName(
        knee_name, glm::normalize(glm::inverse(RhipW_new) * RkneeW_new));

    // Foot: bind orientation under the new knee, then tilt toward the ground
    // normal (self-correcting, clamped).  Uses the freshly computed knee
    // world rotation, so no stale cached read is involved.
    glm::quat RfootW = RkneeW_new * B.qf0;
    if (foot_tilt_w > 1e-3f) {
        const glm::vec3 cur_up = RfootW * glm::vec3(0.0f, 1.0f, 0.0f);
        glm::vec3 ax; float ang;
        if (axisAngleBetween(cur_up, ground_normal, ax, ang)) {
            ang = glm::clamp(ang * foot_tilt_w, 0.0f, glm::radians(30.0f));
            if (ang > 1e-4f) RfootW = glm::angleAxis(ang, ax) * RfootW;
        }
    }
    player->setNodeRotationByName(
        foot_name, glm::normalize(glm::inverse(RkneeW_new) * RfootW));
}

} // namespace game_object
} // namespace engine
