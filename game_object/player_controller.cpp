#include "player_controller.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
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

// Finite-value guards.  A single NaN/Inf that reaches a bone rotation
// propagates through the cached world matrices and is SELF-SUSTAINING
// (the IK reads the NaN hip next frame and writes NaN again), which
// collapses the legs to nothing.  These let the solver detect a bad
// value and fall back to the bind pose so the rig climbs back out.
inline bool isFiniteV3(const glm::vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
inline bool isFiniteQ(const glm::quat& q) {
    return std::isfinite(q.x) && std::isfinite(q.y) &&
           std::isfinite(q.z) && std::isfinite(q.w);
}

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

    // Editor "edit mode" gate: when movement is disabled, hide the input
    // window from the WASD/SPACE *movement* reads (mwin) so the character
    // stays put while the free camera consumes the keys.  Non-movement key
    // reads (debug toggles) keep using the real `window`.
    GLFWwindow* mwin = movement_enabled_ ? window : nullptr;

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
    //   • walking_ is driven by the WASD move keys (read from `window`
    //     below).  applyPose() / the foot IK read it (via walk_weight_)
    //     to gate the arm / leg / knee swing AND the stepping gait; idle
    //     sway plays in both cases.
    // camera_yaw / obstacles / world remain unused for now (the app owns
    // translation); `window` IS read below for the WASD walk intent.
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

    // ── WASD drives the walk ──────────────────────────────────────────
    // Read the movement keys directly instead of inferring "walking" from
    // the position delta.  The camera owns translation (the app's follow
    // block places the body relative to the camera), but inferring the gait
    // from motion meant MOUSE-LOOK — which orbits the body and produces a
    // position delta — falsely triggered the walk and snapped the feet.
    // Gating on the actual move keys means only WASD engages the walk that
    // drives the lower-body foot IK; gait_fwd_xz_ (above) still supplies the
    // real travel direction for foot-step placement.  walk_weight_ below
    // still ramps the gait in/out so press/release eases rather than snaps.
    bool wasd_held = false;
    if (mwin) {
        wasd_held =
            glfwGetKey(mwin, GLFW_KEY_W) == GLFW_PRESS ||
            glfwGetKey(mwin, GLFW_KEY_A) == GLFW_PRESS ||
            glfwGetKey(mwin, GLFW_KEY_S) == GLFW_PRESS ||
            glfwGetKey(mwin, GLFW_KEY_D) == GLFW_PRESS;
    }
    // force_walking_ is set by the Render Debug skeleton-view modes so
    // the user can watch the legs animate without holding WASD; in normal
    // play it stays false and the walk is purely WASD-driven.
    walking_ = wasd_held || force_walking_;

    // Smoothly ramp the gait blend toward the (now stable) walking_
    // state so the foot step/lift ease in and out instead of snapping.
    const float walk_target = walking_ ? 1.0f : 0.0f;
    constexpr float kWalkBlendRate = 8.0f;  // per second
    walk_weight_ +=
        (walk_target - walk_weight_) *
        std::min(1.0f, kWalkBlendRate * std::max(delta_t, 0.0f));

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
        constexpr float kMinWalkRad     = 3.5f;        // floor cadence so the
                                                       // feet still step when
                                                       // WASD translation is
                                                       // small or blocked
        const float swing_rate = glm::clamp(
            speed_mps * kSwingRadPerMps, kMinWalkRad, kMaxSwingRad);
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

    // ── Discrete step-animation input ────────────────────────────────
    // 'B' toggles the step mode itself (so we can A/B against the
    //     existing procedural swing + foot IK at runtime).
    // SPACE = take one step: the NEXT leg in the L,R,L,R alternation
    //     swings forward at the hip while the OTHER leg swings back,
    //     AND the body advances by step_length_m_ along facing dir.
    //     The position write in setPositionAndYaw is dropped while
    //     step mode is on, so this translation isn't immediately
    //     clobbered by the application's camera-follow block.
    // Both keys are edge-detected -- holding either fires once per
    // physical press, not every frame.
    if (window) {
        const bool toggle_down =
            (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS);
        if (toggle_down && !step_toggle_key_down_prev_) {
            step_anim_enabled_ = !step_anim_enabled_;
            // Force a re-seed of the translation target next time step
            // mode comes back on -- otherwise an old target_pos_ from a
            // previous session would yank the body across the map on
            // the first press.
            if (!step_anim_enabled_) step_target_initialized_ = false;
            std::cout << "[step_anim] enabled="
                      << (step_anim_enabled_ ? 1 : 0) << std::endl;
        }
        step_toggle_key_down_prev_ = toggle_down;

        if (step_anim_enabled_) {
            // Seed body target + per-foot world positions on the first
            // frame of step mode (or after a re-enable toggle).  Both
            // feet start under their hips, body target = current
            // position, so the very first press has somewhere to walk
            // FROM that matches the visible bind pose.
            if (!step_target_initialized_) {
                const float yaw_rad = glm::radians(yaw_deg_);
                const glm::vec3 fwd_world(
                    std::cos(yaw_rad), 0.0f, -std::sin(yaw_rad));
                // right = +90° clockwise rotation of fwd around +Y.
                // Verified handed-ness: fwd x right = +Y (Y-up RH).
                const glm::vec3 right_world(
                    -std::sin(yaw_rad), 0.0f, -std::cos(yaw_rad));
                step_target_pos_       = position_;
                // Stagger the feet fore-aft on entry: L back by
                // step_length/2, R ahead by step_length/2.  With both feet
                // at body position (the old init), the very first step
                // only swept step_length forward and the SECOND step
                // swept 1.5*step_length (because the trailing foot had
                // both lagged the body and now had to catch up).  After
                // that, steady state was only step_length per swing -- so
                // the user saw "first big, then tight" as steps 2 → 3
                // shrank from 1.2 m → 0.8 m.  Pre-staggering means step 1
                // already starts from steady-state geometry and every
                // step covers the same world-space arc.
                const float half_stride = step_length_m_ * 0.5f;
                foot_world_xz_[0]      = position_
                                         - right_world * hip_lateral_m_
                                         - fwd_world   * half_stride; // L back
                foot_world_xz_[1]      = position_
                                         + right_world * hip_lateral_m_
                                         + fwd_world   * half_stride; // R ahead
                foot_target_world_xz_[0] = foot_world_xz_[0];
                foot_target_world_xz_[1] = foot_world_xz_[1];
                step_target_initialized_ = true;
            }

            // ── Static test pose short-circuit ──────────────────────
            // When test_pose_enabled_ is true, pin foot targets to a
            // fixed body-relative stride (L 0.5 m ahead, R 0.5 m back
            // by default) and skip the SPACE / auto-step handler.  The
            // body's translation target is held at position_ so the
            // character stands in place while we observe whether the
            // legs actually deform to the requested foot positions.
            // This is the visual smoke test for the bone-rotation
            // propagation bug -- if the rig assumes a wide stance,
            // the IK pipe is alive end-to-end; if it stays in T-pose
            // despite [step_anim.pose] showing big thetaL/thetaR
            // numbers, the engine-side matrix composition needs the
            // fix.
            if (test_pose_enabled_) {
                const float yaw_rad = glm::radians(yaw_deg_);
                const glm::vec3 fwd_world(
                    std::cos(yaw_rad), 0.0f, -std::sin(yaw_rad));
                const glm::vec3 right_world(
                    -std::sin(yaw_rad), 0.0f, -std::cos(yaw_rad));
                foot_target_world_xz_[0] =
                    position_
                    + fwd_world   * test_pose_L_fwd_m_
                    - right_world * hip_lateral_m_;
                foot_target_world_xz_[1] =
                    position_
                    + fwd_world   * test_pose_R_fwd_m_
                    + right_world * hip_lateral_m_;
                step_target_pos_    = position_;        // no walk
                auto_step_timer_    = 0.0f;
                step_key_down_prev_ = false;
                // Done with input handling for this frame.
                static unsigned s_test_log = 0;
                if ((s_test_log++ % 60u) == 0u) {
                    std::cout << "[step_anim.test_pose]"
                              << " L_fwd=" << test_pose_L_fwd_m_
                              << "m R_fwd=" << test_pose_R_fwd_m_
                              << "m foot_tgt_L=("
                              << foot_target_world_xz_[0].x << ","
                              << foot_target_world_xz_[0].z
                              << ") foot_tgt_R=("
                              << foot_target_world_xz_[1].x << ","
                              << foot_target_world_xz_[1].z
                              << ") yaw=" << yaw_deg_
                              << std::endl;
                }
            } else {
            // ── Single step trigger (shared by SPACE and auto-step) ─
            // Captures the "advance body by half a step, move ONLY
            // the stepping foot's target a full step forward" logic
            // so both the manual SPACE press and the auto-step timer
            // call the same code path.  fireStep() also handles the
            // alternation (step_next_leg_) and prints the diagnostic
            // log line.
            auto fireStep = [&](const char* source) {
                const int stepping = step_next_leg_;   // 0=L first
                const int other    = 1 - stepping;
                const float yaw_rad = glm::radians(yaw_deg_);
                const glm::vec3 fwd_world(
                    std::cos(yaw_rad), 0.0f, -std::sin(yaw_rad));
                const glm::vec3 right_world(
                    -std::sin(yaw_rad), 0.0f, -std::cos(yaw_rad));
                const float side_sign = (stepping == 0) ? -1.0f : +1.0f;

                // ── WASD-derived movement direction ────────────────
                // W/S = fore/aft along body facing.  A/D = strafe
                // perpendicular.  SPACE forces a forward step so
                // manual single-step testing still works without
                // holding WASD.  If no WASD held (and not SPACE),
                // skip the body advance entirely so the character
                // stops walking the moment all movement keys are
                // released.
                glm::vec3 step_dir = fwd_world;
                if (std::string(source) != std::string("space")) {
                    const bool w_pressed = mwin &&
                        (glfwGetKey(mwin, GLFW_KEY_W) == GLFW_PRESS);
                    const bool s_pressed = mwin &&
                        (glfwGetKey(mwin, GLFW_KEY_S) == GLFW_PRESS);
                    const bool a_pressed = mwin &&
                        (glfwGetKey(mwin, GLFW_KEY_A) == GLFW_PRESS);
                    const bool d_pressed = mwin &&
                        (glfwGetKey(mwin, GLFW_KEY_D) == GLFW_PRESS);
                    const float wf = (w_pressed ? 1.0f : 0.0f)
                                   - (s_pressed ? 1.0f : 0.0f);
                    const float wr = (d_pressed ? 1.0f : 0.0f)
                                   - (a_pressed ? 1.0f : 0.0f);
                    step_dir = fwd_world * wf + right_world * wr;
                    const float len = glm::length(step_dir);
                    if (len > 1e-3f) {
                        step_dir /= len;
                    } else {
                        return; // no WASD -> no step
                    }
                }

                // Body advances a FULL step_length per press.  Combined
                // with the foot landing step_length/2 ahead of the new
                // body position (below), each cycle of two steps
                // advances each foot by 2*step_length in world space --
                // 1.5 m of foot sweep per swing at step_length=0.8 m,
                // which reads as a real stride rather than a shuffle.
                // The earlier 0.5*step_length advance was the source of
                // "tight" follow-up steps: it made the steady-state per-
                // foot sweep equal step_length, half of what step 2 used
                // to look like (because step 2's trailing foot started
                // at body position from the init).
                step_target_pos_ += step_dir * step_length_m_;

                // Stepping foot target lands half a step AHEAD of
                // where the body will be after the slide.  The other
                // foot's target is left untouched -- it stays
                // anchored at its previous world XZ.
                foot_target_world_xz_[stepping] =
                    step_target_pos_
                    + right_world * (hip_lateral_m_ * side_sign)
                    + step_dir    * (step_length_m_ * 0.5f);

                step_next_leg_ = other;

                std::cout << "[step_anim] step leg="
                          << (stepping == 0 ? "LEFT" : "RIGHT")
                          << " src=" << source
                          << " body_tgt=(" << step_target_pos_.x
                          << "," << step_target_pos_.z
                          << ") foot_tgt_L=(" << foot_target_world_xz_[0].x
                          << "," << foot_target_world_xz_[0].z
                          << ") foot_tgt_R=(" << foot_target_world_xz_[1].x
                          << "," << foot_target_world_xz_[1].z
                          << ") yaw=" << yaw_deg_ << std::endl;
            };

            const bool space_down =
                mwin && (glfwGetKey(mwin, GLFW_KEY_SPACE) == GLFW_PRESS);
            if (space_down && !step_key_down_prev_) {
                fireStep("space");
                // Manual press resets the auto timer so a tap doesn't
                // get followed by an immediate auto-fire.
                auto_step_timer_ = 0.0f;
            }
            step_key_down_prev_ = space_down;

            // ── Auto-step timer ─────────────────────────────────────
            // Accumulates delta_t each frame; when it crosses the
            // interval, fires a step automatically and wraps.  Default
            // is 5 s/step (configurable via setAutoStepIntervalSec).
            if (auto_step_enabled_ && wasd_held) {
                auto_step_timer_ += std::max(0.0f, delta_t);
                if (auto_step_timer_ >= auto_step_interval_s_) {
                    fireStep("auto");
                    auto_step_timer_ = 0.0f;
                }
            }

            // Catch up: the BACK foot slides forward to align with
            // the FRONT foot's fwd projection.  Front foot stays
            // put.  Body holds its current XZ.  After lerp settle:
            // both feet at the same forward position, character
            // standing at attention.  step_angle_deg_ derived per
            // frame from foot-vs-hip projection -> trends to 0 ->
            // arms / spine untwist automatically.
            if (!wasd_held && step_target_initialized_) {
                const float yaw_rad = glm::radians(yaw_deg_);
                const glm::vec3 fwd_world(
                    std::cos(yaw_rad), 0.0f, -std::sin(yaw_rad));
                const glm::vec3 right_world(
                    -std::sin(yaw_rad), 0.0f, -std::cos(yaw_rad));
                const glm::vec3 dL = foot_world_xz_[0] - position_;
                const glm::vec3 dR = foot_world_xz_[1] - position_;
                const float fwdL = dL.x*fwd_world.x + dL.z*fwd_world.z;
                const float fwdR = dR.x*fwd_world.x + dR.z*fwd_world.z;
                const int front_i = (fwdL >= fwdR) ? 0 : 1;
                const int back_i  = 1 - front_i;
                const float front_fwd = (front_i == 0) ? fwdL : fwdR;
                const float back_lat  = (back_i == 0) ? -1.0f : +1.0f;
                foot_target_world_xz_[back_i] =
                    position_
                    + fwd_world   * front_fwd
                    + right_world * (hip_lateral_m_ * back_lat);
                foot_target_world_xz_[front_i] = foot_world_xz_[front_i];
                // Body advances forward by front_fwd so it lands over
                // the feet -- without this the feet end up AHEAD of
                // the hips, derived step_angle_deg_ stays positive
                // for both legs, and the character leans forward.
                step_target_pos_ = position_ + fwd_world * front_fwd;
            }

            } // end else (test_pose_enabled_ == false branch)
        } else {
            step_key_down_prev_ = false;
            auto_step_timer_    = 0.0f;
        }
    }

    // ── Slide body + the stepping foot toward their targets ─────────
    // Same lerp rate for body and feet so the slide and the foot
    // plant complete together.  Rate 1.5/s reaches ~95% in ~2 s,
    // which on the 5 s auto-step interval keeps the body and the
    // swinging foot in continuous motion the entire time instead of
    // settling in 0.4 s and then sitting idle for 4.6 s -- the latter
    // is what the user saw as "teleport".  The PLANTED foot's
    // target == current value, so its per-frame lerp is a no-op and
    // it stays anchored in world space exactly as before.
    constexpr float kStepLerpRate = 4.0f;  // bumped from 1.5: stand-down was settling too slowly so the body looked leaned forward
    const float step_t =
        std::min(1.0f, kStepLerpRate * std::max(delta_t, 0.0f));

    if (step_anim_enabled_ && step_target_initialized_) {
        // Body XZ (Y left to foot-IK / spawn / external).
        position_.x += (step_target_pos_.x - position_.x) * step_t;
        position_.z += (step_target_pos_.z - position_.z) * step_t;

        // Each foot toward its own target.
        for (int i = 0; i < 2; ++i) {
            foot_world_xz_[i].x +=
                (foot_target_world_xz_[i].x - foot_world_xz_[i].x) * step_t;
            foot_world_xz_[i].z +=
                (foot_target_world_xz_[i].z - foot_world_xz_[i].z) * step_t;
        }

        // ── Derive each leg's hip-swing angle from foot vs hip ──────
        // Project (foot - hip) onto the body's forward direction; the
        // sine of the leg's swing angle equals (forward amount) /
        // leg_length.  With this in place the PLANTED foot stays put
        // in world space, and as the body slides past it the planted
        // leg's angle goes negative (foot behind hip) -- which is
        // exactly the half of walking that makes a step read as
        // "one foot moving" instead of a body teleport.
        const float yaw_rad = glm::radians(yaw_deg_);
        const glm::vec3 fwd_world(
            std::cos(yaw_rad), 0.0f, -std::sin(yaw_rad));
        const glm::vec3 right_world(
            -std::sin(yaw_rad), 0.0f, -std::cos(yaw_rad));
        for (int i = 0; i < 2; ++i) {
            const float lat_sign = (i == 0) ? -1.0f : +1.0f;
            const glm::vec3 hip_world =
                position_ + right_world * (hip_lateral_m_ * lat_sign);
            const glm::vec3 diff = foot_world_xz_[i] - hip_world;
            const float fwd_amount =
                diff.x * fwd_world.x + diff.z * fwd_world.z;
            // Fall back to a sensible leg length if the bind capture
            // hasn't run yet (rig not ready) so the angle is bounded.
            const float leg_len =
                (leg_bind_[i].valid && leg_bind_[i].L1 + leg_bind_[i].L2 > 0.1f)
                ? (leg_bind_[i].L1 + leg_bind_[i].L2)
                : 0.9f;
            const float sin_theta =
                glm::clamp(fwd_amount / leg_len, -0.95f, 0.95f);
            step_angle_deg_[i] = glm::degrees(std::asin(sin_theta));
        }

        // ── Per-second live diagnostic ──────────────────────────────
        // Prints current foot world positions + derived leg angles so
        // we can confirm from the log whether the angles ARE
        // oscillating between presses (vs the leg being completely
        // pinned to bind pose by some downstream bug).  This together
        // with [step_anim.pose] (printed from applyPose below) is the
        // pair to compare when the question is "the leg looks still --
        // is the IK actually outputting a swing?".
        static unsigned s_live_log = 0;
        if ((s_live_log++ % 60u) == 0u) {
            std::cout << "[step_anim.live]"
                      << " body=(" << position_.x << "," << position_.z << ")"
                      << " footL=(" << foot_world_xz_[0].x << ","
                                    << foot_world_xz_[0].z << ")"
                      << " footR=(" << foot_world_xz_[1].x << ","
                                    << foot_world_xz_[1].z << ")"
                      << " thetaL=" << step_angle_deg_[0] << "deg"
                      << " thetaR=" << step_angle_deg_[1] << "deg"
                      << " stride=" << step_length_m_
                      << std::endl;
        }
    }


    // ── Ground-touching IK ──────────────────────────────────────────
    // Probe the ground under the body's current XZ each frame and
    // snap position_.y so the BACK foot (the one swept furthest from
    // vertical) stays planted.  Without this the body's Y stays at
    // whatever spawn put it, and as the legs swing the feet lift off
    // the ground.  With it: the planted leg's foot is exactly on the
    // surface, the swinging foot arcs slightly above ground (natural
    // bob), and the body rises/falls with terrain XZ.
    //   • ground_query_ is the app-supplied callback (queryGroundAt
    //     in application.cpp).  No-op if not wired.
    //   • kRootToFoot (1.10 m) is the rig's root-to-foot bind
    //     distance for scene-skinned.gltf.  Tune if the character
    //     hovers or sinks at theta=0.
    //   • The lift correction uses |theta_back| of the more-swept
    //     leg: foot Y lift = leg_len * (1 - cos(theta)).  Subtract
    //     that from the desired pelvis height so the back foot ends
    //     up exactly on ground level.
    if (step_anim_enabled_ && ground_query_) {
        constexpr float kRootToFoot      = 1.10f;
        constexpr float kLegLen          = 0.95f;
        constexpr float kFootSoleDrop    = 0.12f;  // +10cm: feet were sinking into the ground
        float gy = 0.0f;
        glm::vec3 gn(0.0f, 1.0f, 0.0f);
        if (ground_query_(position_.x, position_.z,
                          position_.y - kRootToFoot, gy, gn)) {
            const float max_abs_theta = std::max(
                std::fabs(step_angle_deg_[0]),
                std::fabs(step_angle_deg_[1]));
            const float lift = kLegLen *
                (1.0f - std::cos(glm::radians(max_abs_theta)));
            const float target_root_y =
                gy + kRootToFoot - lift + kFootSoleDrop;
            // Smooth the Y so terrain bumps don't jitter the body --
            // ~6/s lerp settles in ~0.5 s, fast enough that walking
            // up steps still tracks reasonably.
            constexpr float kGroundLerpRate = 6.0f;
            const float t = std::min(1.0f,
                kGroundLerpRate * std::max(delta_t, 0.0f));
            position_.y += (target_root_y - position_.y) * t;

            static unsigned s_ground_log = 0;
            if ((s_ground_log++ % 60u) == 0u) {
                std::cout << "[ground_ik]"
                          << " gy=" << gy
                          << " thetaMaxAbs=" << max_abs_theta
                          << " lift=" << lift
                          << " target_root_y=" << target_root_y
                          << " body_y=" << position_.y
                          << std::endl;
            }
        }
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
    // Guard: a NaN pelvis_drop_ (from a degenerate IK frame) would poison
    // the root translation and NaN-propagate down the WHOLE skeleton --
    // the legs are parented to the root, so they vanish.  Clamp it back to
    // 0 if it ever goes non-finite so the root stays valid and the rig can
    // recover the next frame.
    if (!std::isfinite(pelvis_drop_)) pelvis_drop_ = 0.0f;
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
        // Root rotation present at the moment of bind capture.  Used
        // below to derive each bone's BODY-relative bind rotation
        // (world bind divided by the root yaw at this instant) so the
        // computed swing_axis_local stays valid regardless of how the
        // player turns later.
        const glm::quat root_rot_at_capture =
            axisAngleDeg(glm::vec3(0.0f, 1.0f, 0.0f), yaw_deg_);
        const glm::quat root_rot_inv = glm::inverse(root_rot_at_capture);

        int total_existing = 0;
        int total_captured = 0;
        for (const char* nm : kBoneNames) {
            int idx = player->findNodeIndexByName(nm);
            if (idx < 0) continue; // bone missing on this rig — skip
            total_existing++;

            // ── Validate the world matrix first ─────────────────────
            // The bone's world matrix is updated by DrawableObject's
            // per-frame work, which may not have run yet on the first
            // frame the rig becomes "ready".  Degenerate (all-zero)
            // columns would collapse matRotation() to identity and
            // then root_rot_inv would falsely look like the bone's
            // body-frame rotation -- producing axis_local = world-
            // right-at-capture-yaw for every bone (exactly what the
            // diagnostic printed when this was broken).  Skip this
            // bone for this frame; we'll retry next frame.
            const glm::mat4 W = player->getNodeWorldMatrixByName(nm);
            const glm::vec3 c0(W[0]), c1(W[1]), c2(W[2]);
            if (glm::length(c0) < 1e-4f ||
                glm::length(c1) < 1e-4f ||
                glm::length(c2) < 1e-4f) {
                continue;  // world matrix not populated yet
            }

            BoneBindRot b;
            b.node_idx = idx;

            // Treat a degenerate (zero-length) returned quaternion as
            // identity.  glTF nodes with no explicit rotation field
            // come back as (0,0,0,0) from getNodeRotationByName on
            // this engine -- multiplying our swing dq by that wipes
            // it back to zero, which is the reason arms have been
            // looking frozen.  The visible "arm hangs at side" bind
            // pose is actually driven by the skin's inverse-bind
            // matrices (NOT by the bone's local rotation), so an
            // identity local rotation is the correct interpretation.
            const glm::quat raw = player->getNodeRotationByName(nm);
            const float ql = std::sqrt(
                raw.x * raw.x + raw.y * raw.y +
                raw.z * raw.z + raw.w * raw.w);
            b.rot = (ql > 0.5f)
                ? raw
                : glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // identity

            // Compute the swing axis (the bone-local axis post-bind
            // that, when rotated around, moves the bone in the BODY's
            // fore-aft plane).  Math: in the body's local frame, the
            // player's right axis is (0, 0, -1) (matches right_world
            // formula above when yaw=0).  We want to express that
            // axis in the bone's post-bind LOCAL frame:
            //   axis_local = R_arm_body^-1 * (0, 0, -1)
            //   R_arm_body = root_at_capture^-1 * R_arm_world_bind
            // The world matrix is now known good (checked above), so
            // matRotation gives the real bone-bind world rotation
            // including any clavicle/shoulder chain rotation, and the
            // returned axis is in the bone's own post-bind local
            // frame regardless of how the rig nests its bones.
            const glm::quat R_world = matRotation(W);
            const glm::quat R_body  = root_rot_inv * R_world;
            const glm::vec3 axis    =
                glm::inverse(R_body) * glm::vec3(0.0f, 0.0f, -1.0f);
            const float al = glm::length(axis);
            if (al > 1e-4f) {
                b.swing_axis_local = axis / al;
                b.swing_axis_valid = true;
            }

            bind_rots_[nm] = b;
            total_captured++;
        }
        // Mark captured only when EVERY existing bone has captured
        // valid bind data.  Anything less leaves some bones behind
        // (their world matrix wasn't populated this frame) -- and
        // those bones would then sit unposed forever because the flag
        // stops the retry loop.  When total_captured < total_existing
        // we fall through and try again next frame.
        if (total_existing > 0 && total_captured == total_existing) {
            bind_rots_captured_ = true;
            // One-shot diagnostic so we can verify the swing axes
            // landed in something sensible (e.g. unit length, not all
            // pointing along the bone's length).  If the printout
            // shows the arm axes pointing along Y (the bone length),
            // the rig's parent chain has a rotation we haven't
            // accounted for and the math here needs adjusting.
            auto pa = [&](const char* nm){
                auto it = bind_rots_.find(nm);
                if (it == bind_rots_.end()) {
                    std::cout << "  " << nm << ": MISSING" << std::endl;
                    return;
                }
                const auto& b = it->second;
                std::cout << "  " << nm
                          << " bind=(" << b.rot.x << "," << b.rot.y
                          << "," << b.rot.z << "," << b.rot.w << ")"
                          << " swing_axis_local=(" << b.swing_axis_local.x
                          << "," << b.swing_axis_local.y
                          << "," << b.swing_axis_local.z << ")"
                          << " valid=" << b.swing_axis_valid
                          << std::endl;
            };
            std::cout << "[bind_capture] yaw_at_capture=" << yaw_deg_
                      << std::endl;
            pa("left_upper_arm");
            pa("right_upper_arm");
            pa("left_upper_leg");
            pa("right_upper_leg");
        }
    }

    // ── DIAGNOSTIC: render bind pose only ────────────────────────────
    // Temporarily skip ALL procedural deformation (arm/leg swing, foot
    // IK, step-anim, spine sway) and just write every captured bone's
    // bind rotation.  The mesh then renders exactly at its bind pose,
    // which (for the re-skinned rig) reproduces the source mesh.  Use
    // this to confirm the SKIN is correct in isolation: if the body is
    // intact and un-torn here but collapses once this flag is off, the
    // fault is in the animation/IK posing the bones, not the weights.
    // Set to false to restore normal procedural animation.
    constexpr bool kRenderBindPoseOnly = false;
    if (kRenderBindPoseOnly) {
        for (const auto& kv : bind_rots_) {
            player->setNodeRotationByName(kv.first, kv.second.rot);
        }
        pelvis_drop_ = 0.0f;
        return;
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

    // Swing axis in BONE-LOCAL frame = bone-local Z (0,0,1).
    //
    // Why Z, definitively (this rig: scene-skinned.gltf auto-rig):
    //   • Every bone node is posed by TRANSLATION only -- all bind
    //     rotations are identity -- so a bone's local frame is just the
    //     model frame (the yaw lives in the ROOT node, above the bones).
    //   • The shoulder bones are separated along Z (left z=-0.10,
    //     right z=+0.10), so model-Z is the LEFT/RIGHT (pitch) axis and
    //     model-X is FORWARD/BACK.  The arms hang down -Y, so a fore-aft
    //     walk swing is a rotation about Z.  Rotating about X swings the
    //     arm SIDEWAYS -- that was the old hardcoded (1,0,0) bug and is
    //     the splayed/over-stretched arm pose seen while walking.
    //   • The leg step-path already proved this empirically: its
    //     kSwingAxis is (0,0,1) with the note "X swung sideways; Z gives
    //     fore-aft".  Arms share the legs' hang-down geometry, so arms
    //     use the same Z axis.
    //
    // Do NOT use the per-bone captured swing_axis_local here: on an
    // all-identity-rotation rig that capture resolves to root_rot*right,
    // which double-counts the body yaw (it came out ~(0.81,0,0.58) --
    // mostly X = sideways), so it drives the arms out to the sides.
    // A constant bone-local Z is both correct and yaw-robust because the
    // yaw is applied by the root, not baked into the swing.
    const glm::vec3 swing_axis_local(0, 0, 1);

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

    // Legs: three paths, in priority order.
    //
    // 1) step_anim_enabled_ -- the discrete keyboard-driven step pose
    //    (this iteration).  Only upper_leg rotates at the hip; knee +
    //    foot stay at their BIND rotations so pelvis-knee-foot remain
    //    colinear (no knee bend yet).  Both applyFootIk and the
    //    procedural sin() swing are skipped to keep them from racing
    //    the step pose for ownership of the same bones.  We also zero
    //    pelvis_drop_ since foot IK isn't writing it.
    // 2) applyFootIk -- the existing two-bone foot-planted IK; runs
    //    when step mode is OFF and a ground-query is wired.
    // 3) Procedural swing -- fallback when both above are unavailable
    //    (no rig leg bones, no ground query, etc.).
    if (step_anim_enabled_) {
        // ── Simple direct-write leg rotation ───────────────────────
        // The brute-force test proved that writing a quaternion
        // directly to upper_leg.rotation_ DOES rotate the bone and
        // skin the mesh -- a 90° around bone-local X swung the leg
        // visibly to a horizontal pose.  So we drop the complex
        // solveLegIk frame-conversion math (which was producing
        // hipLocal values that didn't reach the visible mesh) and
        // just write `axisAngleDeg(swing_axis, step_angle_deg_[i])`
        // straight to each upper_leg bone.  step_angle_deg_ was
        // already computed in update() from foot-vs-hip projection,
        // so amplitude is correct (±~31° during the test pose).
        //
        // Axis choice: from the brute-force test we observed that
        // local-X rotates the leg toward body-right.  For a fore/aft
        // swing the right axis is bone-local Z.  Sign tuned so a
        // POSITIVE angle (foot ahead of hip) swings the leg FORWARD;
        // if it goes the wrong way, flip kSwingAxisSign.
        static const char* const hip_names[2]  = {
            "left_upper_leg", "right_upper_leg" };
        static const char* const knee_names[2] = {
            "left_lower_leg", "right_lower_leg" };
        static const char* const foot_names[2] = {
            "left_foot",      "right_foot" };
        const glm::vec3 kSwingAxis(0.0f, 0.0f, 1.0f);  // bone-local Z -- X swung sideways; Z should give fore-aft
        constexpr float kSwingAxisSign = +1.0f;        // flip to -1 if leg swings backward

        // Capture the bind rotations on the first frame we run so we
        // can compose `bind * delta` rather than overwriting bind
        // with a small delta (which would leave the bone at a random
        // resting pose).  bind_rots_ is the existing map populated
        // earlier in applyPose -- safe to read here.
        for (int i = 0; i < 2; ++i) {
            const float swing_deg = kSwingAxisSign * step_angle_deg_[i];
            const glm::quat dq = axisAngleDeg(kSwingAxis, swing_deg);
            auto bit = bind_rots_.find(hip_names[i]);
            if (bit != bind_rots_.end()) {
                player->setNodeRotationByName(
                    hip_names[i], dq); // direct write, no bind composition -- matches the successful brute force exactly
            } else {
                // Bind capture not ready yet -- write the delta only,
                // which at least produces SOME rotation rather than
                // freezing the bone.
                player->setNodeRotationByName(hip_names[i], dq);
            }
            // Knee + foot: keep at their bind rotations (knee straight,
            // foot at bind orientation).  Stiff peg leg for now; knee
            // bend can come once the swing reads right.
            auto kit = bind_rots_.find(knee_names[i]);
            if (kit != bind_rots_.end()) {
                player->setNodeRotationByName(
                    knee_names[i], kit->second.rot);
            }
            auto fit = bind_rots_.find(foot_names[i]);
            if (fit != bind_rots_.end()) {
                player->setNodeRotationByName(
                    foot_names[i], fit->second.rot);
            }
        }


        // ── Upper-body counter-motion (natural walk) ───────────────
        // Real walking has arms swinging OPPOSITE to the same-side
        // leg (left arm forward when LEFT leg is back), and the spine
        // counter-twisting the hips.  All amplitudes scale with the
        // leg swing so the upper body auto-syncs with the gait.
        // Direct write (no bind composition) -- same pattern the leg
        // rotation uses, proven to reach the GPU mesh for this rig.
        //
        //   kArmFactor   — fraction of opposite leg's swing applied
        //                  to each arm.  0.7 reads as a strong arm
        //                  pump; 0.4 reads as a calmer walk.
        //   kElbowFactor — small elbow bend that tightens at peak
        //                  forward arm swing.  Keep small (~0.4) so
        //                  arms don't fold dramatically.
        //   kSpineFactor — fraction of (left-right leg delta) applied
        //                  as a Y-axis twist on the spine.
        constexpr float kArmFactor   = 1.8f;  // larger arm swing (was 1.0; ~45deg peak vs ~25deg)
        constexpr float kElbowFactor = 0.35f; // slightly bigger elbow tuck to match (was 0.25)
        constexpr float kSpineFactor = 0.20f; // a little more counter-twist (was 0.15)
        const glm::vec3 kSpineAxis(0.0f, 1.0f, 0.0f);  // body vertical twist

        // Sign flip per arm.  Leg theta convention: positive = foot
        // ahead of body.  Real walking has arms swinging OPPOSITE to
        // same-side leg (left arm BACK when left leg is FORWARD).  The
        // (-) on each line below provides the opposition; if the arms
        // come out moving WITH their same-side leg, flip the sign.
        const float arm_L_deg = -kArmFactor * step_angle_deg_[0];
        const float arm_R_deg = -kArmFactor * step_angle_deg_[1];

        // Swing the arms about bone-local Z, exactly like the legs.
        //
        // This used to use the per-bone captured swing_axis_local, but
        // on this rig (every bone bind rotation is identity) that capture
        // resolves to root_rot*right, double-counting the body yaw -- it
        // came out ~(0.81,0,0.58) (mostly X), which swings the arms out
        // to the SIDES instead of fore-aft.  Bone-local Z is the true
        // fore-aft (pitch) hinge here (shoulders are separated along Z;
        // arms hang -Y), and it matches the leg path's proven kSwingAxis.
        const glm::vec3 kArmSwingAxis(0.0f, 0.0f, 1.0f);
        auto applyBoneSwing = [&](const char* name, float deg) {
            auto it = bind_rots_.find(name);
            if (it == bind_rots_.end()) return;
            const BoneBindRot& b = it->second;
            const glm::quat dq = axisAngleDeg(kArmSwingAxis, deg);
            player->setNodeRotationByName(name, b.rot * dq);
        };

        applyBoneSwing("left_upper_arm",  arm_L_deg);
        applyBoneSwing("right_upper_arm", arm_R_deg);

        // Elbows: bend INTO the swing (positive sign on |arm_deg|) so
        // the forearm tucks slightly as the arm comes forward.  The
        // lower_arm has its own captured swing_axis_local relative to
        // ITS bind (which is the rest forearm direction relative to
        // the upper arm) -- it's already the right hinge for an
        // elbow-flexion rotation.
        const float elbow_L_deg = kElbowFactor * std::fabs(arm_L_deg);
        const float elbow_R_deg = kElbowFactor * std::fabs(arm_R_deg);
        applyBoneSwing("left_lower_arm",  elbow_L_deg);
        applyBoneSwing("right_lower_arm", elbow_R_deg);

        // Spine: counter-twist the hip rotation.  When left leg is
        // ahead of right (theta_L > theta_R), shoulders rotate to
        // bring the LEFT shoulder back (negative twist around body
        // vertical).  If the spine twists the wrong way, flip sign.
        const float spine_deg = kSpineFactor *
            (step_angle_deg_[0] - step_angle_deg_[1]);
        applyBoneDelta("spine", axisAngleDeg(kSpineAxis, spine_deg));

        // No foot-IK pelvis drop authored here -- decay any residual
        // from a previous mode so the body settles back to position_.y.
        pelvis_drop_ *= 0.5f;
        if (pelvis_drop_ < 1e-4f) pelvis_drop_ = 0.0f;
        const bool ik_ran = true;

        // ── Per-second pose-path diagnostic ─────────────────────────
        // Confirms (a) the bind was captured, (b) the IK path ran for
        // both legs, (c) the controller-supplied angles match what's
        // written upstream in [step_anim.live], AND (d) -- the new
        // bit -- whether the bone rotations we wrote actually
        // propagated into the rig's cached world matrices.
        //
        // Method: read each foot bone's world matrix AND its parent
        // upper-leg hip's world matrix from the rig (these were
        // refreshed on the PREVIOUS frame's DrawableData::update).
        // Log the foot-minus-hip world offset.  Interpretation:
        //   • If offset_L / offset_R CHANGE across log lines (the
        //     vector rotates), then setNodeRotationByName is being
        //     consumed by getCachedMatrix() and reaching the GPU
        //     joints buffer -- skinning is alive, the visible mesh
        //     SHOULD deform.  If it doesn't, the rig is being drawn
        //     through a path that bypasses the joints buffer (look
        //     downstream of base.vert / cluster pipeline).
        //   • If offset_L / offset_R are STATIC across log lines
        //     (numbers don't change despite [step_anim.live] thetaL
        //     ranging ±22°), then the IK writes are being dropped
        //     somewhere between setNodeRotationByName and the cached
        //     matrix -- likely m_use_local_matrix_only_ is suppressing
        //     parent-chain re-evaluation, or the animation update is
        //     overwriting our rotations.  The fix lives in
        //     DrawableData::update / getNodeMatrix, NOT in the IK.
        static unsigned s_pose_log = 0;
        if ((s_pose_log++ % 60u) == 0u) {
            const glm::mat4 H_L = player->getNodeWorldMatrixByName("left_upper_leg");
            const glm::mat4 H_R = player->getNodeWorldMatrixByName("right_upper_leg");
            const glm::mat4 F_L = player->getNodeWorldMatrixByName("left_foot");
            const glm::mat4 F_R = player->getNodeWorldMatrixByName("right_foot");
            const glm::vec3 off_L(F_L[3].x - H_L[3].x,
                                  F_L[3].y - H_L[3].y,
                                  F_L[3].z - H_L[3].z);
            const glm::vec3 off_R(F_R[3].x - H_R[3].x,
                                  F_R[3].y - H_R[3].y,
                                  F_R[3].z - H_R[3].z);
            std::cout << "[step_anim.pose]"
                      << " leg_bind=" << (leg_bind_captured_ ? 1 : 0)
                      << " Lvalid=" << (leg_bind_[0].valid ? 1 : 0)
                      << " Rvalid=" << (leg_bind_[1].valid ? 1 : 0)
                      << " ik_ran=" << (ik_ran ? 1 : 0)
                      << " thetaL=" << step_angle_deg_[0] << "deg"
                      << " thetaR=" << step_angle_deg_[1] << "deg"
                      << " offL=(" << off_L.x << "," << off_L.y << "," << off_L.z << ")"
                      << " offR=(" << off_R.x << "," << off_R.y << "," << off_R.z << ")"
                      << std::endl;
        }
    } else if (!applyFootIk(player)) {
        applyBoneDelta("left_upper_leg",  dq_l_leg);
        applyBoneDelta("right_upper_leg", dq_r_leg);
        applyBoneDelta("left_lower_leg",  dq_l_knee);
        applyBoneDelta("right_lower_leg", dq_r_knee);
    }
}

bool PlayerController::applyFootIk(
    const std::shared_ptr<DrawableObject>& player) {
    // ── Why-am-I-bailing diagnostic ──────────────────────────────────
    // [foot_ik] never printed, so the function is returning before the
    // solve.  Log (throttled ~once/2s) exactly which exit fires so we
    // can see whether it's the master toggle, missing leg bones / wrong
    // bone names, or a bind capture that never validates.  Remove once
    // the IK path is confirmed live.
    static unsigned s_why = 0;
    const bool why = ((s_why++ % 120u) == 0u);
    if (!foot_ik_enabled_) {
        if (why) std::cout << "[foot_ik.why] bail: foot_ik_enabled_=false"
                           << std::endl;
        return false;
    }
    if (!ground_query_) {
        if (why) std::cout << "[foot_ik.why] bail: no ground_query_ set"
                           << std::endl;
        return false;
    }
    const int li_dbg = player->findNodeIndexByName("left_upper_leg");
    const int ri_dbg = player->findNodeIndexByName("right_upper_leg");
    if (li_dbg < 0 && ri_dbg < 0) {
        if (why) std::cout << "[foot_ik.why] bail: no leg bones "
                              "(left_upper_leg idx=" << li_dbg
                           << " right_upper_leg idx=" << ri_dbg
                           << ") -- bone names don't match the rig"
                           << std::endl;
        return false;
    }

    // Snapshot the initialized (bind) leg pose ONCE — the stateless solver
    // needs its frozen bone lengths + reference axes.  Retry until the rig's
    // leg world matrices are actually populated (a valid capture).
    if (!leg_bind_captured_) {
        captureLegBind(player);
        if (leg_bind_[0].valid || leg_bind_[1].valid) leg_bind_captured_ = true;
        else {
            if (why) std::cout << "[foot_ik.why] bail: bind capture not "
                                  "ready (L0valid=" << leg_bind_[0].valid
                               << " L1valid=" << leg_bind_[1].valid
                               << ") -- knee/foot bone names or world "
                                  "matrices not populated"
                               << std::endl;
            return false;
        }
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
        // Gait scaled by the smoothed walk_weight_ so the foot eases into
        // and out of stepping instead of snapping when walking_ flips.
        float step = 0.0f, lift = 0.0f;
        if (walk_weight_ > 1e-3f) {
            step = -std::cos(L.phase) * foot_stride_amp_ * walk_weight_;
            const float sn = std::sin(L.phase);
            lift = (sn > 0.0f ? sn : 0.0f) * foot_lift_amp_ * walk_weight_;
        }
        // Foot rests under its own hip, displaced along travel by the step.
        const float gx = hip_w.x + fwd.x * step;
        const float gz = hip_w.z + fwd.z * step;

        // Probe ~0.9 m below the hip so an upper storey above is ignored.
        float gy = hip_w.y;
        glm::vec3 gn(0.0f, 1.0f, 0.0f);
        FootGround& fg = foot_ground_[L.idx];
        if (ground_query_(gx, gz, hip_w.y - 0.9f, gy, gn)) {
            if (glm::length(gn) < 0.1f) gn = up;
            // Low-pass small ground changes to kill the per-frame Y pop
            // when the probe switches between the decimated collision
            // floor and the exact rendered floor; SNAP through big steps
            // (>0.25 m) so real curbs / stairs stay crisp.
            if (fg.valid && std::fabs(gy - fg.y) < 0.25f) {
                constexpr float kGroundLerp = 0.35f;
                gy = fg.y + (gy - fg.y) * kGroundLerp;
                gn = glm::normalize(glm::mix(fg.nrm, gn, kGroundLerp));
            }
            fg.valid = true; fg.y = gy; fg.nrm = gn;
        } else if (fg.valid) {
            // Probe missed this frame (off-mesh / triangle budget hit):
            // HOLD the last good hit rather than `continue`, which left
            // the leg unposed for a frame and then snapped it back.
            gy = fg.y; gn = fg.nrm;
        } else {
            continue;  // never had a hit yet — can't place this foot
        }

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
    if (!std::isfinite(worst_deficit)) worst_deficit = 0.0f;
    const float target_drop =
        glm::clamp(worst_deficit, 0.0f, pelvis_drop_max_);
    pelvis_drop_ += (target_drop - pelvis_drop_) * 0.2f;
    if (!std::isfinite(pelvis_drop_) || pelvis_drop_ < 1e-4f)
        pelvis_drop_ = 0.0f;

    // Once/sec ground-truth dump (left leg).  If target.y ~= hip.y the ground
    // probe is feeding a near-hip surface (guard then clamps it); otherwise
    // the foot should sit right on target.y.
    if (dbg_have) {
        // std::cout (not printf) so the line lands in the captured engine
        // stdout log alongside the [collision.*] diagnostics; L1/L2 here
        // are the decisive numbers for the separate leg-length question.
        static unsigned s_ik_log = 0;
        if ((s_ik_log++ % 60u) == 0u) {
            std::cout << "[foot_ik] hip=(" << dbg_hip.x << "," << dbg_hip.y
                      << "," << dbg_hip.z << ") foot=(" << dbg_foot.x << ","
                      << dbg_foot.y << "," << dbg_foot.z << ") gy=" << dbg_gy
                      << " target=(" << dbg_target.x << "," << dbg_target.y
                      << "," << dbg_target.z << ") L1=" << leg_bind_[0].L1
                      << " L2=" << leg_bind_[0].L2
                      << " reach=" << (leg_bind_[0].L1 + leg_bind_[0].L2)
                      << " def=" << dbg_def << " drop=" << pelvis_drop_
                      << " ww=" << walk_weight_
                      << " walk=" << (walking_ ? 1 : 0) << std::endl;
        }
    } else if (why) {
        // Reached the solve loop but the LEFT leg (idx 0) never solved, so
        // the [foot_ik] line above can't print.  Surface why: either the
        // left bind is invalid, or every ground probe under it failed with
        // no cached hit yet.
        std::cout << "[foot_ik.why] ran loop but left leg not solved (any="
                  << any << " L0valid=" << leg_bind_[0].valid
                  << " L1valid=" << leg_bind_[1].valid << ")" << std::endl;
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
    // NaN recovery: if the hip world is non-finite, a NaN has already
    // propagated into the rig (e.g. a bad pelvis drop upstream).  Reset
    // this leg's bones to their VALID bind rotations and bail -- writing
    // garbage here is what keeps the rig stuck with collapsed/invisible
    // legs.  Once the root recovers next frame the chain is finite again.
    if (!isFiniteV3(a) || !isFiniteQ(RhipW)) {
        player->setNodeRotationByName(hip_name,  B.qh0);
        player->setNodeRotationByName(knee_name, B.qk0);
        player->setNodeRotationByName(foot_name, B.qf0);
        return;
    }
    const glm::quat qh_cur      = player->getNodeRotationByName(hip_name);
    const glm::quat RhipParentW = RhipW * glm::inverse(qh_cur);

    glm::vec3 d3 = target - a;
    float dist = glm::length(d3);
    if (dist < 1e-5f) return;                       // target on the hip
    const glm::vec3 dir = d3 / dist;

    // ── Joint angle-constraint limits (anatomical; tweak as needed) ──────
    constexpr float kKneeMaxFlexDeg = 150.0f; // max bend at the knee
    constexpr float kHipMaxConeDeg  = 95.0f;  // thigh swing from rest dir
    constexpr float kFootMaxTiltDeg = 30.0f;  // ankle align-to-ground cap

    // Reach clamp: never past full extension (no hyperextension), and never
    // closer than the distance for MAX knee flexion (so the shin can't fold
    // through the thigh).  Law of cosines at interior knee angle
    // (180 - maxFlex): dist² = L1² + L2² - 2·L1·L2·cos(interior).
    const float eps = 0.002f;
    const float hi  = B.L1 + B.L2 - eps;
    const float interior_min = glm::radians(180.0f - kKneeMaxFlexDeg);
    const float dmin_flex = std::sqrt(std::max(0.0f,
        B.L1 * B.L1 + B.L2 * B.L2 -
        2.0f * B.L1 * B.L2 * std::cos(interior_min)));
    const float lo = std::max(std::fabs(B.L1 - B.L2) + eps, dmin_flex);
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

    // Analytic upper-leg direction (exact two-bone solution).
    glm::vec3 u1 = glm::angleAxis(alpha, bendW) * dir;

    // ── HIP CONE constraint ──────────────────────────────────────────────
    // Clamp the thigh direction to within kHipMaxConeDeg of its rest (bind)
    // direction so the hip can't swing into an impossible pose.  rest_dir is
    // the bind upper-leg axis expressed in the CURRENT body frame.
    const glm::quat RhipW_bind = RhipParentW * B.qh0;
    const glm::vec3 rest_dir   = glm::normalize(RhipW_bind * B.ahx);
    {
        const float ang  = std::acos(
            glm::clamp(glm::dot(u1, rest_dir), -1.0f, 1.0f));
        const float cone = glm::radians(kHipMaxConeDeg);
        if (ang > cone) {
            const glm::vec3 axis = glm::cross(rest_dir, u1);
            if (glm::length(axis) > 1e-6f)
                u1 = glm::normalize(
                    glm::angleAxis(cone, glm::normalize(axis)) * rest_dir);
        }
    }

    const glm::vec3 knee  = a + u1 * B.L1;
    const glm::vec3 footP = a + dir * dist;
    glm::vec3 v1 = footP - knee;
    const float v1len = glm::length(v1);
    if (v1len < 1e-6f) return;
    v1 /= v1len;

    // ── KNEE FLEXION constraint ──────────────────────────────────────────
    // Flexion = angle between thigh (u1) and shin (v1): 0 straight, larger
    // when bent.  Clamp to kKneeMaxFlexDeg so the shin can't fold back
    // through the thigh.  Rotating v1 toward u1 keeps the SAME bend plane
    // (no direction flip), it only reduces the magnitude.
    {
        const float flex = std::acos(
            glm::clamp(glm::dot(u1, v1), -1.0f, 1.0f));
        const float maxf = glm::radians(kKneeMaxFlexDeg);
        if (flex > maxf) {
            const glm::vec3 axis = glm::cross(u1, v1);
            if (glm::length(axis) > 1e-6f)
                v1 = glm::normalize(
                    glm::angleAxis(maxf, glm::normalize(axis)) * u1);
        }
    }

    // Hip: swing the bind upper-leg direction (rest_dir) onto the
    // constrained u1, written as a LOCAL rotation.
    const glm::quat RhipW_new = fromTo(rest_dir, u1) * RhipW_bind;
    {
        const glm::quat hipLocal =
            glm::normalize(glm::inverse(RhipParentW) * RhipW_new);
        player->setNodeRotationByName(
            hip_name, isFiniteQ(hipLocal) ? hipLocal : B.qh0);
    }

    // Knee: swing the bind lower-leg direction onto v1 under the NEW hip.
    const glm::quat RkneeW_bind = RhipW_new * B.qk0;
    const glm::vec3 v_bind_now  = RkneeW_bind * B.akx;
    const glm::quat RkneeW_new  = fromTo(v_bind_now, v1) * RkneeW_bind;
    {
        const glm::quat kneeLocal =
            glm::normalize(glm::inverse(RhipW_new) * RkneeW_new);
        player->setNodeRotationByName(
            knee_name, isFiniteQ(kneeLocal) ? kneeLocal : B.qk0);
    }

    // Foot: bind orientation under the new knee, then tilt toward the ground
    // normal (self-correcting, clamped).  Uses the freshly computed knee
    // world rotation, so no stale cached read is involved.
    glm::quat RfootW = RkneeW_new * B.qf0;
    if (foot_tilt_w > 1e-3f) {
        const glm::vec3 cur_up = RfootW * glm::vec3(0.0f, 1.0f, 0.0f);
        glm::vec3 ax; float ang;
        if (axisAngleBetween(cur_up, ground_normal, ax, ang)) {
            // Ankle tilt-to-ground capped at kFootMaxTiltDeg so the foot
            // can't roll past an anatomical limit relative to the shin.
            ang = glm::clamp(ang * foot_tilt_w, 0.0f,
                             glm::radians(kFootMaxTiltDeg));
            if (ang > 1e-4f) RfootW = glm::angleAxis(ang, ax) * RfootW;
        }
    }
    {
        const glm::quat footLocal =
            glm::normalize(glm::inverse(RkneeW_new) * RfootW);
        player->setNodeRotationByName(
            foot_name, isFiniteQ(footLocal) ? footLocal : B.qf0);
    }
}

} // namespace game_object
} // namespace engine
