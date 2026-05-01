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

namespace engine {
namespace game_object {

namespace {

// Wrap an angle into [-180, 180] for shortest-path interpolation.
float wrapDeg(float a) {
    a = std::fmod(a + 180.0f, 360.0f);
    if (a < 0.0f) a += 360.0f;
    return a - 180.0f;
}

// Move `current` toward `target` by at most `max_step`.
float chaseAngleDeg(float current, float target, float max_step) {
    float diff = wrapDeg(target - current);
    if (std::fabs(diff) <= max_step) return target;
    return current + (diff > 0.0f ? max_step : -max_step);
}

// Build a quaternion from an axis-angle in degrees.
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
    const std::vector<std::shared_ptr<DrawableObject>>& obstacles) {

    if (!player || !player->isReady()) {
        // Async load still in flight — nothing to drive yet. Bail out
        // before touching the rig so we don't write into an
        // uninitialised DrawableData.
        return;
    }

    // ── Lazy spawn: drop the player on the ground at the current
    //    camera-relative origin the first time we run. The model is
    //    authored Z-up with the Sketchfab basis matrix doing Z↔Y
    //    flips, so spawning at world (0, terrain, 0) gives the
    //    expected feet-on-ground placement.
    if (!initialized_) {
        position_ = glm::vec3(
            0.0f,
            getTerrainGroundHeight(glm::vec2(0.0f, 0.0f)),
            0.0f);
        yaw_deg_ = camera_yaw_deg;
        initialized_ = true;
    }

    // ── Read WASD ──────────────────────────────────────────────────────
    // We poll directly from GLFW so movement is continuous (the
    // single-shot keyInputCallback in application.cpp would only give
    // us one frame of "pressed" per physical press, which is fine for
    // toggles but bad for sustained movement).
    bool key_w = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
    bool key_s = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
    bool key_a = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
    bool key_d = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;

    // ── Convert input to a planar move direction relative to the
    //    camera's yaw. The camera convention here is the same as
    //    getDirectionByYawAndPitch() in camera_object.cpp:
    //       forward_xz = (cos(-yaw),       0, sin(-yaw))
    //       right_xz   = (cos(-yaw - 90), 0, sin(-yaw - 90))
    //    so W/D follow the camera's gaze in the horizontal plane.
    float yaw_rad = glm::radians(-camera_yaw_deg);
    glm::vec3 fwd  ( std::cos(yaw_rad), 0.0f, std::sin(yaw_rad));
    glm::vec3 right(-std::sin(yaw_rad), 0.0f, std::cos(yaw_rad));

    glm::vec3 input_dir(0.0f);
    if (key_w) input_dir += fwd;
    if (key_s) input_dir -= fwd;
    if (key_d) input_dir += right;
    if (key_a) input_dir -= right;

    walking_ = false;
    glm::vec3 desired_pos = position_;
    if (glm::length(input_dir) > 1e-4f) {
        input_dir = glm::normalize(input_dir);
        desired_pos += input_dir * walk_speed_ * delta_t;
        walking_ = true;

        // Aim the character at the movement direction. atan2 returns
        // the angle of (z, x) in our convention so the resulting yaw
        // matches the camera.
        float target_yaw =
            -glm::degrees(std::atan2(input_dir.z, input_dir.x));
        yaw_deg_ = chaseAngleDeg(
            yaw_deg_, target_yaw, turn_speed_ * delta_t);
    }

    // ── Collision: slide the desired XZ position around obstacles. ─────
    glm::vec2 resolved_xz = resolveXzCollisions(
        glm::vec2(position_.x, position_.z),
        glm::vec2(desired_pos.x, desired_pos.z),
        obstacles);
    desired_pos.x = resolved_xz.x;
    desired_pos.z = resolved_xz.y;

    // ── Terrain clamp: snap feet to ground height. This is "collision
    //    detection against the level" for the heightfield half of the
    //    world. We don't add gravity / jumping here — feet are pinned.
    desired_pos.y =
        getTerrainGroundHeight(glm::vec2(desired_pos.x, desired_pos.z));

    position_ = desired_pos;

    // ── Advance animation phases.  Walk cycle is keyed off planar
    //    speed so it scales with player velocity; idle-bob runs on
    //    a free-running clock.
    if (walking_) {
        // Roughly two steps per second at full walk speed — a
        // pleasant "jog" cadence.
        anim_phase_ += delta_t * 6.5f;
    }
    idle_phase_ += delta_t * 1.5f;

    // Push transforms into the rig.
    applyPose(player);
}

glm::vec2 PlayerController::resolveXzCollisions(
    const glm::vec2& current_xz,
    const glm::vec2& desired_xz,
    const std::vector<std::shared_ptr<DrawableObject>>& obstacles) const {

    glm::vec2 result = desired_xz;

    for (const auto& obj : obstacles) {
        if (!obj || !obj->isReady()) continue;

        // Build a world-axis-aligned XZ box from the obstacle's local
        // bbox transformed by its location_ matrix. (The engine's
        // location_ is otherwise unused for placement, but it does
        // correctly represent "where this drawable lives" — see
        // application.cpp where it's set to inverse(view) at spawn.)
        glm::vec3 mn = obj->getModelBboxMin();
        glm::vec3 mx = obj->getModelBboxMax();
        if (mn.x >= mx.x || mn.z >= mx.z) continue;  // degenerate

        const glm::mat4& M = obj->getLocation();
        // Transform the 4 ground-plane corners — good enough for
        // axis-aligned obstacles, conservative for rotated ones.
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
        // Inflate by player radius so we treat the player as a circle.
        amn -= glm::vec2(player_radius_);
        amx += glm::vec2(player_radius_);

        // Outside the box? No collision.
        if (result.x <= amn.x || result.x >= amx.x ||
            result.y <= amn.y || result.y >= amx.y) continue;

        // Penetration depth on each axis. Push the player out along
        // the shallowest axis so motion grazing the box "slides".
        float dx_neg = result.x - amn.x;  // distance to push -X to clear
        float dx_pos = amx.x - result.x;  // distance to push +X to clear
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

    // Root translation+rotation: yaw is around world-up (Y).
    glm::quat root_rot = axisAngleDeg(glm::vec3(0, 1, 0), yaw_deg_);
    player->setRootNodeTransform(position_, root_rot);

    // Walk-cycle parameters. Amplitudes are deliberately modest — the
    // rig's bind pose is upright, and we only have a handful of bones
    // to swing, so anything bigger reads as flailing.
    const float walk_arm_amp_deg = walking_ ? 35.0f : 0.0f;
    const float walk_leg_amp_deg = walking_ ? 30.0f : 0.0f;

    // Idle: a small breathing-like sway on the spine + chest, plus a
    // subtle vertical bob that we fold into the hips translation. Keeps
    // the character "alive" while standing still.
    const float idle_sway_deg = std::sin(idle_phase_) * 1.5f;

    // Drive opposing arms / legs against each other (left arm forward
    // when right leg forward — natural gait).
    float swing = std::sin(anim_phase_);
    float swing_opp = -swing;

    // Pitch (X-axis) for arms and legs. The model's default frame has
    // arms hanging down along -Y; rotating around the bone's local X
    // swings them forward/back.
    glm::quat l_arm  = axisAngleDeg(glm::vec3(1, 0, 0), swing     * walk_arm_amp_deg);
    glm::quat r_arm  = axisAngleDeg(glm::vec3(1, 0, 0), swing_opp * walk_arm_amp_deg);
    glm::quat l_leg  = axisAngleDeg(glm::vec3(1, 0, 0), swing_opp * walk_leg_amp_deg);
    glm::quat r_leg  = axisAngleDeg(glm::vec3(1, 0, 0), swing     * walk_leg_amp_deg);
    // A little knee bend on the back-swinging leg keeps the gait
    // from looking robotic.
    float l_knee_deg = walking_ ? std::max(0.0f,  swing_opp) * 25.0f : 0.0f;
    float r_knee_deg = walking_ ? std::max(0.0f,  swing)     * 25.0f : 0.0f;
    glm::quat l_knee = axisAngleDeg(glm::vec3(1, 0, 0),  l_knee_deg);
    glm::quat r_knee = axisAngleDeg(glm::vec3(1, 0, 0),  r_knee_deg);
    // Small elbow bend that follows the arm swing forward.
    glm::quat l_elbow = axisAngleDeg(glm::vec3(1, 0, 0), std::max(0.0f, swing)     * 20.0f);
    glm::quat r_elbow = axisAngleDeg(glm::vec3(1, 0, 0), std::max(0.0f, swing_opp) * 20.0f);

    // Spine sway in idle so the character isn't a statue.
    glm::quat spine_q = axisAngleDeg(glm::vec3(0, 0, 1), idle_sway_deg);

    // Bone names match scene-skinned.gltf — see asset's "nodes" block.
    player->setNodeRotationByName("left_upper_arm",  l_arm);
    player->setNodeRotationByName("right_upper_arm", r_arm);
    player->setNodeRotationByName("left_lower_arm",  l_elbow);
    player->setNodeRotationByName("right_lower_arm", r_elbow);
    player->setNodeRotationByName("left_upper_leg",  l_leg);
    player->setNodeRotationByName("right_upper_leg", r_leg);
    player->setNodeRotationByName("left_lower_leg",  l_knee);
    player->setNodeRotationByName("right_lower_leg", r_knee);
    player->setNodeRotationByName("spine",           spine_q);
}

} // namespace game_object
} // namespace engine
