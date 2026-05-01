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
    // finishes loading. Until then, leave the rig at its glTF rest
    // pose instead of teleporting it to the world origin.
    if (!initialized_) return;

    bool key_w = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
    bool key_s = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
    bool key_a = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
    bool key_d = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;

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
        float target_yaw =
            -glm::degrees(std::atan2(input_dir.z, input_dir.x));
        yaw_deg_ = chaseAngleDeg(
            yaw_deg_, target_yaw, turn_speed_ * delta_t);
    }

    glm::vec2 resolved_xz = resolveXzCollisions(
        glm::vec2(position_.x, position_.z),
        glm::vec2(desired_pos.x, desired_pos.z),
        obstacles);
    desired_pos.x = resolved_xz.x;
    desired_pos.z = resolved_xz.y;

    desired_pos.y =
        getTerrainGroundHeight(glm::vec2(desired_pos.x, desired_pos.z));

    if (world && !world->empty()) {
        glm::vec3 contact_normal(0.0f);
        for (int outer = 0; outer < 2; ++outer) {
            if (!world->resolveCapsule(
                    desired_pos,
                    player_radius_,
                    player_height_,
                    contact_normal)) {
                break;
            }
        }
        float ground = getTerrainGroundHeight(
            glm::vec2(desired_pos.x, desired_pos.z));
        if (desired_pos.y < ground) desired_pos.y = ground;
    }

    position_ = desired_pos;

    if (walking_) anim_phase_ += delta_t * 6.5f;
    idle_phase_ += delta_t * 1.5f;

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

    const float walk_arm_amp_deg = walking_ ? 35.0f : 0.0f;
    const float walk_leg_amp_deg = walking_ ? 30.0f : 0.0f;
    const float idle_sway_deg = std::sin(idle_phase_) * 1.5f;

    float swing = std::sin(anim_phase_);
    float swing_opp = -swing;

    glm::quat l_arm  = axisAngleDeg(glm::vec3(1, 0, 0), swing     * walk_arm_amp_deg);
    glm::quat r_arm  = axisAngleDeg(glm::vec3(1, 0, 0), swing_opp * walk_arm_amp_deg);
    glm::quat l_leg  = axisAngleDeg(glm::vec3(1, 0, 0), swing_opp * walk_leg_amp_deg);
    glm::quat r_leg  = axisAngleDeg(glm::vec3(1, 0, 0), swing     * walk_leg_amp_deg);
    float l_knee_deg = walking_ ? std::max(0.0f,  swing_opp) * 25.0f : 0.0f;
    float r_knee_deg = walking_ ? std::max(0.0f,  swing)     * 25.0f : 0.0f;
    glm::quat l_knee = axisAngleDeg(glm::vec3(1, 0, 0),  l_knee_deg);
    glm::quat r_knee = axisAngleDeg(glm::vec3(1, 0, 0),  r_knee_deg);
    glm::quat l_elbow = axisAngleDeg(glm::vec3(1, 0, 0), std::max(0.0f, swing)     * 20.0f);
    glm::quat r_elbow = axisAngleDeg(glm::vec3(1, 0, 0), std::max(0.0f, swing_opp) * 20.0f);
    glm::quat spine_q = axisAngleDeg(glm::vec3(0, 0, 1), idle_sway_deg);

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
