#pragma once

#include <memory>
#include <vector>
#include <string>
#include "renderer/renderer.h"

struct GLFWwindow;

namespace engine {
namespace helper { class CollisionWorld; }
namespace game_object {

class DrawableObject;

// PlayerController — third-person character driver.
//
// Reads WASD each frame, advances the player relative to the camera
// yaw with terrain-clamped Y, resolves collisions against (a) any
// triangle-mesh CollisionWorld passed in and (b) AABBs of fallback
// drawable obstacles, then drives the rigged scene-skinned.gltf rig
// procedurally for an idle-bob + walk cycle.
class PlayerController {
public:
    PlayerController();

    // Per-frame tick. `world` is optional; when non-null it overrides
    // the AABB obstacle path with proper capsule-vs-triangle resolve.
    void update(
        GLFWwindow* window,
        float delta_t,
        float camera_yaw_deg,
        const std::shared_ptr<DrawableObject>& player,
        const std::vector<std::shared_ptr<DrawableObject>>& obstacles,
        const helper::CollisionWorld* world = nullptr);

    glm::vec3 getPosition() const { return position_; }
    float     getYaw() const      { return yaw_deg_; }
    bool      isWalking() const   { return walking_; }
    bool      isSpawned() const   { return initialized_; }

    // Place the player at a specific world position + facing yaw.
    // Calling this flips initialized_ so update() will start running
    // input/physics/animation. The application calls it once the
    // level has finished streaming so the character appears in front
    // of the camera (rather than the lazy default spawn at origin).
    void spawnAt(const glm::vec3& world_position, float yaw_deg) {
        position_    = world_position;
        yaw_deg_     = yaw_deg;
        anim_phase_  = 0.0f;
        idle_phase_  = 0.0f;
        walking_     = false;
        initialized_ = true;
    }

    // Capsule shape — exposed so the caller can tune it without
    // recompiling.  `height` includes the two hemispheres.
    void setCapsule(float radius, float height) {
        player_radius_ = radius;
        player_height_ = height;
    }
    float capsuleRadius() const { return player_radius_; }
    float capsuleHeight() const { return player_height_; }

private:
    void applyPose(const std::shared_ptr<DrawableObject>& player);

    glm::vec2 resolveXzCollisions(
        const glm::vec2& current_xz,
        const glm::vec2& desired_xz,
        const std::vector<std::shared_ptr<DrawableObject>>& obstacles) const;

    glm::vec3 position_      = glm::vec3(0.0f, 0.0f, 0.0f);
    float     yaw_deg_       = 0.0f;
    float     anim_phase_    = 0.0f;
    float     idle_phase_    = 0.0f;
    float     walk_speed_    = 5.0f;
    float     turn_speed_    = 720.0f;
    float     player_radius_ = 0.4f;     // capsule radius
    float     player_height_ = 1.8f;     // total capsule height (incl. hemispheres)
    bool      walking_       = false;
    bool      initialized_   = false;
};

} // namespace game_object
} // namespace engine
