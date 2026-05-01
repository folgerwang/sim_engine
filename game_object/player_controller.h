#pragma once

#include <memory>
#include <vector>
#include <string>
#include "renderer/renderer.h"

struct GLFWwindow;

namespace engine {
namespace game_object {

class DrawableObject;

// PlayerController — third-person character driver.
//
// Responsibilities:
//   1. Read WASD from GLFW each frame, advance the player's world
//      position relative to the camera yaw, with terrain-clamped Y.
//   2. Resolve simple AABB collisions against a list of static
//      "level" obstacles by sliding the player on the obstacle face.
//   3. Drive the rigged scene-skinned.gltf rig procedurally
//      (the file ships with no animation channels) — set the root
//      node's TRS for translation/rotation, and rotate a handful
//      of named bones (hips, chest, arms, legs) for an idle-bob and
//      a walk cycle that swings opposing arms+legs in sync with the
//      player's planar speed.
//
// Lifetime: owned by RealWorldApplication; `update()` is called once
// per frame from the main loop, AFTER the camera has been updated for
// the same frame (so we can read the latest camera yaw).
class PlayerController {
public:
    PlayerController();

    // Per-frame tick.
    //   window         - GLFW window for polling W/A/S/D state.
    //   delta_t        - frame time in seconds.
    //   camera_yaw_deg - main camera yaw in degrees (player movement
    //                    is interpreted relative to this so pressing W
    //                    walks "into the screen" regardless of how the
    //                    camera is rotated).
    //   player         - the DrawableObject that loaded scene-skinned.gltf.
    //   obstacles      - other DrawableObjects to collide against
    //                    (treated as world-axis-aligned bounding boxes
    //                    in the XZ plane).
    void update(
        GLFWwindow* window,
        float delta_t,
        float camera_yaw_deg,
        const std::shared_ptr<DrawableObject>& player,
        const std::vector<std::shared_ptr<DrawableObject>>& obstacles);

    // World-space position of the player's pivot (feet).
    glm::vec3 getPosition() const { return position_; }

    // Facing yaw in degrees (0 = +X, increases CCW in XZ plane,
    // matching the camera convention).
    float getYaw() const { return yaw_deg_; }

    // True when the player is moving fast enough that the walk cycle
    // is being driven this frame.
    bool isWalking() const { return walking_; }

private:
    // Apply the procedural pose to the rig's bones based on the
    // current anim_phase_ and walking_ state.
    void applyPose(const std::shared_ptr<DrawableObject>& player);

    // Resolve collisions in the XZ plane: if `desired_xz` would push
    // the player into one of the obstacles, push them back along the
    // shallowest axis so the motion becomes a "slide". The XY
    // component of obstacles is ignored.
    glm::vec2 resolveXzCollisions(
        const glm::vec2& current_xz,
        const glm::vec2& desired_xz,
        const std::vector<std::shared_ptr<DrawableObject>>& obstacles) const;

    glm::vec3 position_      = glm::vec3(0.0f, 0.0f, 0.0f);
    float     yaw_deg_       = 0.0f;
    float     anim_phase_    = 0.0f;     // accumulated walk-cycle phase
    float     idle_phase_    = 0.0f;     // accumulated idle-bob phase
    float     walk_speed_    = 5.0f;     // m/s when WASD held
    float     turn_speed_    = 720.0f;   // deg/s — how fast yaw chases motion direction
    float     player_radius_ = 0.4f;     // collision radius in XZ
    bool      walking_       = false;
    bool      initialized_   = false;
};

} // namespace game_object
} // namespace engine
