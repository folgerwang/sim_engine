#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>
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

    // Per-frame "follow the camera" hook.  Updates ONLY position +
    // yaw, without resetting animation phases (which spawnAt would).
    // The application calls this every frame after the one-shot
    // spawn block to keep the character glued to the camera (e.g.
    // 2 m in front, on the ground).  No-op until initialized_ has
    // been set by a prior spawnAt — otherwise we'd race against the
    // async player load.
    void setPositionAndYaw(const glm::vec3& world_position,
                           float yaw_deg) {
        if (!initialized_) return;
        position_ = world_position;
        yaw_deg_  = yaw_deg;
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
    // Previous frame's position_ — used to detect motion driven by
    // external code (the application's per-frame setPositionAndYaw
    // follow block).  When the controller is in stationary mode and
    // doesn't own movement, this is the only way it can tell whether
    // to play the walking animation vs. the idle sway.  Set to
    // position_ at first update() so frame 1 doesn't false-positive
    // a "huge teleport" as walking.
    glm::vec3 last_position_  = glm::vec3(0.0f, 0.0f, 0.0f);
    bool      last_position_valid_ = false;

    // ── Bind-pose rotation cache for named bones ─────────────────
    // applyPose() drives the rig procedurally by writing each named
    // bone's local rotation each frame.  The asset's bind pose has
    // its own rotations baked into node.rotation_ (orienting bones
    // along their local "down-the-arm" axes etc.) — overwriting
    // those with a raw axis-angle wipes the bind orientation and
    // distorts the pose (one frame "twist where it should be swing",
    // next frame the wrong axis pivots an entire limb sideways).
    //
    // Solution: snapshot each named bone's bind rotation on first
    // applyPose() call after isReady() flips, then on every subsequent
    // call compose:  final_rot = bind_rot * swing_delta_in_local_frame.
    // The swing delta is what we author each frame (axisAngleDeg(X, deg)
    // etc.) and the bind preserves the bone's orientation.
    struct BoneBindRot {
        glm::quat rot;
        int       node_idx = -1;   // cached for fast re-lookup
    };
    bool                          bind_rots_captured_ = false;
    std::map<std::string, BoneBindRot> bind_rots_;
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
