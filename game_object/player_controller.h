#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <functional>
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

    // ── Foot inverse-kinematics ─────────────────────────────
    // The application owns the scene + collision data, so it supplies a
    // ground-query callback: given a world XZ column and a Y "hint"
    // (roughly the expected foot level) it returns the ground height and
    // surface normal beneath that column.  PlayerController uses it to
    // plant each foot independently via a two-bone (hip+knee) solve, tilt
    // the foot to the surface, and drop the pelvis when a foot can't
    // reach.  With no callback set (or foot IK disabled) the legs fall
    // back to the pure procedural swing.
    //   x, z    — world-space column to probe.
    //   y_hint  — expected foot Y (lets the probe pick the nearest
    //             surface and ignore an upper storey above the player).
    //   out_y   — world Y of the ground hit.
    //   out_nrm — unit surface normal at the hit.
    //   returns — true on a hit; false leaves out_* untouched.
    using GroundQueryFn = std::function<
        bool(float x, float z, float y_hint,
             float& out_y, glm::vec3& out_nrm)>;
    void setGroundQuery(GroundQueryFn fn) { ground_query_ = std::move(fn); }
    void setFootIkEnabled(bool e) { foot_ik_enabled_ = e; }
    bool footIkEnabled() const    { return foot_ik_enabled_; }
    // Live-tunable IK knobs (wired to the Physics > Foot IK menu).
    void setFootStrideAmp(float v)  { foot_stride_amp_  = v; }
    void setFootLiftAmp(float v)    { foot_lift_amp_    = v; }
    void setFootSoleDrop(float v)   { foot_sole_drop_   = v; }
    void setFootTiltWeight(float v) { foot_tilt_weight_ = v; }
    void setPelvisDropMax(float v)  { pelvis_drop_max_  = v; }
    // Amount the rig root was lowered this frame because a foot could
    // not otherwise reach its ground target (>= 0).  For diagnostics.
    float pelvisDrop() const      { return pelvis_drop_; }

private:
    void applyPose(const std::shared_ptr<DrawableObject>& player);

    // Plants both feet on the ground via two-bone IK and updates
    // pelvis_drop_.  Returns true when it took ownership of the legs
    // (so applyPose skips the procedural leg/knee swing).  Returns
    // false — a no-op — when disabled, no ground_query_ is set, or
    // the rig lacks the *_upper_leg / *_lower_leg / *_foot bones.
    bool applyFootIk(const std::shared_ptr<DrawableObject>& player);

    // Snapshot the rig's initialized (bind) leg pose ONCE: per-leg bone
    // lengths, bind local rotations, and reference axes.  Everything the
    // stateless solver needs is captured here so the per-frame solve
    // never depends on the previous frame's (possibly drifted) pose.
    void captureLegBind(const std::shared_ptr<DrawableObject>& player);

    // Stateless pole-vector two-bone solve for one leg: places the foot
    // bone on `target` in a single pass using the frozen bind bone
    // lengths, then orients the foot toward `ground_normal`.  Reads only
    // the hip world position + parent rotation (neither depends on the
    // leg's own IK), so it cannot diverge.  out_deficit = how far the
    // target lay beyond reach (0 if reachable) -> pelvis-drop signal.
    void solveLegIk(const std::shared_ptr<DrawableObject>& player,
                    int leg_idx, const char* hip_name,
                    const char* knee_name, const char* foot_name,
                    const glm::vec3& target,
                    const glm::vec3& ground_normal,
                    float foot_tilt_w, float& out_deficit);

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

    // ── Foot IK state ────────────────────────────────
    GroundQueryFn ground_query_;          // app-supplied ground probe
    bool   foot_ik_enabled_ = true;       // master toggle
    // Horizontal facing of motion, captured in update() from the
    // per-frame XZ velocity (kept from the last non-trivial step so
    // it stays valid while standing).  Used to place the walking
    // foot-step offsets along the direction of travel.
    glm::vec2 gait_fwd_xz_ = glm::vec2(0.0f, 1.0f);
    // Smoothed amount the pelvis (root) is lowered this frame because a
    // foot couldn't otherwise reach its ground target.  Applied as a
    // negative-Y offset inside applyPose's setRootNodeTransform call and
    // lerped toward its target each frame to avoid popping.
    float  pelvis_drop_      = 0.0f;

    // ── Flicker-suppression state (IK jitter fix) ────────────────────
    // walk_weight_: smoothed 0..1 gait blend.  walking_ is now a
    // hysteresis boolean (see update()); this eases the foot step/lift
    // in and out so crossing the walk/idle threshold no longer SNAPS the
    // foot targets — one of the main flicker sources.
    float  walk_weight_      = 0.0f;
    // Per-leg ground-probe cache + low-pass.  The collision floor is
    // decimated/holey while the rendered floor is exact, so queryGroundAt
    // can switch sources between frames and the hit Y pops up/down.  We
    // smooth small steps and HOLD the last good hit when a probe briefly
    // misses, instead of skipping the leg (which froze it for a frame
    // then snapped it back — visible flicker).
    struct FootGround {
        bool      valid = false;
        float     y     = 0.0f;
        glm::vec3 nrm   = glm::vec3(0.0f, 1.0f, 0.0f);
    };
    FootGround foot_ground_[2];
    // Tuning (metres / unitless).  Conservative defaults so the primary
    // goal (feet on the ground) still reads well even if the gait is
    // slightly off; edit + recompile to tweak.
    float  foot_stride_amp_  = 0.18f;     // fwd/back foot travel, walking
    float  foot_lift_amp_    = 0.10f;     // max foot lift mid-swing
    float  foot_sole_drop_   = 0.08f;     // ankle bone -> sole distance
    float  foot_tilt_weight_ = 0.6f;      // 0=flat, 1=full align to normal
    float  pelvis_drop_max_  = 0.6f;      // clamp so a bug can't sink her

    // ── Captured bind (default-pose) leg reference for the IK ─────────
    // Filled once by captureLegBind() from the initialized skeleton.
    // The solver treats this as the canonical default pose and uses the
    // frozen bone lengths; nothing here changes after capture.
    struct LegIkBind {
        bool      valid = false;
        float     L1 = 0.0f, L2 = 0.0f;     // bind bone lengths (world)
        glm::quat qh0{1,0,0,0};             // bind hip  local rotation
        glm::quat qk0{1,0,0,0};             // bind knee local rotation
        glm::quat qf0{1,0,0,0};             // bind foot local rotation
        glm::vec3 ahx{0.0f,-1.0f,0.0f};     // upper-leg axis, hip-local
        glm::vec3 akx{0.0f,-1.0f,0.0f};     // lower-leg axis, knee-local
        glm::vec3 bend_hp{0.0f,0.0f,1.0f};  // knee hinge axis, hip-parent
    };
    LegIkBind leg_bind_[2];               // 0 = left, 1 = right
    bool      leg_bind_captured_ = false;
};

} // namespace game_object
} // namespace engine
