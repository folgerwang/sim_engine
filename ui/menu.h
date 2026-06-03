#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include "renderer/renderer.h"
#include "scene_rendering/skydome.h"
#include "shaders/global_definition.glsl.h"
#include "helper/gpu_profiler.h"
#include "chat_box.h"
#include "title_screen_config.h"
#include "imgui.h"

namespace plugins { class PluginManager; }

namespace engine {
namespace game_object { class MeshLoadTaskManager; class DrawableObject; }
namespace scene_rendering { class SSAO; class ClusterRenderer; class VirtualTextureManager; }
namespace ui {

// One entry in the editor Outliner.  The application rebuilds this list
// each frame from its live scene objects (player, NPCs, drawables) and
// hands it to the Menu via setSceneObjects().  `obj` is borrowed (the app
// owns the shared_ptr) and used by the Details panel for transform/visibility.
struct EditorSceneObject {
    std::string                          name;
    engine::game_object::DrawableObject* obj = nullptr;
};

// Game-level state machine driven by the title-screen menu.
enum class GameState {
    TitleScreen,   // showing background + menu items
    Loading,       // "New Game" clicked — meshes streaming
    InGame         // meshes loaded, gameplay active
};

// Live-tunable parameters for PlayerController's two-bone foot IK.
// Owned by the menu so the ImGui sliders bind directly to the
// fields; the application copies them into the controller every
// frame and pushes the resulting pelvis drop back for display.
// Defaults MUST match PlayerController's own initial values so a
// fresh session behaves identically until the user changes them.
struct FootIkParams {
    bool  enabled         = true;
    float stride_amp      = 0.18f;   // walking fwd/back foot travel (m)
    float lift_amp        = 0.10f;   // max foot lift mid-swing (m)
    float sole_drop       = 0.08f;   // ankle-bone -> sole distance (m)
    float tilt_weight     = 0.6f;    // 0=flat foot, 1=full align to normal
    float pelvis_drop_max = 0.6f;    // clamp on the IK pelvis drop (m)
};

class Menu {
    std::vector<std::string> gltf_file_names_;
    std::vector<std::string> to_load_gltf_names_;
    // One-shot "snap the player to the camera" request — drained by
    // consumeResetPlayerPosition() from application.cpp.  Set by the
    // "Reset player position" Game Objects menu item; the application
    // calls PlayerController::spawnAt() with the current camera pose
    // when this is true, which re-anchors the character to wherever
    // the user is currently looking.
    bool reset_player_position_requested_ = false;

    // Per-frame debug overlay data — see setPlayerDebugInfo().
    bool      has_player_debug_info_ = false;
    glm::vec3 player_world_pos_     = glm::vec3(0.0f);
    glm::vec3 player_cam_pos_       = glm::vec3(0.0f);
    glm::mat4 player_view_proj_     = glm::mat4(1.0f);
    // Player world-space AABB (post-cached-matrix transform of the
    // mesh bind-pose AABBs).  When player_bbox_valid_ is true the
    // menu paints a wireframe box overlay so we can visually verify
    // where the character's renderable volume actually sits — useful
    // for the "feet sunk below floor" diagnostic.  Set via the new
    // setPlayerDebugInfo overload that takes bbox_min/bbox_max.
    bool      player_bbox_valid_   = false;
    glm::vec3 player_bbox_min_     = glm::vec3(0.0f);
    glm::vec3 player_bbox_max_     = glm::vec3(0.0f);
    // ── Foot-IK tuning (bound to the Physics > Foot IK menu) ──────────
    FootIkParams foot_ik_params_;
    float        foot_ik_pelvis_drop_live_ = 0.0f;  // read-only readout
    bool turn_off_water_pass_ = false;
    bool turn_off_grass_pass_ = false;
    bool turn_off_ray_tracing_ = false;
    bool turn_off_volume_moist_ = false;
    bool turn_off_shadow_pass_ = false;
    // CSM silhouette prepass — fills each cascade's in-camera-frustum
    // region with depth=1 over a 0-cleared depth buffer so out-of-frustum
    // texels reject every shadow caster via LESS_OR_EQUAL.  Default ON;
    // turn off to measure baseline shadow pass cost without the
    // optimization.  See csm_silhouette_prepass.mesh for details.
    bool csm_silhouette_prepass_enabled_ = true;

public:
    // ─── CSM drawable-shadow draw mode ────────────────────────────────────
    // Toggles how the drawable shadow path amplifies geometry across the
    // CSM_CASCADE_COUNT cascades.  All three modes draw the same set of
    // shadow casters; they differ in HOW the per-cascade VP is applied:
    //
    //   kRegular         — N separate single-layer draws, one per cascade.
    //                      VS transforms with that cascade's VP.  No GS,
    //                      no mesh shader.  Baseline; portable, slowest in
    //                      the typical case but on Blackwell may end up
    //                      competitive vs GS amplification.
    //   kGeometryShader  — Single layered draw; base_depthonly_csm.geom
    //                      broadcasts each tri to every cascade layer.
    //                      Historical default; cheap on Ada, expensive on
    //                      Blackwell where GS is emulated on top of the
    //                      mesh-shader frontend.
    //   kMeshShader      — Single layered draw; task+mesh shaders perform
    //                      the per-cascade amplification natively.
    //                      Mirrors the cluster shadow path which already
    //                      uses task+mesh and benchmarks well across
    //                      Ada / Blackwell.
    //
    // Exposed via the Shadow menu so the user can A/B in-app.
    enum class CsmDrawMode : int {
        kRegular        = 0,
        kGeometryShader = 1,
        kMeshShader     = 2,
    };

private:
    CsmDrawMode csm_draw_mode_ = CsmDrawMode::kMeshShader;
    bool turn_on_airflow_ = false;
    uint32_t debug_draw_type_ = 0;
    // PBR / forward-pass debug visualisation; values match
    // DEBUG_RENDER_MODE_* in global_definition.glsl.h.  Driven by the
    // "Render Debug" combo in the menu bar.
    int debug_render_mode_ = 0;
    // Hi-Z mip level chosen for the DEBUG_RENDER_MODE_HIZ visualisation.
    // 0 = half-res (richest detail), higher = increasingly down-sampled.
    // Clamped to the actual pyramid mip count by the menu UI.  Packed into
    // camera_info.input_features bits 24..27 by drawScene each frame.
    int hiz_debug_mip_ = 0;
    // Forward vs deferred rendering toggle.  Application syncs its
    // own deferred_rendering_enabled_ from this each frame.  Defaults
    // to ON so the new G-buffer + compute resolve path is the
    // out-of-the-box experience; flip OFF in the Render Debug menu
    // to A/B against the legacy forward bindless path.
    bool deferred_rendering_ = true;
    float air_flow_strength_ = 50.0f;
    float water_flow_strength_ = 1.0f;
    float light_ext_factor_ = 0.004f;
    float view_ext_factor_ = 0.10f;
    float view_ext_exponent_ = 1.0f;
    float cloud_ambient_intensity_ = 1.0f;
    float cloud_phase_intensity_ = 0.5f;
    float cloud_moist_to_pressure_ratio_ = 0.05f;
    float global_flow_dir_ = 85.0f;
    float global_flow_speed_ = 0.5f;
    float cloud_noise_thresold_ = 0.3f;
    float cloud_noise_scrolling_speed_ = 6.519f;
    float cloud_noise_scale_[2] = { 4.45f, 4.387f };
    float cloud_noise_weight_[2][4] =
        { {2.532f, 1.442f, 0.7f, 0.9f},
        {0.2f, 0.2f, 0.2f, 0.2f} };

    glsl::WeatherControl weather_controls_;

    // Title-screen config loaded from XML.
    TitleScreenConfig title_config_;
    GameState game_state_ = GameState::TitleScreen;

    // Meshes requested by "New Game" — consumed by application each frame.
    std::vector<std::string> new_game_mesh_requests_;
    bool new_game_requested_ = false;  // one-shot flag

    ImTextureID rt_texture_id_;
    ImTextureID main_texture_id_;

    // Fantasy-twilight background image — loaded once in the
    // constructor from assets/ui/fantasy_bg.png and rendered
    // full-screen at the bottom of the ImGui draw order so it
    // sits behind the game scene whenever the scene itself
    // doesn't cover the framebuffer (e.g. during async mesh
    // loads, or any part of the view that isn't opaque).
    // bg_texture_info_ owns the Vulkan image + memory + view;
    // bg_texture_id_ is the ImGui-side descriptor handle.
    std::shared_ptr<renderer::TextureInfo> bg_texture_info_;
    ImTextureID bg_texture_id_ = ImTextureID(0);
    bool bg_enabled_ = true;  // set to false once scene meshes are loaded

    // Analog clock face overlay — loaded from assets/ui/clock_face.png.
    std::shared_ptr<renderer::TextureInfo> clock_tex_info_;
    ImTextureID clock_tex_id_ = ImTextureID(0);

    // Device + sampler kept for re-registering textures / cleanup.
    std::shared_ptr<renderer::Device> device_;
    std::shared_ptr<renderer::Sampler> sampler_;

    // Stars detected in the background image at load time.
    // Positions are normalised [0,1] so they scale with any viewport.
    struct DetectedStar {
        float nx, ny;       // normalised position in the image
        float radius;       // approximate radius in pixels (at 1920 wide)
        float brightness;   // 0..1 peak brightness
        float speed;        // twinkle frequency (Hz)
        float phase;        // twinkle phase offset
    };
    std::vector<DetectedStar> detected_stars_;

    // CSM debug visualisation
    bool show_csm_debug_ = false;
    std::array<ImTextureID, CSM_CASCADE_COUNT> csm_debug_tex_ids_ = {};

    // RVT pool viewer — direct visualisation of the four 4096² layer
    // pool textures.  Useful for verifying that registration is
    // actually copying texel data into the pool (each populated page
    // shows up as a 128×128 patch within the atlas; empty slots stay
    // black).  Tex-ids are assigned once at startup by
    // RealWorldApplication::registerVtPoolImTextureIds().
    bool show_vt_pool_debug_ = false;
    std::array<ImTextureID, 4> vt_pool_tex_ids_ = {};

    // Collision-mesh debug visualisation. When ON the application
    // skips the regular object forward pass and draws the static
    // CollisionWorld with hashed-per-triangle colours instead. Also
    // toggleable via F1 in the application's key callback (which
    // forwards through toggleCollisionDebug() below).
    bool show_collision_debug_ = false;

    // ── Mesh-Category inspector window ───────────────────────────────
    // Snapshot of the LLM classifier's verdicts pushed by application
    // via setMaterialCategorySnapshot() once the async classify
    // finishes.  Each entry is (name, MeshCategory cast to uint32_t)
    // — uint32_t in the pair instead of the enum keeps menu.h free
    // of a collision_mesh.h include (the colour switch in draw()
    // hardcodes the same RGB table to stay in lockstep with the
    // shader, so we don't actually need the enum type here).
    bool show_mesh_category_inspector_ = false;
    bool mesh_category_snapshot_valid_ = false;
    std::vector<std::pair<std::string, uint32_t>>
        mesh_category_materials_;
    std::vector<std::pair<std::string, uint32_t>>
        mesh_category_objects_;
    // Search box text — filters both tables by case-insensitive
    // substring match.  Cleared on first frame the inspector opens.
    char mesh_category_filter_[64] = {0};

    // ── Classifier status banner (always-on top-right overlay) ───────
    // Application pushes the current classifier state every frame via
    // setClassifierStatus().  draw() then paints a small text label
    // at the top of the viewport ("LLM: pending (12s)", "LLM: ready
    // 193/193", "LLM: failed") so the user has visual confirmation
    // even when the console output is hidden / scrolled past.  Values
    // mirror what the [mat.cls] log lines say.
    //
    // The enum has to be PUBLIC because application.cpp names it
    // when calling setClassifierStatus(); the storage fields below
    // stay private.
public:
    enum class ClassifierStatus : int {
        Idle    = 0,  // not kicked off yet
        Pending = 1,  // worker running
        Ready   = 2,  // succeeded
        Failed  = 3,  // future returned false
    };
private:
    ClassifierStatus classifier_status_ = ClassifierStatus::Idle;
    float            classifier_elapsed_s_ = 0.0f;
    int              classifier_mats_done_ = 0;
    int              classifier_objs_done_ = 0;
    // Live byte-progress counters published by the worker thread
    // and pushed in through setClassifierStatus() each frame.
    // Bytes-received / estimated-total drives the on-screen
    // ProgressBar.  estimated_total is a heuristic baked from
    // collected name counts in application.cpp — overshoots cap at
    // 100 % so the bar never reads ridiculous.
    size_t           classifier_bytes_received_ = 0;
    size_t           classifier_bytes_estimated_total_ = 0;

    // ── Collision-mesh build progress (second bar, below the LLM one) ─
    // The collision world is built incrementally across frames (a batch
    // of primitives per frame) once the classifier has landed.
    // application.cpp pushes done/total each frame via
    // setCollisionBuildStatus(); draw() paints a ProgressBar pinned
    // directly under the classifier bar.  meshes_ is the running count
    // of Floor collision meshes actually added to the world.
public:
    enum class CollisionBuildStatus : int {
        Idle     = 0,  // not started (classifier not done yet)
        Building = 1,  // incremental build in progress
        Done     = 2,  // every queued primitive processed
    };
private:
    CollisionBuildStatus collision_build_status_ = CollisionBuildStatus::Idle;
    size_t               collision_build_done_   = 0;
    size_t               collision_build_total_  = 0;
    size_t               collision_build_meshes_ = 0;

    // Which CollisionShape mode application.cpp should pass to
    // CollisionMesh::buildFromDrawablePrimitive when (re)building
    // the collision world for visualisation. Three user-facing
    // options live in the Physics submenu; the application maps
    // them onto the engine's CollisionShape enum at build time.
    // Default is Simplified (QEM-decimated triangles) — keeps the
    // source silhouette so capsule physics slides naturally along
    // thin walls and stairs.  Gaps can appear where neighbouring
    // primitives don't share vertices (see the 3x3x3 weld in
    // collision_mesh.cpp); switch to Volume for a gap-free shell
    // when inspecting coverage of low-detail props.
public:
    enum class CollisionDebugShape : int {
        Original   = 0,   // CollisionShape::None
        Simplified = 1,   // CollisionShape::Decimate (post-weld)
        Volume     = 2,   // CollisionShape::VoxelCube
    };
private:
    CollisionDebugShape collision_debug_shape_ =
        CollisionDebugShape::Simplified;

    // ── Skeleton render-debug mode ───────────────────────────────────
    // Set from the "Render Debug -> Skeleton view" submenu.
    // CharacterOnly  : the skinned mesh draws; the 19 bone cubes hide.
    // BoneWithCharacter : both draw (useful to verify the cubes line up
    //                  with the rig as it animates).
    // BoneOnly       : only the 19 cubes draw; the mesh hides (lets you
    //                  inspect the skeleton in isolation).
    // Defaults to CharacterOnly so the regular gameplay view stays
    // clean unless the user opts in to skeleton overlays.
    // Enum is PUBLIC so external code (application.cpp) can name
    // engine::ui::Menu::SkeletonDebugMode::*; storage stays private.
public:
    enum class SkeletonDebugMode : int {
        CharacterOnly     = 0,
        BoneWithCharacter = 1,
        BoneOnly          = 2,
    };
private:
    SkeletonDebugMode skeleton_debug_mode_ = SkeletonDebugMode::CharacterOnly;
    // Set by setCollisionDebugShape() whenever the menu choice
    // changes; the application clears it after rebuilding the
    // collision world to match.
    bool collision_world_dirty_ = false;

    // ── Isolate-debug slider ─────────────────────────────────────────
    // When enabled, the collision debug draw shows ONLY the single mesh
    // at collision_isolate_index_ (scrub it to find a broken/missing
    // mesh, then read its index + identity off the overlay).  The app
    // pushes the live mesh count (for the slider range) and the
    // isolated mesh's identity string each frame.
    bool        collision_isolate_enabled_ = false;
    int         collision_isolate_index_   = 0;
    size_t      collision_mesh_count_      = 0;
    std::string collision_isolate_info_;

    // Name of the scene object currently under the mouse cursor (set each
    // frame by the application's ray-pick; empty = nothing hovered).  Drawn
    // as a small tooltip at the mouse position in draw().
    std::string hover_object_name_;

    // ---- Time-of-day -------------------------------------------------------
    // Hours in [0, 24) in the player's **local timezone**.  Initialised
    // at startup from localtime_s / localtime_r so the sun position and
    // the clock overlay both reflect the player's local time of day.
    // The hour-angle formula in Skydome::update treats this value as
    // local solar time, so local noon (12.0) puts the sun at its highest
    // point for the configured latitude/longitude.
    //
    // When the user yanks the slider by more than tod_jump_threshold_
    // in a single frame we treat it as a "skip" (debug "go to dawn"
    // button etc.) and the application will reset the sky / IBL mini-
    // buffers so they re-bootstrap instead of EMA-blending toward the
    // new lighting over many seconds.
    float tod_hours_      = 0.0f;  // overwritten in constructor from local wall clock
    float tod_prev_hours_ = 0.0f;
    bool  tod_auto_advance_ = true;      // tick forward with real time
    // Speed factor: game time / real time.  5.0 means 1 real-second
    // advances the in-game clock by 5 game-seconds, so a full 24-hour
    // cycle takes ~4.8 real hours.  Slider in the Skydome window
    // exposes this directly; advanceTimeOfDay() converts to game-hours
    // internally (game-seconds-added / 3600).
    float tod_advance_speed_ = 5.0f;
    float tod_jump_threshold_ = 0.5f;    // hours

    // ---- IBL / sky cubemap debug visualisation ----------------------------
    // Set from the application after IblCreator/Skydome textures exist.
    // Every cubemap is exposed as a 2D array of ImTextureIDs indexed
    // [mip][face], so each debug row gets its own mip slider.  Cubemaps
    // that only have one mip (sky mini-buffer, IBL diffuse) leave higher
    // mip rows zeroed and the per-row slider is range-clamped to [0, 0].
    //
    // ImTextureIDs are obtained from `TextureInfo::surface_views[mip][face]`
    // (per-mip per-face 2D ImageViews populated by createCubemapTexture
    // for use_as_framebuffer cubemaps) wrapped via `Helper::addImTextureID`.
    bool show_ibl_debug_ = false;
public:
    // Public so the application can declare matching arrays when
    // gathering per-mip ImTextureIDs to feed the setters below.
    static constexpr int kIblDebugMaxMips = 10;
    using IblDebugFaceArray = std::array<ImTextureID, 6>;
    using IblDebugMipFaceArray = std::array<IblDebugFaceArray, kIblDebugMaxMips>;
private:

    IblDebugMipFaceArray envmap_face_mip_tex_ids_      = {};
    IblDebugMipFaceArray mini_envmap_face_mip_tex_ids_ = {};
    IblDebugMipFaceArray diffuse_face_mip_tex_ids_     = {};
    IblDebugMipFaceArray specular_face_mip_tex_ids_    = {};
    IblDebugMipFaceArray sheen_face_mip_tex_ids_       = {};

    // Number of mips the application registered for each cubemap.  Caps
    // each row's mip slider so we don't index zeroed slots.
    int ibl_debug_envmap_num_mips_      = 1;
    int ibl_debug_mini_envmap_num_mips_ = 1;
    int ibl_debug_diffuse_num_mips_     = 1;
    int ibl_debug_specular_num_mips_    = 1;
    int ibl_debug_sheen_num_mips_       = 1;

    // Currently-selected mip per row (driven by the ImGui sliders).
    int ibl_debug_envmap_mip_      = 0;
    int ibl_debug_mini_envmap_mip_ = 0;
    int ibl_debug_diffuse_mip_     = 0;
    int ibl_debug_specular_mip_    = 0;
    int ibl_debug_sheen_mip_       = 0;

    float ibl_debug_thumb_size_   = 96.0f;
    // Exposure multiplier applied to every thumbnail via ImGui::Image's
    // tint colour.  Atmospheric / IBL data is HDR linear; ImGui doesn't
    // tonemap, so dim mid-tones (~0.05 linear) display as near-black and
    // small step artifacts in atmosphere LUT raymarching can look like
    // concentric rings.  Boosting this scales the displayed value into
    // the visible sRGB range.  Stored as linear scale (1.0 = no change).
    float ibl_debug_exposure_     = 1.0f;

    // ---- Dynamic camera-positioned reflection cubemap debug viewer --------
    // The DynamicCubemap probe captures one face per frame at the main
    // camera position and reprojects the others.  The viewer below shows
    // each ping-pong buffer's 6 face slices as a 6-thumbnail strip so we
    // can visually verify that fresh captures look right and that the
    // depth-aware reprojection is producing coherent output as the
    // camera moves.  ImTextureIDs are wrapped from the per-face 2D layer
    // views exposed by DynamicCubemap::getColorFaceView.
    bool show_dynamic_cube_debug_ = false;

    // Toggle in-scene probe debug-draw — when on, the application
    // renders a small icosphere at every grid probe position via
    // AmbientProbeSystem::drawDebug, so you can see where probes are
    // placed and what irradiance each one is currently emitting.
    bool show_probe_debug_ = false;

    // Hi-Z pyramid (last-frame depth in cluster Z-cull) debug viewer.
    // Each mip is registered as an ImTextureID by the application.  The
    // viewer shows them as a horizontal strip — left = mip 0 (full res),
    // right = top mip (1×1).  Lets us verify that the per-frame Hi-Z
    // build is actually populating the pyramid with sane values before
    // chasing further bugs in the cluster cull's sample math.
    bool show_hiz_debug_ = false;
    std::vector<ImTextureID> hiz_debug_tex_ids_;
    // [ping_pong_idx][face_idx] — populated by setDynamicCubeFaceTextureIds().
    std::array<IblDebugFaceArray, 2> dynamic_cube_face_tex_ids_ = {};
    // Live frame state mirrored from DynamicCubemap so the viewer can
    // pick the right ping-pong index AND highlight which face was just
    // freshly rendered (vs reprojected) this frame.  Set every frame
    // by setDynamicCubeFrameInfo() before draw().
    int  dynamic_cube_current_read_idx_ = 0;
    int  dynamic_cube_current_face_     = 0;
    uint64_t dynamic_cube_frame_index_  = 0;
    glm::vec3 dynamic_cube_face_capture_pos_[6] = {};
    float dynamic_cube_thumb_size_ = 128.0f;
    float dynamic_cube_exposure_   = 1.0f;

    std::shared_ptr<ChatBox> chat_box_;

    // Optional GPU profiler — set from application after init.
    engine::helper::GpuProfiler* gpu_profiler_ = nullptr;

    // Plugin system — set from application after init.
    plugins::PluginManager* plugin_manager_ = nullptr;

    // Optional async mesh loader — set from application after init.
    // Used by the HUD spinner overlay (see draw()) to show which
    // meshes are still loading. Non-owning.
    engine::game_object::MeshLoadTaskManager* mesh_load_task_manager_ = nullptr;

    // Optional SSAO — set from application after init.
    engine::scene_rendering::SSAO* ssao_ = nullptr;

    // Optional ClusterRenderer — set from application after init.
    engine::scene_rendering::ClusterRenderer* cluster_renderer_ = nullptr;
    bool show_smart_mesh_window_ = false;

    // Optional VirtualTextureManager — set from application after
    // init.  The pool debug viewer reads getSlotStatusGrid() from
    // this each frame to draw the per-slot activity heatmap.
    engine::scene_rendering::VirtualTextureManager* vt_manager_ = nullptr;

public:
    // ── Editor (UE-style docked layout) ───────────────────────────────
    // Feed of live scene objects for the Outliner/Details panels, set by
    // the application each frame.  editor_layout_built_ latches the one-time
    // DockBuilder default arrangement.  content_dir_ is the Content Browser's
    // current folder (relative to the working dir).
    std::vector<EditorSceneObject> editor_objects_;
    int          editor_selected_   = -1;
    bool         editor_layout_built_ = false;
    float        outliner_list_h_   = 0.0f;   // draggable list/details split (px)
    bool         editor_enabled_    = false;  // editor UI off unless --editor
    std::string  content_dir_       = "assets";
    // Screen-space rect (main-viewport pixels) of the dockspace central node
    // = the 3D Viewport region.  The application reads this each frame to set
    // the camera aspect and (next step) size the offscreen scene render.
    glm::vec2    viewport_pos_      = glm::vec2(0.0f);
    glm::vec2    viewport_size_     = glm::vec2(0.0f);
    bool         viewport_valid_    = false;
    void setSceneObjects(std::vector<EditorSceneObject> objs) {
        editor_objects_ = std::move(objs);
    }
    void setEditorEnabled(bool e) { editor_enabled_ = e; }
    bool isEditorEnabled() const  { return editor_enabled_; }
    bool      isViewportValid() const { return viewport_valid_ && editor_enabled_; }
    glm::vec2 getViewportPos()  const { return viewport_pos_; }
    glm::vec2 getViewportSize() const { return viewport_size_; }
    // Aspect of the active viewport: the editor panel in editor mode, else the
    // full window (so in-game framing matches the editor viewport's handling).
    float     getViewportAspect() const {
        ImVec2 p, s, c; getViewportScreenRect(p, s, c);
        return (s.y > 1.0f) ? (s.x / s.y) : 1.0f;
    }
private:
    void drawEditorDockSpace();   // host window + default layout
    void drawOutlinerPanel();   // object list (top) + selected-object details (below)
    void drawDetailsContent();  // details widgets, drawn inline under the list
    void drawContentBrowserPanel();
    void drawOutputPanel();   // UE5-style console Output Log (bottom tab)
    // Absolute screen rect/centre of the editor Viewport (central dock node);
    // falls back to the full main viewport when the editor is inactive.  Used
    // to keep loading bars / dialog boxes inside the viewport, not over panels.
    void   getViewportScreenRect(ImVec2& pos, ImVec2& size, ImVec2& center) const;
    ImVec2 viewportCenter() const;
public:

    void setBackgroundEnabled(bool enabled) { bg_enabled_ = enabled; }
    bool isBackgroundEnabled() const { return bg_enabled_; }
    void setGpuProfiler(engine::helper::GpuProfiler* profiler) {
        gpu_profiler_ = profiler;
    }
    void setPluginManager(plugins::PluginManager* pm) {
        plugin_manager_ = pm;
    }
    void setMeshLoadTaskManager(engine::game_object::MeshLoadTaskManager* m) {
        mesh_load_task_manager_ = m;
    }
    void setSSAO(engine::scene_rendering::SSAO* s) {
        ssao_ = s;
    }
    void setClusterRenderer(engine::scene_rendering::ClusterRenderer* cr) {
        cluster_renderer_ = cr;
    }
    void setVtManager(engine::scene_rendering::VirtualTextureManager* m) {
        vt_manager_ = m;
    }

    // Game state accessors.
    GameState getGameState() const { return game_state_; }
    void setGameState(GameState s) { game_state_ = s; }

    // Returns true once after "New Game" is clicked, then resets.
    bool consumeNewGameRequest() {
        if (new_game_requested_) {
            new_game_requested_ = false;
            return true;
        }
        return false;
    }
    const std::vector<std::string>& getNewGameMeshes() const {
        return new_game_mesh_requests_;
    }
    Menu(
        GLFWwindow* window,
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::Instance>& instance,
        const renderer::QueueFamilyList& queue_family_list,
        const renderer::SwapChainInfo& swap_chain_info,
        const std::shared_ptr<renderer::Queue>& graphics_queue,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::Sampler>& sampler,
        const std::shared_ptr<renderer::ImageView>& rt_image_view,
        const std::shared_ptr<renderer::ImageView>& main_image_view);

    std::vector<std::string> getToLoadGltfNamesAndClear() {
        auto result = to_load_gltf_names_;
        to_load_gltf_names_.clear();
        return result;
    }

    // (getToLoadPlayerNameAndClear / spawn_gltf_name_ were removed
    // with the legacy "Spawn player" menu item — the player is now
    // eager-loaded once at startup with a fixed asset.)

    // ── Player debug overlay state ────────────────────────────────────
    // Application stashes the player's world position + camera view-
    // projection here each frame; the menu's draw() projects the world
    // point onto the screen and paints a red marker so the user can
    // visually locate the character even if the 3D draw is broken.
    // has_player_debug_info_ is false until the player is spawned,
    // keeping the overlay invisible during loading.
    void setPlayerDebugInfo(
        bool has_player,
        const glm::vec3& world_pos,
        const glm::mat4& view_proj,
        const glm::vec3& cam_pos) {
        has_player_debug_info_ = has_player;
        player_world_pos_      = world_pos;
        player_view_proj_      = view_proj;
        player_cam_pos_        = cam_pos;
        // Don't invalidate the bbox here — overloaded version sets it
        // and we want to allow callers to drive the box independently.
    }

    // Extended overload that ALSO carries the player's world-space
    // AABB so the menu's overlay can draw a wireframe box around the
    // character.  Pass bbox_valid=false when the bbox isn't ready yet
    // (e.g. async load hasn't finished) — the overlay falls back to
    // just the position marker + HUD line in that case.
    void setPlayerDebugInfo(
        bool has_player,
        const glm::vec3& world_pos,
        const glm::mat4& view_proj,
        const glm::vec3& cam_pos,
        bool bbox_valid,
        const glm::vec3& bbox_min,
        const glm::vec3& bbox_max) {
        setPlayerDebugInfo(has_player, world_pos, view_proj, cam_pos);
        player_bbox_valid_ = bbox_valid;
        player_bbox_min_   = bbox_min;
        player_bbox_max_   = bbox_max;
    }

    // ── Material / Object → MeshCategory inspector ────────────────────
    // Application pushes a snapshot of the LLM classifier's tables
    // here ONCE, the first frame after the async classifyAll() lands.
    // The menu then renders an ImGui inspector window when
    // show_mesh_category_inspector_ is true (toggled from the Render
    // Debug menu) — two scrollable tables, one for materials and one
    // for objects, each row showing a colour swatch + the name + the
    // category tag.  Sorted alphabetically by name within each table.
    //
    // The pair vectors are stored, not the source unordered_maps, so
    // we keep deterministic ordering across draws and don't depend on
    // hash-map iteration order changing between frames.
    void setMaterialCategorySnapshot(
        const std::vector<std::pair<std::string, uint32_t>>& materials,
        const std::vector<std::pair<std::string, uint32_t>>& objects) {
        mesh_category_materials_ = materials;
        mesh_category_objects_   = objects;
        // Sort alphabetically so the user can scroll through and the
        // order doesn't depend on hash-map iteration.
        auto by_name = [](const auto& a, const auto& b) {
            return a.first < b.first;
        };
        std::sort(mesh_category_materials_.begin(),
                  mesh_category_materials_.end(), by_name);
        std::sort(mesh_category_objects_.begin(),
                  mesh_category_objects_.end(), by_name);
        mesh_category_snapshot_valid_ = true;
    }
    bool meshCategorySnapshotValid() const {
        return mesh_category_snapshot_valid_;
    }

    // Per-frame push of the classifier's current state so the banner
    // can render without each frame re-reading anything from the
    // async machinery.  Application calls this from the same place
    // it polls s_mat_classifier_future.
    void setClassifierStatus(
        ClassifierStatus status,
        float elapsed_s,
        int mats_done,
        int objs_done,
        size_t bytes_received,
        size_t bytes_estimated_total) {
        classifier_status_   = status;
        classifier_elapsed_s_ = elapsed_s;
        classifier_mats_done_ = mats_done;
        classifier_objs_done_ = objs_done;
        classifier_bytes_received_ = bytes_received;
        classifier_bytes_estimated_total_ = bytes_estimated_total;
    }

    // Per-frame push of the incremental collision-build progress so the
    // second (lower) progress bar can render.  Called from the build
    // block in application.cpp every frame while the world is being
    // generated, and once more with Done when the last primitive lands.
    void setCollisionBuildStatus(
        CollisionBuildStatus status,
        size_t done,
        size_t total,
        size_t meshes) {
        collision_build_status_ = status;
        collision_build_done_   = done;
        collision_build_total_  = total;
        collision_build_meshes_ = meshes;
    }

    // One-shot "snap the player to where the camera is looking" request.
    // The menu sets this when the user picks "Reset player position";
    // application.cpp reads it each frame, clears it, and (if set) calls
    // PlayerController::spawnAt() with the current camera + facing.  Used
    // as the recovery path when the player ends up off-screen — either
    // because the auto-spawn block fired before the player asset was
    // fully ready, or because the user wandered the camera away from
    // wherever the character actually stands.
    bool consumeResetPlayerPosition() {
        bool v = reset_player_position_requested_;
        reset_player_position_requested_ = false;
        return v;
    }

    // Live foot-IK params edited by the Physics > Foot IK menu; the
    // application reads these each frame and applies them to the
    // PlayerController.  setFootIkPelvisDrop feeds the read-only
    // pelvis-drop value shown beneath the sliders.
    const FootIkParams& footIkParams() const { return foot_ik_params_; }
    void setFootIkPelvisDrop(float v) { foot_ik_pelvis_drop_live_ = v; }

    // CSM debug visualisation
    inline bool showCsmDebug() const { return show_csm_debug_; }
    void setCsmDebugTextureIds(const std::array<ImTextureID, CSM_CASCADE_COUNT>& ids) {
        csm_debug_tex_ids_ = ids;
    }

    // RVT pool viewer.
    inline bool showVtPoolDebug() const { return show_vt_pool_debug_; }
    inline void setShowVtPoolDebug(bool v) { show_vt_pool_debug_ = v; }
    void setVtPoolTextureIds(const std::array<ImTextureID, 4>& ids) {
        vt_pool_tex_ids_ = ids;
    }

    // Collision-mesh debug visualisation. The menu owns the canonical
    // state; the application reads it each frame to decide whether to
    // run the normal forward pass or CollisionWorld::drawDebug.
    inline bool isCollisionDebugOn() const { return show_collision_debug_; }
    inline void setCollisionDebug(bool on) { show_collision_debug_ = on; }
    inline void toggleCollisionDebug() {
        show_collision_debug_ = !show_collision_debug_;
    }

    // Collision-shape selector accessors.
    // Render-debug skeleton view: bone-only / both / character-only.
    // Application reads this each frame and toggles per-drawable
    // visibility on the player mesh and the 19 bone-marker cubes.
    inline SkeletonDebugMode skeletonDebugMode() const {
        return skeleton_debug_mode_;
    }
    inline void setSkeletonDebugMode(SkeletonDebugMode m) {
        skeleton_debug_mode_ = m;
    }

    inline CollisionDebugShape collisionDebugShape() const {
        return collision_debug_shape_;
    }
    inline void setCollisionDebugShape(CollisionDebugShape s) {
        if (collision_debug_shape_ != s) {
            collision_debug_shape_ = s;
            collision_world_dirty_ = true;
        }
    }
    inline bool collisionWorldDirty() const {
        return collision_world_dirty_;
    }

    // Isolate-debug slider accessors.  collisionIsolateIndex() is only
    // meaningful when collisionIsolateEnabled() is true.  The app pushes
    // the mesh count (slider range) and the isolated mesh's identity.
    inline bool collisionIsolateEnabled() const {
        return collision_isolate_enabled_;
    }
    inline int collisionIsolateIndex() const {
        return collision_isolate_index_;
    }
    inline void setCollisionMeshCount(size_t n) {
        collision_mesh_count_ = n;
    }
    inline void setCollisionIsolateInfo(const std::string& s) {
        collision_isolate_info_ = s;
    }
    // Name of the object under the mouse cursor (ray-picked by the
    // application).  Pass "" to clear.  Rendered at the cursor in draw().
    inline void setHoverObjectName(const std::string& s) {
        hover_object_name_ = s;
    }
    // Step the isolated mesh index by `delta` (Left/Right arrow hotkeys).
    // Auto-enables isolate mode so the keys work even when the menu /
    // slider is hidden, and clamps to [0, mesh_count-1].
    inline void stepCollisionIsolate(int delta) {
        collision_isolate_enabled_ = true;
        collision_isolate_index_ += delta;
        if (collision_isolate_index_ < 0) collision_isolate_index_ = 0;
        if (collision_mesh_count_ > 0) {
            const int max_idx = static_cast<int>(collision_mesh_count_) - 1;
            if (collision_isolate_index_ > max_idx)
                collision_isolate_index_ = max_idx;
        }
    }
    inline void clearCollisionWorldDirty() {
        collision_world_dirty_ = false;
    }

    // ---- TOD (time-of-day) accessors --------------------------------------
    // Hours in [0, 24) the application should feed to Skydome::update.
    inline float getTimeOfDayHours() const { return tod_hours_; }
    inline void  setTimeOfDayHours(float h) {
        tod_hours_ = std::fmod(h + 24.0f, 24.0f);
    }
    // Returns true once when the user has yanked the slider far enough
    // for the application to call resetMiniBuffer() on Skydome / IblCreator.
    // Caller-driven: this method internally advances the "previous" value
    // each call, so the application should call it exactly once per frame.
    bool consumeTodJump() {
        bool jumped =
            std::fabs(tod_hours_ - tod_prev_hours_) > tod_jump_threshold_;
        // Wrap-around: a small positive change like 23.9 -> 0.1 (== 0.2h)
        // shouldn't count as a jump.
        float wrapped = std::fabs(tod_hours_ - tod_prev_hours_);
        if (wrapped > 12.0f) {
            wrapped = 24.0f - wrapped;
            jumped = wrapped > tod_jump_threshold_;
        }
        tod_prev_hours_ = tod_hours_;
        return jumped;
    }
    // Optional: tick TOD forward each frame when auto-advance is on.
    // tod_advance_speed_ is "game-time / real-time" (e.g. 100.0 means
    // game runs 100x real time).  Convert to added-game-hours by
    // multiplying by real-seconds-elapsed and dividing by 3600.
    void advanceTimeOfDay(float real_delta_seconds) {
        if (!tod_auto_advance_) return;
        const float game_hours_added =
            tod_advance_speed_ * real_delta_seconds / 3600.0f;
        tod_hours_ = std::fmod(
            tod_hours_ + game_hours_added + 24.0f,
            24.0f);
    }

    // ---- IBL / sky debug texture binding ----------------------------------
    // The application calls these once after the IBL/Sky textures exist.
    // Each setter takes the full per-mip per-face ImTextureID array and a
    // num_mips count; the menu's per-row slider is range-clamped to
    // [0, num_mips - 1].
    void setEnvmapFaceMipTextureIds(
        const IblDebugMipFaceArray& ids, int num_mips) {
        envmap_face_mip_tex_ids_ = ids;
        ibl_debug_envmap_num_mips_ = num_mips;
    }
    void setMiniEnvmapFaceMipTextureIds(
        const IblDebugMipFaceArray& ids, int num_mips) {
        mini_envmap_face_mip_tex_ids_ = ids;
        ibl_debug_mini_envmap_num_mips_ = num_mips;
    }
    void setIblDiffuseFaceMipTextureIds(
        const IblDebugMipFaceArray& ids, int num_mips) {
        diffuse_face_mip_tex_ids_ = ids;
        ibl_debug_diffuse_num_mips_ = num_mips;
    }
    void setIblSpecularFaceMipTextureIds(
        const IblDebugMipFaceArray& ids, int num_mips) {
        specular_face_mip_tex_ids_ = ids;
        ibl_debug_specular_num_mips_ = num_mips;
    }
    void setIblSheenFaceMipTextureIds(
        const IblDebugMipFaceArray& ids, int num_mips) {
        sheen_face_mip_tex_ids_ = ids;
        ibl_debug_sheen_num_mips_ = num_mips;
    }

    // ---- Hi-Z pyramid debug viewer ----------------------------------------
    // Application supplies one ImTextureID per Hi-Z pyramid mip (0..N-1).
    // The viewer renders them in a horizontal strip so we can verify the
    // per-frame Hi-Z build dispatch is producing sane values across the
    // entire mip chain.  Black = near, white = far (raw Vulkan depth).
    void setHiZDebugTextureIds(std::vector<ImTextureID> ids) {
        hiz_debug_tex_ids_ = std::move(ids);
    }
    // Read by the application so it can run the Hi-Z BUILD dispatch
    // even when the cull-side toggle is off, letting the user inspect
    // the pyramid contents independently of the cull behaviour.
    bool isHiZDebugOn() const { return show_hiz_debug_; }

    // ---- Dynamic-cubemap debug viewer ------------------------------------
    // Application calls this once at startup with both ping-pong buffers'
    // 6 face ImTextureIDs already wrapped via Helper::addImTextureID.
    void setDynamicCubeFaceTextureIds(
        const std::array<IblDebugFaceArray, 2>& ids) {
        dynamic_cube_face_tex_ids_ = ids;
    }
    // Application calls this every frame so the viewer can pick the
    // current ping-pong slice and highlight the freshly-rendered face.
    void setDynamicCubeFrameInfo(
        int current_read_idx, int current_face,
        uint64_t frame_index,
        const glm::vec3 face_capture_pos[6]) {
        dynamic_cube_current_read_idx_ = current_read_idx;
        dynamic_cube_current_face_     = current_face;
        dynamic_cube_frame_index_      = frame_index;
        for (int f = 0; f < 6; ++f) {
            dynamic_cube_face_capture_pos_[f] = face_capture_pos[f];
        }
    }

    // Probe in-scene debug-draw toggle.
    inline bool showProbeDebug() const { return show_probe_debug_; }

    inline bool isShadowPassTurnOff() const {
        return turn_off_shadow_pass_;
    }

    inline bool isCsmSilhouettePrepassEnabled() const {
        return csm_silhouette_prepass_enabled_;
    }

    // Current selected CSM drawable-shadow draw-mode (see enum comment above).
    inline CsmDrawMode getCsmDrawMode() const { return csm_draw_mode_; }
    inline void setCsmDrawMode(CsmDrawMode m) { csm_draw_mode_ = m; }

    // Render-debug visualisation for the forward + cluster bindless paths.
    // Returned value is one of DEBUG_RENDER_MODE_* (defined in
    // global_definition.glsl.h).  application.cpp packs it into the upper
    // bits of camera_info.input_features each frame.
    inline int getDebugRenderMode() const { return debug_render_mode_; }

    // Forward vs deferred toggle — read by application drawScene to
    // route the cluster opaque pass through the G-buffer + compute
    // resolve (true) or the legacy forward bindless pipeline (false).
    inline bool isDeferredRendering() const { return deferred_rendering_; }
    inline void setDeferredRendering(bool on) { deferred_rendering_ = on; }

    // Hi-Z mip selector for DEBUG_RENDER_MODE_HIZ.  Clamped externally to
    // the pyramid's actual mip count.
    inline int  getHiZDebugMip() const { return hiz_debug_mip_; }
    inline void setHiZDebugMip(int m)  { hiz_debug_mip_ = m; }

    inline bool isWaterPassTurnOff() {
        return turn_off_water_pass_;
    }

    inline bool isGrassPassTurnOff() {
        return turn_off_grass_pass_;
    }

    inline bool isRayTracingTurnOff() {
        return turn_off_ray_tracing_;
    }

    inline bool isVolumeMoistTurnOff() {
        return turn_off_volume_moist_;
    }

    inline bool isAirfowOn() {
        return turn_on_airflow_;
    }

    inline uint32_t getDebugDrawType() {
        return debug_draw_type_;
    }

    inline const glsl::WeatherControl& getWeatherControls() {
        return weather_controls_;
    }

    inline const float getLightExtFactor() const {
        return light_ext_factor_ / 1000.0f;
    }

    inline const float getViewExtFactor() const {
        return view_ext_factor_ / 1000.0f;
    }

    inline const float getViewExtExponent() const {
        return view_ext_exponent_;
    }

    inline const float getCloudAmbientIntensity() const {
        return cloud_ambient_intensity_;
    }

    inline const float getCloudPhaseIntensity() const {
        return cloud_phase_intensity_;
    }

    inline const float getCloudMoistToPressureRatio() const {
        return cloud_moist_to_pressure_ratio_;
    }

    inline float getAirFlowStrength() {
        return air_flow_strength_;
    }

    inline float getWaterFlowStrength() {
        return water_flow_strength_;
    }

    inline float getGloalFlowDir() {
        return glm::radians(global_flow_dir_);
    }

    inline float getGlobalFlowSpeed() {
        return global_flow_speed_;
    }

    inline glm::vec4 getCloudNoiseWeight(const int32_t& idx) {
        return glm::vec4(
            cloud_noise_weight_[idx][0],
            cloud_noise_weight_[idx][1],
            cloud_noise_weight_[idx][2],
            cloud_noise_weight_[idx][3]);
    }

    inline float getCloudNoiseThresold() {
        return cloud_noise_thresold_;
    }

    inline float getCloudNoiseScrollingSpeed() {
        return -pow(2.0f, cloud_noise_scrolling_speed_);
    }

    inline glm::vec2 getCloudNoiseScale() {
        return glm::vec2(pow(10.0f, -cloud_noise_scale_[0]), pow(10.0f, -cloud_noise_scale_[1]));
    }

    void init(
        GLFWwindow* window,
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::Instance>& instance,
        const renderer::QueueFamilyList& queue_family_list,
        const renderer::SwapChainInfo& swap_chain_info,
        const std::shared_ptr<renderer::Queue>& graphics_queue,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::RenderPass>& render_pass);

    bool draw(
        const std::shared_ptr<renderer::CommandBuffer>& command_buffer,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const std::shared_ptr<renderer::Framebuffer>& framebuffer,
        const glm::uvec2& screen_size,
        const std::shared_ptr<scene_rendering::Skydome>& skydome,
        bool& dump_volume_noise,
        const float& delta_t);

    void destroy();

    // Release GPU resources (background texture etc.) at final shutdown.
    // Not called during swap chain recreation.
    void destroyResources();
};

}// namespace ui
}// namespace engine
