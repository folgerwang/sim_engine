#pragma once
#include <array>
#include <cmath>
#include "renderer/renderer.h"
#include "scene_rendering/skydome.h"
#include "shaders/global_definition.glsl.h"
#include "helper/gpu_profiler.h"
#include "chat_box.h"
#include "title_screen_config.h"
#include "imgui.h"

namespace plugins { class PluginManager; }

namespace engine {
namespace game_object { class MeshLoadTaskManager; }
namespace scene_rendering { class SSAO; class ClusterRenderer; }
namespace ui {

// Game-level state machine driven by the title-screen menu.
enum class GameState {
    TitleScreen,   // showing background + menu items
    Loading,       // "New Game" clicked — meshes streaming
    InGame         // meshes loaded, gameplay active
};

class Menu {
    std::vector<std::string> gltf_file_names_;
    std::vector<std::string> to_load_gltf_names_;
    std::string spawn_gltf_name_;
    bool turn_off_water_pass_ = false;
    bool turn_off_grass_pass_ = false;
    bool turn_off_ray_tracing_ = false;
    bool turn_off_volume_moist_ = false;
    bool turn_off_shadow_pass_ = false;
    bool turn_on_airflow_ = false;
    uint32_t debug_draw_type_ = 0;
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

    // Collision-mesh debug visualisation. When ON the application
    // skips the regular object forward pass and draws the static
    // CollisionWorld with hashed-per-triangle colours instead. Also
    // toggleable via F1 in the application's key callback (which
    // forwards through toggleCollisionDebug() below).
    bool show_collision_debug_ = false;

    // Which CollisionShape mode application.cpp should pass to
    // CollisionMesh::buildFromDrawablePrimitive when (re)building
    // the collision world for visualisation. Three user-facing
    // options live in the Physics submenu; the application maps
    // them onto the engine's CollisionShape enum at build time.
    // Default is Volume (5cm voxel cubes) -- the cleanest read
    // for inspecting what gameplay actually collides against.
public:
    enum class CollisionDebugShape : int {
        Original   = 0,   // CollisionShape::None
        Simplified = 1,   // CollisionShape::Decimate (post-weld)
        Volume     = 2,   // CollisionShape::VoxelCube
    };
private:
    CollisionDebugShape collision_debug_shape_ =
        CollisionDebugShape::Volume;
    // Set by setCollisionDebugShape() whenever the menu choice
    // changes; the application clears it after rebuilding the
    // collision world to match.
    bool collision_world_dirty_ = false;

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
    // Speed factor: game time / real time.  100.0 means 1 real-second
    // advances the in-game clock by 100 game-seconds, so a full 24-hour
    // cycle takes ~14.4 real minutes.  Slider in the Skydome window
    // exposes this directly; advanceTimeOfDay() converts to game-hours
    // internally (game-seconds-added / 3600).
    float tod_advance_speed_ = 100.0f;
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

    std::string getToLoadPlayerNameAndClear() {
        auto result = spawn_gltf_name_;
        spawn_gltf_name_ = "";
        return result;
    }

    // CSM debug visualisation
    inline bool showCsmDebug() const { return show_csm_debug_; }
    void setCsmDebugTextureIds(const std::array<ImTextureID, CSM_CASCADE_COUNT>& ids) {
        csm_debug_tex_ids_ = ids;
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

    inline bool isShadowPassTurnOff() const {
        return turn_off_shadow_pass_;
    }

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
