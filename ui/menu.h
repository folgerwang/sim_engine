#pragma once
#include <array>
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

    std::shared_ptr<ChatBox> chat_box_;

    // Optional GPU profiler — set from application after init.
    engine::helper::GpuProfiler* gpu_profiler_ = nullptr;

    // Plugin system — set from application after init.
    plugins::PluginManager* plugin_manager_ = nullptr;

    // Optional async mesh loader — set from application after init.
    // Used by the HUD spinner overlay (see draw()) to show which
    // meshes are still loading. Non-owning.
    engine::game_object::MeshLoadTaskManager* mesh_load_task_manager_ = nullptr;

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
