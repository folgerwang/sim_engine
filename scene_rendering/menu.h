#pragma once
#include "renderer/renderer.h"
#include "scene_rendering/skydome.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace scene_rendering {

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

    ImTextureID texture_id_;

public:
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
        const std::shared_ptr<renderer::ImageView>& image_view);

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
};

}// namespace scene_rendering
}// namespace engine
