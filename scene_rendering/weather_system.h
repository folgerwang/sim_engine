#pragma once
#include "renderer/renderer.h"
#include "scene_rendering/skydome.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace scene_rendering {

class WeatherSystem {
    renderer::TextureInfo temp_volume_[2];
    renderer::TextureInfo moisture_volume_[2];
    renderer::TextureInfo pressure_volume_[2];
    renderer::TextureInfo airflow_volume_;
    renderer::TextureInfo cloud_shadow_volume_;
    renderer::TextureInfo ground_airflow_info_;
    renderer::TextureInfo temp_ground_airflow_info_;

    std::shared_ptr<renderer::DescriptorSet> temperature_init_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSetLayout> temperature_init_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> temperature_init_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> temperature_init_pipeline_;
    std::shared_ptr<renderer::DescriptorSet> airflow_tex_desc_set_[2];
    std::shared_ptr<renderer::DescriptorSetLayout> airflow_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> airflow_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> airflow_pipeline_;
    std::shared_ptr<renderer::DescriptorSet> cloud_shadow_tex_desc_set_[2];
    std::shared_ptr<renderer::DescriptorSetLayout> cloud_shadow_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> cloud_shadow_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> cloud_shadow_init_pipeline_;
    std::shared_ptr<renderer::DescriptorSet> cloud_shadow_merge_tex_desc_set_;
    std::shared_ptr<renderer::DescriptorSetLayout> cloud_shadow_merge_desc_set_layout_;
    std::shared_ptr<renderer::PipelineLayout> cloud_shadow_merge_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> cloud_shadow_merge_pipeline_;

public:
    WeatherSystem(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& rock_layer_tex,
        const std::vector<std::shared_ptr<renderer::ImageView>>& soil_water_layer_tex);

    inline std::shared_ptr<renderer::ImageView> getTempTex(int idx) {
        return temp_volume_[idx].view;
    }

    inline std::shared_ptr<renderer::ImageView> getMoistureTex(int idx) {
        return moisture_volume_[idx].view;
    }

    inline std::shared_ptr<renderer::ImageView> getPressureTex(int idx) {
        return pressure_volume_[idx].view;
    }

    inline std::vector<std::shared_ptr<renderer::ImageView>> getTempTexes() {
        return { temp_volume_[0].view, temp_volume_[1].view };
    }

    inline std::vector<std::shared_ptr<renderer::ImageView>> getMoistureTexes() {
        return { moisture_volume_[0].view, moisture_volume_[1].view };
    }

    inline std::shared_ptr<renderer::ImageView> getAirflowTex() {
        return airflow_volume_.view;
    }

    inline std::shared_ptr<renderer::ImageView> getCloudLightingTex() {
        return cloud_shadow_volume_.view;
    }

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& rock_layer_tex,
        const std::vector<std::shared_ptr<renderer::ImageView>>& soil_water_layer_tex);

    void initTemperatureBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf);

    void updateAirflowBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glsl::WeatherControl& weather_controls,
        const float& global_flow_angle,
        const float& global_flow_speed,
        const float& moist_to_pressure_ratio,
        int dbuf_idx,
        float current_time);

    void updateCloudShadow(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glm::vec3& sun_dir,
        const float& light_ext_factor,
        const int& dbuf_idx,
        const float& current_time);

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine
