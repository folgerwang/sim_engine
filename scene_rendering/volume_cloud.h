#pragma once
#include "renderer/renderer.h"
#include "scene_rendering/skydome.h"

namespace engine {
namespace scene_rendering {

// Ground-fog knobs, filled from the weather menu each frame and pushed
// into the volume-cloud march (VolumeMoistrueParams.fog_*).
struct GroundFogParams {
    float density = 0.0f;          // 0 = fog off
    float height_falloff = 0.06f;
    float base_y = 0.0f;
    float g = 0.55f;
    float sun_intensity = 1.2f;
    float ambient_intensity = 0.35f;
};

class VolumeCloud {
    renderer::TextureInfo fog_cloud_tex_;
    renderer::TextureInfo blurred_fog_cloud_tex_;

    std::shared_ptr<renderer::DescriptorSetLayout> render_cloud_fog_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> render_cloud_fog_desc_set_[2];
    std::shared_ptr<renderer::PipelineLayout> render_cloud_fog_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> render_cloud_fog_pipeline_;
    std::shared_ptr<renderer::DescriptorSetLayout> blur_image_desc_set_layout_;
    std::shared_ptr<renderer::DescriptorSet> blur_image_x_tex_desc_set_;
    std::shared_ptr<renderer::PipelineLayout> blur_image_pipeline_layout_;
    std::shared_ptr<renderer::Pipeline> blur_image_x_pipeline_;
    std::shared_ptr<renderer::DescriptorSet> blur_image_y_merge_tex_desc_set_;
    std::shared_ptr<renderer::Pipeline> blur_image_y_merge_pipeline_;

public:
    VolumeCloud(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::Sampler>& point_clamp_texture_sampler,
        const std::shared_ptr<renderer::ImageView>& src_depth,
        const std::shared_ptr<renderer::ImageView>& hdr_color,
        const std::vector<std::shared_ptr<renderer::ImageView>>& moisture_texes,
        const std::vector<std::shared_ptr<renderer::ImageView>>& temp_texes,
        const std::shared_ptr<renderer::ImageView>& cloud_lighting_tex,
        const std::shared_ptr<renderer::ImageView>& scattering_lut_tex,
        const std::shared_ptr<renderer::ImageView>& detail_noise_tex,
        const std::shared_ptr<renderer::ImageView>& rough_noise_tex,
        const std::shared_ptr<renderer::ImageView>& csm_shadow_tex,
        const std::shared_ptr<renderer::BufferInfo>& runtime_lights_buffer,
        const glm::uvec2& display_size);

    void recreate(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::Sampler>& point_clamp_texture_sampler,
        const std::shared_ptr<renderer::ImageView>& src_depth,
        const std::shared_ptr<renderer::ImageView>& hdr_color,
        const std::vector<std::shared_ptr<renderer::ImageView>>& moisture_texes,
        const std::vector<std::shared_ptr<renderer::ImageView>>& temp_texes,
        const std::shared_ptr<renderer::ImageView>& cloud_lighting_tex,
        const std::shared_ptr<renderer::ImageView>& scattering_lut_tex,
        const std::shared_ptr<renderer::ImageView>& detail_noise_tex,
        const std::shared_ptr<renderer::ImageView>& rough_noise_tex,
        const std::shared_ptr<renderer::ImageView>& csm_shadow_tex,
        const std::shared_ptr<renderer::BufferInfo>& runtime_lights_buffer,
        const glm::uvec2& display_size);

    void renderVolumeCloud(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set,
        const std::shared_ptr<renderer::Image>& hdr_color,
        const std::shared_ptr<scene_rendering::Skydome>& skydome,
        const float& view_ext_factor,
        const float& view_ext_exponent,
        const float& ambient_intensity,
        const float& phase_intensity,
        const float& moist_to_pressure_ratio,
        const glm::vec4& noise_weights_0,
        const glm::vec4& noise_weights_1,
        const float& noise_thresold,
        const float& noise_scrolling_speed,
        const glm::vec2& noise_scale,
        const GroundFogParams& ground_fog,
        const glm::uvec2& display_size,
        int dbuf_idx,
        float current_time);

    void destroy(const std::shared_ptr<renderer::Device>& device);
};

}// namespace scene_rendering
}// namespace engine
