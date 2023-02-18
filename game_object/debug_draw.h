#pragma once
#include "renderer/renderer.h"

namespace engine {
namespace game_object {

class DebugDrawObject {
    const renderer::DeviceInfo& device_info_;

    glm::vec2 min_;
    glm::vec2 max_;

    static std::shared_ptr<renderer::DescriptorSet> debug_draw_desc_set_;
    static std::shared_ptr<renderer::DescriptorSetLayout> debug_draw_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> debug_draw_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> debug_draw_pipeline_;

public:
    DebugDrawObject() = delete;
    DebugDrawObject(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
        const glm::vec2& min,
        const glm::vec2& max);

    ~DebugDrawObject() {
        destroy();
    }

    void destroy();

    static void initStaticMembers(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size);

    static void createStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size);

    static void recreateStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::RenderPass>& render_pass,
        const renderer::GraphicPipelineInfo& graphic_pipeline_info,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
        const glm::uvec2& display_size);

    static void destroyStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    static void updateStaticDescriptorSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& temp_volume_tex,
        const std::shared_ptr<renderer::ImageView>& moisture_volume_tex,
        const std::shared_ptr<renderer::ImageView>& airflow_tex);

    static void generateStaticDescriptorSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& temp_volume_tex,
        const std::shared_ptr<renderer::ImageView>& moisture_volume_tex,
        const std::shared_ptr<renderer::ImageView>& airflow_tex);

    static void generateAllDescriptorSets(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const std::shared_ptr<renderer::ImageView>& temp_volume_tex,
        const std::shared_ptr<renderer::ImageView>& moisture_volume_tex,
        const std::shared_ptr<renderer::ImageView>& airflow_tex);

    static void draw(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const renderer::DescriptorSetList& desc_set_list,
        const glm::vec3& camera_pos,
        uint32_t debug_type);
};

} // namespace game_object
} // namespace engine