#pragma once
#include <unordered_map>
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace game_object {

class ViewCamera {
    static std::shared_ptr<renderer::DescriptorSetLayout> update_view_camera_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> update_view_camera_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> update_view_camera_pipeline_;

    std::shared_ptr<renderer::DescriptorSet> update_view_camera_desc_set_[2];
    std::shared_ptr<renderer::BufferInfo> view_camera_buffer_;

public:
    ViewCamera() = delete;
    ViewCamera(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool);

    void update(const std::shared_ptr<renderer::Device>& device, const float& time);

    void createViewCameraUpdateDescSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& rock_layer,
        const renderer::TextureInfo& soil_water_layer_0,
        const renderer::TextureInfo& soil_water_layer_1,
        const renderer::BufferInfo& game_objects_buffer);

    void initViewCameraBuffer(
        const std::shared_ptr<renderer::Device>& device);

    static void initStaticMembers(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::DescriptorSetLayoutList& global_desc_set_layouts);

    static void createStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    static void recreateStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    void generateDescriptorSet(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::Sampler>& texture_sampler,
        const renderer::TextureInfo& rock_layer,
        const renderer::TextureInfo& soil_water_layer_0,
        const renderer::TextureInfo& soil_water_layer_1,
        const renderer::BufferInfo& game_objects_buffer);

    static void destroyStaticMembers(
        const std::shared_ptr<renderer::Device>& device);

    void destroy(
        const std::shared_ptr<renderer::Device>& device);

    void updateViewCameraBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glsl::ViewCameraParams& game_camera_params,
        int soil_water);

    glsl::ViewCameraInfo readCameraInfo(
        const std::shared_ptr<renderer::Device>& device);

    std::shared_ptr<renderer::BufferInfo> getViewCameraBuffer();
};

} // namespace game_object
} // namespace engine