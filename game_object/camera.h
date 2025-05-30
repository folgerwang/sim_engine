#pragma once
#include <unordered_map>
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace game_object {

extern glm::vec3 getDirectionByYawAndPitch(float yaw, float pitch);

class ViewCamera {
    const std::shared_ptr<renderer::Device>& m_device_;
    const std::shared_ptr<renderer::DescriptorPool>& m_descriptor_pool_;

    static std::shared_ptr<renderer::DescriptorSetLayout> s_update_view_camera_desc_set_layout_;
    static std::shared_ptr<renderer::PipelineLayout> s_update_view_camera_pipeline_layout_;
    static std::shared_ptr<renderer::Pipeline> s_update_view_camera_pipeline_;

    std::shared_ptr<renderer::DescriptorSet> m_update_view_camera_desc_set_[2];
    std::shared_ptr<renderer::BufferInfo> m_view_camera_buffer_;
    glsl::ViewCameraInfo m_camera_info_;
    bool m_is_ortho_ = false;

public:
    ViewCamera() = delete;
    ViewCamera(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const glsl::ViewCameraParams& view_camera_params,
        bool is_ortho);

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
        const std::shared_ptr<renderer::Device>& device,
        const glsl::ViewCameraParams& view_camera_params);

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

    void updateViewCameraInfo(
        const glsl::ViewCameraParams& view_camera_params,
        std::shared_ptr<glm::vec3> input_camera_pos);

    void updateViewCameraBuffer(
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glsl::ViewCameraParams& game_camera_params,
        int soil_water);

    void readGpuCameraInfo(
        const std::shared_ptr<renderer::Device>& device);

    const glsl::ViewCameraInfo& getCameraInfo() const {
        return m_camera_info_;
    }

    std::shared_ptr<renderer::BufferInfo> getViewCameraBuffer() {
        return m_view_camera_buffer_;
    }

    void destroy(
        const std::shared_ptr<renderer::Device>& device);
};

} // namespace game_object
} // namespace engine