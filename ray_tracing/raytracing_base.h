#pragma once
#include "renderer/renderer.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace ray_tracing {

inline uint32_t alignedSize(uint32_t size, uint32_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

struct BottomLevelDataInfoBase {
    renderer::BufferInfo   as_buffer{};
    renderer::BufferInfo   scratch_buffer{};

    renderer::AccelerationStructure   as_handle{};
    renderer::DeviceAddress   as_device_address = 0;
};

struct TopLevelDataInfoBase {
    renderer::BufferInfo   instance_buffer{};
    renderer::BufferInfo   as_buffer{};
    renderer::BufferInfo   scratch_buffer{};

    renderer::AccelerationStructure   as_handle{};
    renderer::DeviceAddress   as_device_address = 0;
};

struct RayTracingRenderingInfoBase {
    renderer::ShaderModuleList shader_modules;
    renderer::RtShaderGroupCreateInfoList shader_groups;
    std::shared_ptr<renderer::DescriptorSetLayout> rt_desc_set_layout;
    std::shared_ptr<renderer::PipelineLayout> rt_pipeline_layout;
    std::shared_ptr<renderer::Pipeline> rt_pipeline;
    std::shared_ptr<renderer::DescriptorSet> rt_desc_set;
    renderer::StridedDeviceAddressRegion raygen_shader_sbt_entry;
    renderer::StridedDeviceAddressRegion miss_shader_sbt_entry;
    renderer::StridedDeviceAddressRegion hit_shader_sbt_entry;
    renderer::StridedDeviceAddressRegion callable_shader_sbt_entry;
    renderer::BufferInfo raygen_shader_binding_table;
    renderer::BufferInfo miss_shader_binding_table;
    renderer::BufferInfo hit_shader_binding_table;
    renderer::BufferInfo callable_shader_binding_table;
    renderer::BufferInfo ubo;
    renderer::TextureInfo result_image;
};

struct UniformDataBase {
};

class RayTracingBase {
protected:
    std::shared_ptr<BottomLevelDataInfoBase> bl_data_info_;
    std::shared_ptr<TopLevelDataInfoBase> tl_data_info_;
    std::shared_ptr<RayTracingRenderingInfoBase> rt_render_info_;
    glm::uvec2 rt_size_ = glm::vec2(0.0f);

    virtual void initBottomLevelDataInfo(
        const std::shared_ptr<renderer::Device>& device) = 0;

    virtual void initTopLevelDataInfo(
        const std::shared_ptr<renderer::Device>& device) = 0;

    virtual void createRayTracingPipeline(
        const std::shared_ptr<renderer::Device>& device) = 0;

    virtual void createShaderBindingTables(
        const std::shared_ptr<renderer::Device>& device,
        const renderer::PhysicalDeviceRayTracingPipelineProperties& rt_pipeline_properties,
        const renderer::PhysicalDeviceAccelerationStructureFeatures& as_features) = 0;

    virtual void createRtResources(
        const std::shared_ptr<renderer::Device>& device) = 0;

    virtual void createDescriptorSets(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::BufferInfo>& camera_info) = 0;

public:
    virtual void init(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::BufferInfo>& camera_info,
        const renderer::PhysicalDeviceRayTracingPipelineProperties& rt_pipeline_properties,
        const renderer::PhysicalDeviceAccelerationStructureFeatures& as_features,
        glm::uvec2 size) = 0;

    virtual void draw(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glsl::ViewParams& view_params) = 0;

    virtual renderer::TextureInfo getFinalImage() = 0;

    virtual void destroy(
        const std::shared_ptr<renderer::Device>& device) = 0;
};

} //namespace ray_tracing
} //namespace engine
