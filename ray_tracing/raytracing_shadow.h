#pragma once
#include "raytracing_base.h"
#include "game_object/gltf.h"

namespace engine {
namespace ray_tracing {

class RayTracingShadowTest : public RayTracingBase {
    struct BottomLevelDataInfo : public BottomLevelDataInfoBase {
        renderer::BufferInfo   rt_geometry_matrix_buffer{};
        renderer::BufferInfo   rt_geometry_info_buffer{};
    };

    struct TopLevelDataInfo : public TopLevelDataInfoBase {
    };

    struct RayTracingRenderingInfo : public RayTracingRenderingInfoBase {
    };

    struct UniformData : public UniformDataBase {
        vec4      light_pos;
    };

    std::shared_ptr<game_object::GltfData> game_object_{};

    virtual void initBottomLevelDataInfo(
        const renderer::DeviceInfo& device_info) final;

    virtual void initTopLevelDataInfo(
        const renderer::DeviceInfo& device_info) final;

    virtual void createRayTracingPipeline(
        const renderer::DeviceInfo& device_info) final;

    virtual void createShaderBindingTables(
        const renderer::DeviceInfo& device_info,
        const renderer::PhysicalDeviceRayTracingPipelineProperties& rt_pipeline_properties,
        const renderer::PhysicalDeviceAccelerationStructureFeatures& as_features) final;

    virtual void createRtResources(
        const renderer::DeviceInfo& device_info) final;

    virtual void createDescriptorSets(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool) final;

public:
    RayTracingShadowTest() {}

    virtual void init(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const renderer::PhysicalDeviceRayTracingPipelineProperties& rt_pipeline_properties,
        const renderer::PhysicalDeviceAccelerationStructureFeatures& as_features,
        glm::uvec2 size) final;

    virtual renderer::TextureInfo draw(
        const renderer::DeviceInfo& device_info,
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glsl::ViewParams& view_params) final;

    virtual void destroy(
        const std::shared_ptr<renderer::Device>& device) final;
};

} //namespace ray_tracing
} //namespace engine
