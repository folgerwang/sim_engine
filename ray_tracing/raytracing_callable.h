#pragma once
#include "raytracing_base.h"

namespace engine {
namespace ray_tracing {

class RayTracingCallableTest : public RayTracingBase {
    struct BottomLevelDataInfo : public BottomLevelDataInfoBase {
        renderer::BufferInfo   vertex_buffer{};
        renderer::BufferInfo   index_buffer{};
        renderer::BufferInfo   transform_buffer{};
    };

    struct TopLevelDataInfo : public TopLevelDataInfoBase {
    };

    struct RayTracingRenderingInfo : public RayTracingRenderingInfoBase {
    };

    struct UniformData : public UniformDataBase {
    };

    virtual void initBottomLevelDataInfo(
        const std::shared_ptr<renderer::Device>& device) final;

    virtual void initTopLevelDataInfo(
        const std::shared_ptr<renderer::Device>& device) final;

    virtual void createRayTracingPipeline(
        const std::shared_ptr<renderer::Device>& device) final;

    virtual void createShaderBindingTables(
        const std::shared_ptr<renderer::Device>& device,
        const renderer::PhysicalDeviceRayTracingPipelineProperties& rt_pipeline_properties,
        const renderer::PhysicalDeviceAccelerationStructureFeatures& as_features) final;

    virtual void createRtResources(
        const std::shared_ptr<renderer::Device>& device) final;

    virtual void createDescriptorSets(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::BufferInfo>& camera_info) final;

public:
    RayTracingCallableTest() {}

    virtual void init(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
        const std::shared_ptr<renderer::BufferInfo>& camera_info,
        const renderer::PhysicalDeviceRayTracingPipelineProperties& rt_pipeline_properties,
        const renderer::PhysicalDeviceAccelerationStructureFeatures& as_features,
        glm::uvec2 size) final;

    virtual void draw(
        const std::shared_ptr<renderer::Device>& device,
        const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
        const glsl::ViewParams& view_params) final;

    virtual renderer::TextureInfo getFinalImage() final {
        return rt_render_info_->result_image;
    }

    virtual void destroy(
        const std::shared_ptr<renderer::Device>& device) final;
};

} //namespace ray_tracing
} //namespace engine
