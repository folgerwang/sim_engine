#pragma once
#include <memory>
#include <vector>
#include <optional>
#include <functional>
#include <string>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/hash.hpp>

#include "renderer_definition.h"

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace engine {
namespace renderer {
class Instance;
class Device;
class Sampler;
class ImageView;
class Buffer;
class DescriptorSet;
class CommandBuffer;
class Image;
class Pipeline;
class RenderPass;
class Framebuffer;
class PipelineLayout;
class DescriptorSetLayout;
class DescriptorPool;
class DeviceMemory;
class PhysicalDevice;
class ShaderModule;
class Swapchain;
class Surface;
class CommandPool;
class Queue;
class Semaphore;
class Fence;
struct ImageResourceInfo;

struct MemoryRequirements {
    uint64_t        size;
    uint64_t        alignment;
    uint32_t        memory_type_bits;
};

struct BufferCopyInfo {
    uint64_t        src_offset;
    uint64_t        dst_offset;
    uint64_t        size;
};

struct ImageSubresourceLayers {
    ImageAspectFlags      aspect_mask = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
    uint32_t              mip_level = 0;
    uint32_t              base_array_layer = 0;
    uint32_t              layer_count = 1;
};

struct BufferImageCopyInfo {
    uint64_t        buffer_offset;
    uint32_t        buffer_row_length;
    uint32_t        buffer_image_height;
    ImageSubresourceLayers    image_subresource;
    glm::ivec3      image_offset;
    glm::uvec3      image_extent;
};

struct ImageCopyInfo {
    ImageSubresourceLayers    src_subresource;
    glm::ivec3                src_offset;
    ImageSubresourceLayers    dst_subresource;
    glm::ivec3                dst_offset;
    glm::uvec3                extent;
};

struct ImageBlitInfo {
    ImageSubresourceLayers    src_subresource;
    glm::ivec3                src_offsets[2];
    ImageSubresourceLayers    dst_subresource;
    glm::ivec3                dst_offsets[2];
};

struct ImageResolveInfo {
    ImageSubresourceLayers    src_subresource;
    glm::ivec3                src_offset;
    ImageSubresourceLayers    dst_subresource;
    glm::ivec3                dst_offset;
    glm::uvec3                extent;
};

union ClearColorValue {
    float       float32[4];
    int32_t     int32[4];
    uint32_t    uint32[4];
};

struct ClearDepthStencilValue {
    float       depth;
    uint32_t    stencil;
};

union ClearValue {
    ClearColorValue           color;
    ClearDepthStencilValue    depth_stencil;
};

struct VertexInputBindingDescription {
    uint32_t            binding;
    uint32_t            stride;
    VertexInputRate     input_rate;
};

struct VertexInputAttributeDescription {
    uint32_t            buffer_view;
    uint32_t            binding;
    uint32_t            location;
    Format              format;
    uint64_t            offset;
    uint64_t            buffer_offset;
};

struct IndexInputBindingDescription {
    uint32_t            buffer_view;
    uint64_t            offset;
    uint64_t            index_count;
    IndexType           index_type;
};

struct ImageResourceInfo {
    ImageLayout             image_layout = ImageLayout::UNDEFINED;
    AccessFlags             access_flags = 0;
    PipelineStageFlags      stage_flags = 0;
};

struct BufferResourceInfo {
    AccessFlags             access_flags = 0;
    PipelineStageFlags      stage_flags = 0;
};

struct PipelineInputAssemblyStateCreateInfo {
    PrimitiveTopology                  topology;
    bool                               restart_enable;
};

struct PipelineColorBlendAttachmentState {
    bool                     blend_enable;
    BlendFactor              src_color_blend_factor;
    BlendFactor              dst_color_blend_factor;
    BlendOp                  color_blend_op;
    BlendFactor              src_alpha_blend_factor;
    BlendFactor              dst_alpha_blend_factor;
    BlendOp                  alpha_blend_op;
    ColorComponentFlags      color_write_mask;
};

struct PipelineColorBlendStateCreateInfo {
    bool                                          logic_op_enable;
    LogicOp                                       logic_op;
    uint32_t                                      attachment_count;
    const PipelineColorBlendAttachmentState* attachments;
    glm::vec4                                     blend_constants;
};

struct PipelineRasterizationStateCreateInfo {
    bool                                       depth_clamp_enable;
    bool                                       rasterizer_discard_enable;
    PolygonMode                                polygon_mode;
    CullModeFlags                              cull_mode;
    FrontFace                                  front_face;
    bool                                       depth_bias_enable;
    float                                      depth_bias_constant_factor;
    float                                      depth_bias_clamp;
    float                                      depth_bias_slope_factor;
    float                                      line_width;
};

struct PipelineMultisampleStateCreateInfo {
    SampleCountFlagBits         rasterization_samples;
    bool                        sample_shading_enable;
    float                       min_sample_shading;
    const SampleMask* sample_mask;
    bool                        alpha_to_coverage_enable;
    bool                        alpha_to_one_enable;
};

struct StencilOpState {
    StencilOp       fail_op;
    StencilOp       pass_op;
    StencilOp       depth_fail_op;
    CompareOp       compare_op;
    uint32_t        compare_mask;
    uint32_t        write_mask;
    uint32_t        reference;
};

struct PipelineDepthStencilStateCreateInfo {
    bool                        depth_test_enable;
    bool                        depth_write_enable;
    CompareOp                   depth_compare_op;
    bool                        depth_bounds_test_enable;
    bool                        stencil_test_enable;
    StencilOpState              front;
    StencilOpState              back;
    float                       min_depth_bounds;
    float                       max_depth_bounds;
};

struct AttachmentDescription {
    AttachmentDescriptionFlags    flags;
    Format                        format;
    SampleCountFlagBits           samples;
    AttachmentLoadOp              load_op;
    AttachmentStoreOp             store_op;
    AttachmentLoadOp              stencil_load_op;
    AttachmentStoreOp             stencil_store_op;
    ImageLayout                   initial_layout;
    ImageLayout                   final_layout;
};

struct AttachmentReference {
    AttachmentReference() :
        attachment_(0),
        layout_(ImageLayout::UNDEFINED) {}
    AttachmentReference(uint32_t attachment, ImageLayout layout) :
        attachment_(attachment),
        layout_(layout) {}
    uint32_t                      attachment_;
    ImageLayout                   layout_;
};

struct SubpassDescription {
    SubpassDescriptionFlags         flags;
    PipelineBindPoint               pipeline_bind_point;
    std::vector<AttachmentReference> input_attachments;
    std::vector<AttachmentReference> color_attachments;
    std::vector<AttachmentReference> resolve_attachments;
    std::vector<AttachmentReference> depth_stencil_attachment;
    uint32_t                        preserve_attachment_count;
    const uint32_t* preserve_attachments;
};

struct SubpassDependency {
    uint32_t                src_subpass;
    uint32_t                dst_subpass;
    PipelineStageFlags      src_stage_mask;
    PipelineStageFlags      dst_stage_mask;
    AccessFlags             src_access_mask;
    AccessFlags             dst_access_mask;
    DependencyFlags         dependency_flags;
};

struct DescriptorSetLayoutBinding {
    uint32_t              binding;
    DescriptorType        descriptor_type;
    uint32_t              descriptor_count;
    ShaderStageFlags      stage_flags;
    std::shared_ptr<Sampler> immutable_samplers;
};

struct PushConstantRange {
    ShaderStageFlags      stage_flags;
    uint32_t              offset;
    uint32_t              size;
};

struct WriteDescriptor {
    uint32_t binding = (uint32_t)-1;
    DescriptorType desc_type = DescriptorType::MAX_ENUM;
    std::shared_ptr<DescriptorSet> desc_set = nullptr;
};

struct TextureDescriptor : public WriteDescriptor{
    std::shared_ptr<Sampler> sampler = nullptr;
    std::shared_ptr<ImageView> texture = nullptr;
    ImageLayout image_layout = ImageLayout::SHADER_READ_ONLY_OPTIMAL;
};

struct BufferDescriptor : public WriteDescriptor {
    uint32_t offset = 0;
    uint32_t range = 0;
    std::shared_ptr<Buffer> buffer = nullptr;
};

struct AccelerationStructDescriptor : public WriteDescriptor {
    std::vector<AccelerationStructure> acc_structs;
};

struct QueueFamilyInfo {
    QueueFlags queue_flags_;
    uint32_t queue_count_;
    uint32_t index_;
    bool present_support_;
};

struct QueueFamilyList {
    std::vector<QueueFamilyInfo> queue_families_;

    uint32_t getQueueFamilyIndex(QueueFlagBits queue_flags) const {
        for (auto family : queue_families_) {
            auto query_flags = static_cast<uint32_t>(queue_flags);
            if ((static_cast<uint32_t>(family.queue_flags_) & query_flags) == query_flags) {
                return family.index_;
            }
        }
        return (uint32_t)-1;
    }

    const QueueFamilyInfo getQueueInfo(uint32_t family_index) const {
        if (family_index != (uint32_t)-1) {
            return queue_families_[family_index];
        }
        else {
            return QueueFamilyInfo{ 
                static_cast<uint32_t>(QueueFlagBits::FLAG_BITS_MAX_ENUM),
                0,
                (uint32_t)-1,
                false };
        }
    }

    std::vector<uint32_t> getGraphicAndPresentFamilyIndex() const {
        auto query_flags = static_cast<uint32_t>(QueueFlagBits::GRAPHICS_BIT);
        for (auto family : queue_families_) {
            if ((static_cast<uint32_t>(family.queue_flags_) & query_flags) && family.present_support_) {
                return { static_cast<uint32_t>(family.index_) };
            }
        }

        std::vector<uint32_t> index;
        index.reserve(2);
        for (auto family : queue_families_) {
            if ((static_cast<uint32_t>(family.queue_flags_) & query_flags)) {
                index.push_back(family.index_);
                break;
            }
        }

        for (auto family : queue_families_) {
            if (family.present_support_) {
                index.push_back(family.index_);
                break;
            }
        }

        if (index.size() == 2) {
            return index;
        }
        else {
            return {};
        }
    }

    bool isComplete() {
        return getGraphicAndPresentFamilyIndex().size() > 0;
    }
};

struct TextureInfo {
    bool                               linear = true;
    glm::uvec3                         size = glm::uvec3(0);
    std::shared_ptr<Image>             image;
    std::shared_ptr<DeviceMemory>      memory;
    std::shared_ptr<ImageView>         view;
    std::vector<std::vector<std::shared_ptr<ImageView>>> surface_views;
    std::vector<std::shared_ptr<Framebuffer>> framebuffers;

    void destroy(const std::shared_ptr<Device>& device);
};

struct BufferInfo {
    std::shared_ptr<Buffer>             buffer;
    std::shared_ptr<DeviceMemory>       memory;

    void destroy(const std::shared_ptr<Device>& device);
};

struct MemoryBarrier {
    AccessFlags             src_access_mask;
    AccessFlags             dst_access_mask;
};

struct BufferMemoryBarrier {
    AccessFlags                 src_access_mask;
    AccessFlags                 dst_access_mask;
    uint32_t                    src_queue_family_index = (~0U);
    uint32_t                    dst_queue_family_index = (~0U);
    std::shared_ptr<Buffer>     buffer;
    uint64_t                    offset;
    uint64_t                    size;
};

struct ImageSubresourceRange {
    ImageAspectFlags      aspect_mask = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
    uint32_t              base_mip_level = 0;
    uint32_t              level_count = 1;
    uint32_t              base_array_layer = 0;
    uint32_t              layer_count = 1;
};

struct ImageMemoryBarrier {
    AccessFlags                src_access_mask;
    AccessFlags                dst_access_mask;
    ImageLayout                old_layout;
    ImageLayout                new_layout;
    uint32_t                   src_queue_family_index = (~0U);
    uint32_t                   dst_queue_family_index = (~0U);
    std::shared_ptr<Image>     image;
    ImageSubresourceRange      subresource_range;
};

struct BarrierList {
    std::vector<MemoryBarrier>          memory_barriers;
    std::vector<BufferMemoryBarrier>    buffer_barriers;
    std::vector<ImageMemoryBarrier>     image_barriers;
};

//indexed
struct DrawIndexedIndirectCommand {
    uint32_t    index_count;
    uint32_t    instance_count;
    uint32_t    first_index;
    int32_t     vertex_offset;
    uint32_t    first_instance;
};

//non indexed
struct DrawIndirectCommand {
    uint32_t    vertex_count;
    uint32_t    instance_count;
    uint32_t    first_vertex;
    uint32_t    first_instance;
};

union DeviceOrHostAddressConst {
    DeviceAddress   device_address;
    const void*     host_address;
};

union DeviceOrHostAddress {
    DeviceAddress   device_address;
    void*           host_address;
};

struct AccelerationStructureGeometryTrianglesData {
    StructureType                    struct_type;
    const void*                      p_next;
    Format                           vertex_format;
    DeviceOrHostAddressConst         vertex_data;
    DeviceSize                       vertex_stride;
    uint32_t                         max_vertex;
    IndexType                        index_type;
    DeviceOrHostAddressConst         index_data;
    DeviceOrHostAddressConst         transform_data;
};

struct AccelerationStructureGeometryAabbsData {
    StructureType                    struct_type;
    const void*                      p_next;
    DeviceOrHostAddressConst         data;
    DeviceSize                       stride;
};

struct AccelerationStructureGeometryInstancesData {
    StructureType                    struct_type;
    const void*                      p_next;
    Bool32                           array_of_pointers;
    DeviceOrHostAddressConst         data;
};

union AccelerationStructureGeometryData {
    AccelerationStructureGeometryTrianglesData    triangles;
    AccelerationStructureGeometryAabbsData        aabbs;
    AccelerationStructureGeometryInstancesData    instances;
};

struct VertexInput {
    uint32_t            base = -1;
    uint32_t            stride = 0;
};

struct AccelerationStructureGeometry {
    GeometryType                         geometry_type;
    AccelerationStructureGeometryData    geometry;
    GeometryFlags                        flags;
    uint32_t                             max_primitive_count = 0;
    VertexInput                          position;
    VertexInput                          normal;
    VertexInput                          uv;
    VertexInput                          color;
    VertexInput                          tangent;
    uint32_t                             index_base = -1;
    uint32_t                             index_by_bytes = 16;
};

struct AccelerationStructureBuildGeometryInfo {
    AccelerationStructureType                 type;
    BuildAccelerationStructureFlags           flags;
    BuildAccelerationStructureMode            mode;
    AccelerationStructure                     src_as;
    AccelerationStructure                     dst_as;
    std::vector<std::shared_ptr<AccelerationStructureGeometry>> geometries;
    DeviceOrHostAddress                       scratch_data;
};

struct AccelerationStructureBuildSizesInfo {
    StructureType                   struct_type;
    const void*                     p_next;
    DeviceSize                      as_size;
    DeviceSize                      update_scratch_size;
    DeviceSize                      build_scratch_size;
};

struct AccelerationStructureBuildRangeInfo {
    uint32_t    primitive_count;
    uint32_t    primitive_offset;
    uint32_t    first_vertex;
    uint32_t    transform_offset;
};

struct TransformMatrix {
    float    matrix[3][4];
};

struct AccelerationStructureInstance {
    TransformMatrix               transform;
    uint32_t                      instance_custom_index : 24;
    uint32_t                      mask : 8;
    uint32_t                      instance_shader_binding_table_record_offset : 24;
    GeometryInstanceFlags         flags : 8;
    uint64_t                      acceleration_structure_reference;
};

struct PhysicalDeviceRayTracingPipelineProperties {
    uint32_t           shader_group_handle_size;
    uint32_t           max_ray_recursion_depth;
    uint32_t           max_shader_group_stride;
    uint32_t           shader_group_base_alignment;
    uint32_t           shader_group_handle_capture_replay_size;
    uint32_t           max_ray_dispatch_invocation_count;
    uint32_t           shader_group_handle_alignment;
    uint32_t           max_ray_hit_attribute_size;
};

struct PhysicalDeviceAccelerationStructureFeatures {
    Bool32             acceleration_structure;
    Bool32             acceleration_structure_capture_replay;
    Bool32             acceleration_structure_indirect_build;
    Bool32             acceleration_structure_host_commands;
    Bool32             descriptor_binding_acceleration_structure_update_after_bind;
};

struct RayTracingShaderGroupCreateInfo {
    RayTracingShaderGroupType         type;
    uint32_t                          general_shader = (uint32_t)-1;
    uint32_t                          closest_hit_shader = (uint32_t)-1;
    uint32_t                          any_hit_shader = (uint32_t)-1;
    uint32_t                          intersection_shader = (uint32_t)-1;
    const void*                       p_shader_group_capture_replay_handle = nullptr;
};

struct StridedDeviceAddressRegion {
    DeviceAddress       device_address;
    DeviceSize          stride;
    DeviceSize          size;
};

struct GraphicPipelineInfo {
    std::shared_ptr<PipelineColorBlendStateCreateInfo> blend_state_info;
    std::shared_ptr<PipelineRasterizationStateCreateInfo> rasterization_info;
    std::shared_ptr<PipelineMultisampleStateCreateInfo> ms_info;
    std::shared_ptr<PipelineDepthStencilStateCreateInfo> depth_stencil_info;
};

typedef std::vector<std::shared_ptr<PhysicalDevice>> PhysicalDeviceList;
typedef std::vector<std::shared_ptr<DescriptorSetLayout>> DescriptorSetLayoutList;
typedef std::vector<std::shared_ptr<DescriptorSet>> DescriptorSetList;
typedef std::vector<std::shared_ptr<ShaderModule>> ShaderModuleList;
typedef std::vector<renderer::RayTracingShaderGroupCreateInfo> RtShaderGroupCreateInfoList;
typedef std::vector<std::shared_ptr<WriteDescriptor>> WriteDescriptorList;
typedef std::vector<std::shared_ptr<AccelerationStructureGeometry>> AccelerationStructureGeometryList;
} // namespace renderer
} // namespace engine