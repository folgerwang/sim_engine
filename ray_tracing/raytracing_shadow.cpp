#include <string>
#include "raytracing_shadow.h"
#include "renderer/renderer_helper.h"

namespace engine {
namespace {

enum {
    kRayGenIndex,
    kRayMissIndex,
    kShadowMissIndex,
    kClosestHitIndex,
    kNumRtShaders
};

std::shared_ptr<renderer::DescriptorSetLayout> createRtDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(7);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        0,
        SET_FLAG_BIT(ShaderStage, RAYGEN_BIT_KHR) |
        SET_FLAG_BIT(ShaderStage, CLOSEST_HIT_BIT_KHR),
        renderer::DescriptorType::ACCELERATION_STRUCTURE_KHR);
    bindings[1] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        1,
        SET_FLAG_BIT(ShaderStage, RAYGEN_BIT_KHR),
        renderer::DescriptorType::STORAGE_IMAGE);
    bindings[2] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        2,
        SET_FLAG_BIT(ShaderStage, RAYGEN_BIT_KHR) |
        SET_FLAG_BIT(ShaderStage, CLOSEST_HIT_BIT_KHR) |
        SET_FLAG_BIT(ShaderStage, MISS_BIT_KHR),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[3] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        3,
        SET_FLAG_BIT(ShaderStage, CLOSEST_HIT_BIT_KHR),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[4] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        4,
        SET_FLAG_BIT(ShaderStage, CLOSEST_HIT_BIT_KHR),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[5] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        5,
        SET_FLAG_BIT(ShaderStage, CLOSEST_HIT_BIT_KHR),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[6] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        6,
        SET_FLAG_BIT(ShaderStage, RAYGEN_BIT_KHR) |
        SET_FLAG_BIT(ShaderStage, CLOSEST_HIT_BIT_KHR) |
        SET_FLAG_BIT(ShaderStage, MISS_BIT_KHR),
        renderer::DescriptorType::UNIFORM_BUFFER);
    return device->createDescriptorSetLayout(bindings);
}

}

namespace ray_tracing {

void RayTracingShadowTest::init(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::BufferInfo>& camera_info,
    const renderer::PhysicalDeviceRayTracingPipelineProperties& rt_pipeline_properties,
    const renderer::PhysicalDeviceAccelerationStructureFeatures& as_features,
    glm::uvec2 size) {

    bl_data_info_ = std::make_shared<BottomLevelDataInfo>();
    tl_data_info_ = std::make_shared<TopLevelDataInfo>();
    rt_render_info_ = std::make_shared<RayTracingRenderingInfo>();
    rt_size_ = size;

    auto bl_data_info =
        std::reinterpret_pointer_cast<BottomLevelDataInfo>(bl_data_info_);

    game_object_ =
        game_object::DrawableObject::loadGltfModel(
            device,
            "assets/vulkanscene_shadow.gltf");

    std::vector<glsl::VertexBufferInfo> geom_info_list;
    for (auto i_node = 0; i_node < game_object_->nodes_.size(); i_node++) {
        auto& node = game_object_->nodes_[i_node];
        if (node.mesh_idx_ != -1) {
            auto& mesh = game_object_->meshes_[node.mesh_idx_];
            for (auto i = 0; i < mesh.primitives_.size(); i++) {
                auto& src_geom = mesh.primitives_[i].as_geometry;
                glsl::VertexBufferInfo dst_geom;
                dst_geom.matrix = glm::mat3x4(glm::transpose(node.cached_matrix_));
                dst_geom.position_base = src_geom->position.base;
                dst_geom.position_stride = src_geom->position.stride;
                dst_geom.normal_base = src_geom->normal.base;
                dst_geom.normal_stride = src_geom->normal.stride;
                dst_geom.uv_base = src_geom->uv.base;
                dst_geom.uv_stride = src_geom->uv.stride;
                dst_geom.color_base = src_geom->color.base;
                dst_geom.color_stride = src_geom->color.stride;
                dst_geom.tangent_base = src_geom->tangent.base;
                dst_geom.tangent_stride = src_geom->tangent.stride;
                dst_geom.index_base = src_geom->index_base;
                dst_geom.index_bytes = src_geom->index_by_bytes;
                geom_info_list.push_back(dst_geom);
            }
        }
    }

    // Transform buffer
    renderer::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT) |
        SET_FLAG_BIT(BufferUsage, ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        bl_data_info->rt_geometry_info_buffer.buffer,
        bl_data_info->rt_geometry_info_buffer.memory,
        std::source_location::current(),
        sizeof(glsl::VertexBufferInfo) * geom_info_list.size(),
        geom_info_list.data());

    initBottomLevelDataInfo(device);
    initTopLevelDataInfo(device);
    createRayTracingPipeline(device);
    createShaderBindingTables(
        device,
        rt_pipeline_properties,
        as_features);
    createRtResources(device);
    createDescriptorSets(
        device,
        descriptor_pool,
        camera_info);
}

void RayTracingShadowTest::initBottomLevelDataInfo(
    const std::shared_ptr<renderer::Device>& device) {

    auto bl_data_info =
        std::reinterpret_pointer_cast<BottomLevelDataInfo>(bl_data_info_);

    renderer::AccelerationStructureGeometryList rt_geometries;
    std::vector<glm::mat3x4> transform_matrices;
    for (auto i_node = 0; i_node < game_object_->nodes_.size(); i_node++) {
        auto& node = game_object_->nodes_[i_node];
        if (node.mesh_idx_ != -1) {
            auto& mesh = game_object_->meshes_[node.mesh_idx_];
            for (auto i = 0; i < mesh.primitives_.size(); i++) {
                auto matrix = glm::mat3x4(glm::transpose(node.cached_matrix_));
                transform_matrices.push_back(matrix);
                rt_geometries.push_back(mesh.primitives_[i].as_geometry);
            }
        }
    }

    // Transform buffer
    renderer::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT) |
        SET_FLAG_BIT(BufferUsage, ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        bl_data_info->rt_geometry_matrix_buffer.buffer,
        bl_data_info->rt_geometry_matrix_buffer.memory,
        std::source_location::current(),
        transform_matrices.size() * sizeof(glm::mat3x4),
        transform_matrices.data());

    auto matrix_device_address =
        bl_data_info->rt_geometry_matrix_buffer.buffer->getDeviceAddress();

    auto num_geoms = 0;
    for (auto i_node = 0; i_node < game_object_->nodes_.size(); i_node++) {
        auto& node = game_object_->nodes_[i_node];
        if (node.mesh_idx_ != -1) {
            auto& mesh = game_object_->meshes_[node.mesh_idx_];
            for (auto i = 0; i < mesh.primitives_.size(); i++) {
                rt_geometries[num_geoms]->geometry.triangles.transform_data.device_address =
                    matrix_device_address;
                num_geoms++;
            }
        }
    }

    // Get size info
    renderer::AccelerationStructureBuildGeometryInfo as_build_geometry_info{};
    as_build_geometry_info.type = renderer::AccelerationStructureType::BOTTOM_LEVEL_KHR;
    as_build_geometry_info.flags = SET_FLAG_BIT(BuildAccelerationStructure, PREFER_FAST_TRACE_BIT_KHR);

    as_build_geometry_info.geometries = rt_geometries;

    renderer::AccelerationStructureBuildSizesInfo as_build_size_info{};
    as_build_size_info.struct_type = renderer::StructureType::ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    device->getAccelerationStructureBuildSizes(
        renderer::AccelerationStructureBuildType::DEVICE_KHR,
        as_build_geometry_info,
        as_build_size_info);

    device->createBuffer(
        as_build_size_info.as_size,
        SET_FLAG_BIT(BufferUsage, ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        bl_data_info->as_buffer.buffer,
        bl_data_info->as_buffer.memory,
        std::source_location::current());

    bl_data_info->as_handle =
        device->createAccelerationStructure(
            bl_data_info->as_buffer.buffer,
            renderer::AccelerationStructureType::BOTTOM_LEVEL_KHR);

    device->createBuffer(
        as_build_size_info.build_scratch_size,
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        bl_data_info->scratch_buffer.buffer,
        bl_data_info->scratch_buffer.memory,
        std::source_location::current());

    as_build_geometry_info.mode = renderer::BuildAccelerationStructureMode::BUILD_KHR;
    as_build_geometry_info.dst_as = bl_data_info->as_handle;
    as_build_geometry_info.scratch_data.device_address =
        bl_data_info->scratch_buffer.buffer->getDeviceAddress();

    std::vector<renderer::AccelerationStructureBuildRangeInfo> as_build_range_infos(num_geoms);
    for (auto i = 0; i < num_geoms; i++) {
        auto& geometry = rt_geometries[i];
        as_build_range_infos[i].primitive_count = geometry->max_primitive_count;
        as_build_range_infos[i].primitive_offset = 0;
        as_build_range_infos[i].first_vertex = 0;
        as_build_range_infos[i].transform_offset = i * sizeof(glm::mat3x4);
    }

    // Build the acceleration structure on the device via a one-time command buffer submission
    // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
    auto cmd_buf = device->setupTransientCommandBuffer();
    cmd_buf->buildAccelerationStructures({ as_build_geometry_info }, as_build_range_infos);
    device->submitAndWaitTransientCommandBuffer();

    bl_data_info->as_device_address =
        device->getAccelerationStructureDeviceAddress(bl_data_info->as_handle);
}

void RayTracingShadowTest::initTopLevelDataInfo(
    const std::shared_ptr<renderer::Device>& device) {

    auto bl_data_info =
        std::reinterpret_pointer_cast<BottomLevelDataInfo>(bl_data_info_);
    auto tl_data_info =
        std::reinterpret_pointer_cast<TopLevelDataInfo>(tl_data_info_);

    renderer::TransformMatrix transform_matrix = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f };

    uint32_t primitive_count = 1;

    renderer::AccelerationStructureInstance instance{};
    instance.transform = transform_matrix;
    instance.instance_custom_index = 0;
    instance.mask = 0xFF;
    instance.instance_shader_binding_table_record_offset = 0;
    instance.flags = SET_FLAG_BIT(GeometryInstance, TRIANGLE_FACING_CULL_DISABLE_BIT_KHR);
    instance.acceleration_structure_reference = bl_data_info->as_device_address;

    // Instance buffer
    renderer::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT) |
        SET_FLAG_BIT(BufferUsage, ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        tl_data_info->instance_buffer.buffer,
        tl_data_info->instance_buffer.memory,
        std::source_location::current(),
        sizeof(instance),
        &instance);

    auto as_geometry = std::make_shared<renderer::AccelerationStructureGeometry>();
    as_geometry->flags = SET_FLAG_BIT(Geometry, OPAQUE_BIT_KHR);
    as_geometry->geometry_type = renderer::GeometryType::INSTANCES_KHR;
    as_geometry->geometry.instances.struct_type =
        renderer::StructureType::ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    as_geometry->geometry.instances.array_of_pointers = 0x00;
    as_geometry->geometry.instances.data.device_address =
        tl_data_info->instance_buffer.buffer->getDeviceAddress();
    as_geometry->max_primitive_count = primitive_count;

    // Get size info
    /*
    The pSrcAccelerationStructure, dstAccelerationStructure, and mode members of pBuildInfo are ignored. Any VkDeviceOrHostAddressKHR members of pBuildInfo are ignored by this command, except that the hostAddress member of VkAccelerationStructureGeometryTrianglesDataKHR::transformData will be examined to check if it is NULL.*
    */
    // Get size info
    renderer::AccelerationStructureBuildGeometryInfo as_build_geometry_info{};
    as_build_geometry_info.type = renderer::AccelerationStructureType::TOP_LEVEL_KHR;
    as_build_geometry_info.flags = SET_FLAG_BIT(BuildAccelerationStructure, PREFER_FAST_TRACE_BIT_KHR);
    as_build_geometry_info.geometries.push_back(as_geometry);

    renderer::AccelerationStructureBuildSizesInfo as_build_size_info{};
    as_build_size_info.struct_type = renderer::StructureType::ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    device->getAccelerationStructureBuildSizes(
        renderer::AccelerationStructureBuildType::DEVICE_KHR,
        as_build_geometry_info,
        as_build_size_info);

    device->createBuffer(
        as_build_size_info.as_size,
        SET_FLAG_BIT(BufferUsage, ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        tl_data_info->as_buffer.buffer,
        tl_data_info->as_buffer.memory,
        std::source_location::current());

    tl_data_info->as_handle =
        device->createAccelerationStructure(
            tl_data_info->as_buffer.buffer,
            renderer::AccelerationStructureType::TOP_LEVEL_KHR);

    device->createBuffer(
        as_build_size_info.build_scratch_size,
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        tl_data_info->scratch_buffer.buffer,
        tl_data_info->scratch_buffer.memory,
        std::source_location::current());

    as_build_geometry_info.mode = renderer::BuildAccelerationStructureMode::BUILD_KHR;
    as_build_geometry_info.dst_as = tl_data_info->as_handle;
    as_build_geometry_info.scratch_data.device_address =
        tl_data_info->scratch_buffer.buffer->getDeviceAddress();

    renderer::AccelerationStructureBuildRangeInfo as_build_range_info{};
    as_build_range_info.primitive_count = primitive_count;
    as_build_range_info.primitive_offset = 0;
    as_build_range_info.first_vertex = 0;
    as_build_range_info.transform_offset = 0;

    // Build the acceleration structure on the device via a one-time command buffer submission
    // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
    auto cmd_buf = device->setupTransientCommandBuffer();
    cmd_buf->buildAccelerationStructures({ as_build_geometry_info }, { as_build_range_info });
    device->submitAndWaitTransientCommandBuffer();

    tl_data_info->as_device_address =
        device->getAccelerationStructureDeviceAddress(tl_data_info->as_handle);
}

void RayTracingShadowTest::createRayTracingPipeline(
    const std::shared_ptr<renderer::Device>& device) {

    auto rt_render_info =
        std::reinterpret_pointer_cast<RayTracingRenderingInfo>(rt_render_info_);

    rt_render_info->shader_modules.resize(kNumRtShaders);
    rt_render_info->shader_modules[kRayGenIndex] =
        renderer::helper::loadShaderModule(
            device,
            "ray_tracing/raytracing_shadow/rt_raygen_rgen.spv",
            renderer::ShaderStageFlagBits::RAYGEN_BIT_KHR,
            std::source_location::current());
    rt_render_info->shader_modules[kRayMissIndex] =
        renderer::helper::loadShaderModule(
            device,
            "ray_tracing/raytracing_shadow/rt_miss_rmiss.spv",
            renderer::ShaderStageFlagBits::MISS_BIT_KHR,
            std::source_location::current());
    rt_render_info->shader_modules[kShadowMissIndex] =
        renderer::helper::loadShaderModule(
            device,
            "ray_tracing/raytracing_shadow/rt_shadow_rmiss.spv",
            renderer::ShaderStageFlagBits::MISS_BIT_KHR,
            std::source_location::current());
    rt_render_info->shader_modules[kClosestHitIndex] =
        renderer::helper::loadShaderModule(
            device,
            "ray_tracing/raytracing_shadow/rt_closesthit_rchit.spv",
            renderer::ShaderStageFlagBits::CLOSEST_HIT_BIT_KHR,
            std::source_location::current());

    rt_render_info->shader_groups.resize(kNumRtShaders);
    rt_render_info->shader_groups[kRayGenIndex].type = renderer::RayTracingShaderGroupType::GENERAL_KHR;
    rt_render_info->shader_groups[kRayGenIndex].general_shader = kRayGenIndex;
    rt_render_info->shader_groups[kRayMissIndex].type = renderer::RayTracingShaderGroupType::GENERAL_KHR;
    rt_render_info->shader_groups[kRayMissIndex].general_shader = kRayMissIndex;
    rt_render_info->shader_groups[kShadowMissIndex].type = renderer::RayTracingShaderGroupType::GENERAL_KHR;
    rt_render_info->shader_groups[kShadowMissIndex].general_shader = kShadowMissIndex;
    rt_render_info->shader_groups[kClosestHitIndex].type = renderer::RayTracingShaderGroupType::TRIANGLES_HIT_GROUP_KHR;
    rt_render_info->shader_groups[kClosestHitIndex].closest_hit_shader = kClosestHitIndex;

    rt_render_info->rt_desc_set_layout =
        createRtDescriptorSetLayout(device);
    rt_render_info->rt_pipeline_layout =
        device->createPipelineLayout(
            { rt_render_info->rt_desc_set_layout },
            { },
            std::source_location::current());
    rt_render_info->rt_pipeline =
        device->createPipeline(
            rt_render_info->rt_pipeline_layout,
            rt_render_info->shader_modules,
            rt_render_info->shader_groups,
            std::source_location::current(),
            2);
}

void RayTracingShadowTest::createShaderBindingTables(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::PhysicalDeviceRayTracingPipelineProperties& rt_pipeline_properties,
    const renderer::PhysicalDeviceAccelerationStructureFeatures& as_features) {

    auto rt_render_info =
        std::reinterpret_pointer_cast<RayTracingRenderingInfo>(rt_render_info_);

    const uint32_t handle_size = rt_pipeline_properties.shader_group_handle_size;
    const uint32_t handle_size_aligned = 
        alignedSize(rt_pipeline_properties.shader_group_handle_size,
            rt_pipeline_properties.shader_group_handle_alignment);
    const uint32_t group_count =
        static_cast<uint32_t>(rt_render_info->shader_groups.size());
    const uint32_t sbt_size = group_count * handle_size_aligned;

    std::vector<uint8_t> shader_handle_storage(sbt_size);
    device->getRayTracingShaderGroupHandles(
        rt_render_info->rt_pipeline,
        group_count,
        sbt_size,
        shader_handle_storage.data());

    renderer::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, SHADER_BINDING_TABLE_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        rt_render_info->raygen_shader_binding_table.buffer,
        rt_render_info->raygen_shader_binding_table.memory,
        std::source_location::current(),
        handle_size,
        shader_handle_storage.data());

    renderer::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, SHADER_BINDING_TABLE_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        rt_render_info->miss_shader_binding_table.buffer,
        rt_render_info->miss_shader_binding_table.memory,
        std::source_location::current(),
        handle_size * 2,
        shader_handle_storage.data() + handle_size_aligned);

    renderer::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, SHADER_BINDING_TABLE_BIT_KHR) |
        SET_FLAG_BIT(BufferUsage, SHADER_DEVICE_ADDRESS_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        SET_FLAG_BIT(MemoryAllocate, DEVICE_ADDRESS_BIT),
        rt_render_info->hit_shader_binding_table.buffer,
        rt_render_info->hit_shader_binding_table.memory,
        std::source_location::current(),
        handle_size,
        shader_handle_storage.data() + handle_size_aligned * 3);

    rt_render_info->raygen_shader_sbt_entry.device_address =
        rt_render_info->raygen_shader_binding_table.buffer->getDeviceAddress();
    rt_render_info->raygen_shader_sbt_entry.size = handle_size_aligned;
    rt_render_info->raygen_shader_sbt_entry.stride = handle_size_aligned;

    rt_render_info->miss_shader_sbt_entry.device_address =
        rt_render_info->miss_shader_binding_table.buffer->getDeviceAddress();
    rt_render_info->miss_shader_sbt_entry.size = handle_size_aligned * 2;
    rt_render_info->miss_shader_sbt_entry.stride = handle_size_aligned;

    rt_render_info->hit_shader_sbt_entry.device_address =
        rt_render_info->hit_shader_binding_table.buffer->getDeviceAddress();
    rt_render_info->hit_shader_sbt_entry.size = handle_size_aligned;
    rt_render_info->hit_shader_sbt_entry.stride = handle_size_aligned;
}

void RayTracingShadowTest::createRtResources(
    const std::shared_ptr<renderer::Device>& device) {
    auto rt_render_info =
        std::reinterpret_pointer_cast<RayTracingRenderingInfo>(rt_render_info_);

    device->createBuffer(
        sizeof(UniformData),
        SET_FLAG_BIT(BufferUsage, UNIFORM_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_COHERENT_BIT),
        0,
        rt_render_info->ubo.buffer,
        rt_render_info->ubo.memory,
        std::source_location::current());

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R16G16B16A16_SFLOAT,
        rt_size_,
        1,
        rt_render_info->result_image,
        SET_3_FLAG_BITS(ImageUsage, SAMPLED_BIT, STORAGE_BIT, TRANSFER_SRC_BIT),
        renderer::ImageLayout::GENERAL,
        std::source_location::current());
}

void RayTracingShadowTest::createDescriptorSets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::BufferInfo>& camera_info) {

    auto bl_data_info =
        std::reinterpret_pointer_cast<BottomLevelDataInfo>(bl_data_info_);
    auto tl_data_info =
        std::reinterpret_pointer_cast<TopLevelDataInfo>(tl_data_info_);
    auto rt_render_info =
        std::reinterpret_pointer_cast<RayTracingRenderingInfo>(rt_render_info_);

    rt_render_info->rt_desc_set =
        device->createDescriptorSets(
            descriptor_pool,
            rt_render_info->rt_desc_set_layout,
            1)[0];

    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(7);

    renderer::Helper::addOneAccelerationStructure(
        descriptor_writes,
        rt_render_info->rt_desc_set,
        renderer::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        0,
        { tl_data_info->as_handle });

    renderer::Helper::addOneTexture(
        descriptor_writes,
        rt_render_info->rt_desc_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        1,
        nullptr,
        rt_render_info->result_image.view,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        rt_render_info->rt_desc_set,
        renderer::DescriptorType::STORAGE_BUFFER,
        2,
        camera_info->buffer,
        camera_info->buffer->getSize());

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        rt_render_info->rt_desc_set,
        renderer::DescriptorType::STORAGE_BUFFER,
        3,
        game_object_->buffers_[0].buffer,
        game_object_->buffers_[0].buffer->getSize());

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        rt_render_info->rt_desc_set,
        renderer::DescriptorType::STORAGE_BUFFER,
        4,
        game_object_->buffers_[0].buffer,
        game_object_->buffers_[0].buffer->getSize());

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        rt_render_info->rt_desc_set,
        renderer::DescriptorType::STORAGE_BUFFER,
        5,
        bl_data_info->rt_geometry_info_buffer.buffer,
        bl_data_info->rt_geometry_info_buffer.buffer->getSize());

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        rt_render_info->rt_desc_set,
        renderer::DescriptorType::UNIFORM_BUFFER,
        6,
        rt_render_info->ubo.buffer,
        rt_render_info->ubo.buffer->getSize());

    device->updateDescriptorSets(descriptor_writes);
}

void RayTracingShadowTest::draw(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glsl::ViewParams& view_params) {

    auto rt_render_info =
        std::reinterpret_pointer_cast<RayTracingRenderingInfo>(rt_render_info_);

    UniformData uniform_data;
    uniform_data.light_pos = vec4(0.5, 1, 0, 0);

    device->updateBufferMemory(
        rt_render_info->ubo.memory,
        sizeof(uniform_data),
        &uniform_data);

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::RAY_TRACING,
        rt_render_info->rt_pipeline);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::RAY_TRACING,
        rt_render_info->rt_pipeline_layout,
        { rt_render_info->rt_desc_set });

    cmd_buf->traceRays(
        rt_render_info->raygen_shader_sbt_entry,
        rt_render_info->miss_shader_sbt_entry,
        rt_render_info->hit_shader_sbt_entry,
        {0, 0, 0},
        glm::uvec3(rt_size_, 1));
}

void RayTracingShadowTest::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    auto bl_data_info =
        std::reinterpret_pointer_cast<BottomLevelDataInfo>(bl_data_info_);
    auto tl_data_info =
        std::reinterpret_pointer_cast<TopLevelDataInfo>(tl_data_info_);
    auto rt_render_info =
        std::reinterpret_pointer_cast<RayTracingRenderingInfo>(rt_render_info_);

    game_object_->destroy(device);

    bl_data_info->as_buffer.destroy(device);
    bl_data_info->scratch_buffer.destroy(device);
    bl_data_info->rt_geometry_matrix_buffer.destroy(device);
    bl_data_info->rt_geometry_info_buffer.destroy(device);
    device->destroyAccelerationStructure(bl_data_info->as_handle);

    tl_data_info->as_buffer.destroy(device);
    tl_data_info->instance_buffer.destroy(device);
    tl_data_info->scratch_buffer.destroy(device);
    device->destroyAccelerationStructure(tl_data_info->as_handle);

    rt_render_info_->callable_shader_binding_table.destroy(device);
    rt_render_info_->hit_shader_binding_table.destroy(device);
    rt_render_info_->miss_shader_binding_table.destroy(device);
    rt_render_info_->raygen_shader_binding_table.destroy(device);
    rt_render_info_->result_image.destroy(device);
    device->destroyDescriptorSetLayout(rt_render_info->rt_desc_set_layout);
    device->destroyPipelineLayout(rt_render_info->rt_pipeline_layout);
    device->destroyPipeline(rt_render_info->rt_pipeline);
    rt_render_info->ubo.destroy(device);
}

} //namespace ray_tracing
} //namespace engine
