#include <iostream>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <chrono>

#include "engine_helper.h"
#include "game_object/game_objects_list.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace ego = engine::game_object;
namespace engine {

namespace {

static renderer::WriteDescriptorList addGameObjectsInfoBuffer(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::BufferInfo& game_object_buffer,
    const renderer::BufferInfo& game_instance_buffer,
    const renderer::BufferInfo& game_camera_buffer,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer,
    const renderer::TextureInfo& water_flow,
    const std::shared_ptr<renderer::ImageView>& airflow_tex) {
    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(7);

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        engine::renderer::DescriptorType::STORAGE_BUFFER,
        GAME_OBJECTS_BUFFER_INDEX,
        game_object_buffer.buffer,
        game_object_buffer.buffer->getSize());

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        engine::renderer::DescriptorType::STORAGE_BUFFER,
        INSTANCE_BUFFER_INDEX,
        game_instance_buffer.buffer,
        game_instance_buffer.buffer->getSize());

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        engine::renderer::DescriptorType::STORAGE_BUFFER,
        CAMERA_OBJECT_BUFFER_INDEX,
        game_camera_buffer.buffer,
        game_camera_buffer.buffer->getSize());

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ROCK_LAYER_BUFFER_INDEX,
        texture_sampler,
        rock_layer.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SOIL_WATER_LAYER_BUFFER_INDEX,
        texture_sampler,
        soil_water_layer.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        WATER_FLOW_BUFFER_INDEX,
        texture_sampler,
        water_flow.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_AIRFLOW_INDEX,
        texture_sampler,
        airflow_tex,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

static std::shared_ptr<renderer::DescriptorSetLayout> createGameObjectsUpdateDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(7);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        GAME_OBJECTS_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[1] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        INSTANCE_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[2] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        CAMERA_OBJECT_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        renderer::DescriptorType::STORAGE_BUFFER);
    bindings[3] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        ROCK_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[4] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SOIL_WATER_LAYER_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[5] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        WATER_FLOW_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));
    bindings[6] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SRC_AIRFLOW_INDEX,
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT));

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::PipelineLayout> createGameObjectsPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::GameObjectsUpdateParams);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

static std::shared_ptr<renderer::DescriptorSetLayout> createGameObjectsDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(1);
    bindings[0] = renderer::helper::getBufferDescriptionSetLayoutBinding(
        OBJECT_INSTANCE_BUFFER_INDEX,
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT),
        renderer::DescriptorType::UNIFORM_BUFFER);

    return device->createDescriptorSetLayout(bindings);
}

renderer::WriteDescriptorList addGameObjectsBuffers(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const renderer::BufferInfo& buffer) {
    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(1);

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::UNIFORM_BUFFER,
        OBJECT_INSTANCE_BUFFER_INDEX,
        buffer.buffer,
        buffer.buffer->getSize());

    return descriptor_writes;
}

} // namespace

namespace game_object {

GameObjectsList::GameObjectsList(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::BufferInfo>& game_camera_buffer,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::TextureInfo& water_flow,
    const std::shared_ptr<renderer::ImageView>& airflow_tex)
    : device_info_(device_info){

    if (!game_objects_buffer_) {
        game_objects_buffer_ = std::make_shared<renderer::BufferInfo>();
        device_info.device->createBuffer(
            __FILE__ + __LINE__,
            kMaxNumObjects * sizeof(glsl::GameObjectInfo),
            SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
            0,
            game_objects_buffer_->buffer,
            game_objects_buffer_->memory);
    }

    if (!game_objects_instance_buffer_) {
        game_objects_instance_buffer_ = std::make_shared<renderer::BufferInfo>();
        device_info.device->createBuffer(
            __FILE__ + __LINE__,
            kMaxNumObjects * sizeof(glsl::InstanceDataInfo),
            SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT),
            SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
            0,
            game_objects_instance_buffer_->buffer,
            game_objects_instance_buffer_->memory);
    }

    if (game_objects_update_desc_set_layout_ == nullptr) {
        game_objects_update_desc_set_layout_ =
            createGameObjectsUpdateDescriptorSetLayout(device_info.device);
    }

    createGameObjectUpdateDescSet(
        device_info.device,
        descriptor_pool,
        texture_sampler,
        game_camera_buffer,
        rock_layer,
        soil_water_layer_0,
        soil_water_layer_1,
        water_flow,
        airflow_tex);

    {
        if (game_objects_update_pipeline_layout_) {
            device_info.device->destroyPipelineLayout(game_objects_update_pipeline_layout_);
            game_objects_update_pipeline_layout_ = nullptr;
        }

        if (game_objects_update_pipeline_layout_ == nullptr) {
            game_objects_update_pipeline_layout_ =
                createGameObjectsPipelineLayout(
                    device_info.device,
                    { game_objects_update_desc_set_layout_ });
        }
    }

    {
        if (game_objects_update_pipeline_) {
            device_info.device->destroyPipeline(game_objects_update_pipeline_);
            game_objects_update_pipeline_ = nullptr;
        }

        if (game_objects_update_pipeline_ == nullptr) {
            assert(game_objects_update_pipeline_layout_);
            game_objects_update_pipeline_ =
                renderer::helper::createComputePipeline(
                    device_info.device,
                    game_objects_update_pipeline_layout_,
                    "update_game_objects_comp.spv");
        }
    }

    if (game_objects_desc_set_layout_ == nullptr) {
        game_objects_desc_set_layout_ =
            createGameObjectsDescriptorSetLayout(device_info.device);
    }

    game_objects_desc_set_ = device_info.device->createDescriptorSets(
        descriptor_pool, game_objects_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto buffer_descs = addGameObjectsBuffers(
        game_objects_desc_set_,
        *game_objects_instance_buffer_);
    device_info.device->updateDescriptorSets(buffer_descs);
}

void GameObjectsList::createGameObjectUpdateDescSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::BufferInfo>& game_camera_buffer,
    const renderer::TextureInfo& rock_layer,
    const renderer::TextureInfo& soil_water_layer_0,
    const renderer::TextureInfo& soil_water_layer_1,
    const renderer::TextureInfo& water_flow,
    const std::shared_ptr<renderer::ImageView>& airflow_tex) {
    // create a global ibl texture descriptor set.
    for (int soil_water = 0; soil_water < 2; soil_water++) {
        // game objects buffer update set.
        game_objects_update_buffer_desc_set_[soil_water] =
            device->createDescriptorSets(
                descriptor_pool, game_objects_update_desc_set_layout_, 1)[0];

        assert(game_objects_buffer_);
        auto write_descs = addGameObjectsInfoBuffer(
            game_objects_update_buffer_desc_set_[soil_water],
            texture_sampler,
            *game_objects_buffer_,
            *game_objects_instance_buffer_,
            *game_camera_buffer,
            rock_layer,
            soil_water == 0 ? soil_water_layer_0 : soil_water_layer_1,
            water_flow,
            airflow_tex);

        device->updateDescriptorSets(write_descs);
    }
}

void GameObjectsList::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    game_objects_buffer_->destroy(device);
    game_objects_instance_buffer_->destroy(device);
    device->destroyDescriptorSetLayout(game_objects_update_desc_set_layout_);
    device->destroyPipelineLayout(game_objects_update_pipeline_layout_);
    device->destroyPipeline(game_objects_update_pipeline_);
    device->destroyDescriptorSetLayout(game_objects_desc_set_layout_);
}

void GameObjectsList::updateGameObjectsBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glm::vec2& world_min,
    const glm::vec2& world_range,
    const glm::vec3& camera_pos,
    float air_flow_strength,
    float water_flow_strength,
    int update_frame_count,
    int soil_water,
    float delta_t,
    bool enble_airflow) {

    cmd_buf->addBufferBarrier(
        game_objects_instance_buffer_->buffer,
        { SET_FLAG_BIT(Access, SHADER_READ_BIT), SET_FLAG_BIT(PipelineStage, VERTEX_SHADER_BIT) },
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        game_objects_instance_buffer_->buffer->getSize());

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, game_objects_update_pipeline_);

    assert(num_objects_ <= kMaxNumObjects);
    glsl::GameObjectsUpdateParams params;
    params.num_objects = num_objects_;
    params.delta_t = delta_t;
    params.frame_count = update_frame_count;
    params.world_min = world_min;
    params.inv_world_range = 1.0f / world_range;
    params.enble_airflow = enble_airflow ? 1 : 0;
    params.air_flow_strength = air_flow_strength;
    params.water_flow_strength = water_flow_strength;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        game_objects_update_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        game_objects_update_pipeline_layout_,
        { game_objects_update_buffer_desc_set_[soil_water] });

    cmd_buf->dispatch((num_objects_ + 63) / 64, 1);

    cmd_buf->addBufferBarrier(
        game_objects_instance_buffer_->buffer,
        { SET_FLAG_BIT(Access, SHADER_WRITE_BIT), SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) },
        { SET_FLAG_BIT(Access, SHADER_READ_BIT), SET_FLAG_BIT(PipelineStage, VERTEX_SHADER_BIT) },
        game_objects_instance_buffer_->buffer->getSize());
}

} // game_object
} // engine