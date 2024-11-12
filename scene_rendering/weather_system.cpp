#include <vector>

#include "renderer/renderer_helper.h"
#include "engine_helper.h"
#include "game_object/terrain.h"
#include "weather_system.h"
#include "shaders/weather/weather_common.glsl.h"

namespace {
namespace er = engine::renderer;
namespace erh = er::helper;

er::WriteDescriptorList addTemperatureInitTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const er::TextureInfo& temp_tex,
    const er::TextureInfo& moisture_tex,
    const er::TextureInfo& pressure_tex) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(3);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEMP_TEX_INDEX,
        nullptr,
        temp_tex.view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_MOISTURE_TEX_INDEX,
        nullptr,
        moisture_tex.view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_PRESSURE_TEX_INDEX,
        nullptr,
        pressure_tex.view,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

er::WriteDescriptorList addAirflowTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& src_temp_tex,
    const er::TextureInfo& src_moisture_tex,
    const er::TextureInfo& src_pressure_tex,
    const er::TextureInfo& dst_temp_tex,
    const er::TextureInfo& dst_moisture_tex,
    const er::TextureInfo& dst_pressure_tex,
    const er::TextureInfo& dst_airflow_tex,
    const std::shared_ptr<er::ImageView>& rock_layer_tex,
    const std::shared_ptr<er::ImageView>& soil_water_layer_tex) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(9);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_TEMP_TEX_INDEX,
        texture_sampler,
        src_temp_tex.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_MOISTURE_TEX_INDEX,
        texture_sampler,
        src_moisture_tex.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_PRESSURE_TEX_INDEX,
        texture_sampler,
        src_pressure_tex.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEMP_TEX_INDEX,
        nullptr,
        dst_temp_tex.view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_MOISTURE_TEX_INDEX,
        nullptr,
        dst_moisture_tex.view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_PRESSURE_TEX_INDEX,
        nullptr,
        dst_pressure_tex.view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_AIRFLOW_TEX_INDEX,
        nullptr,
        dst_airflow_tex.view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ROCK_LAYER_BUFFER_INDEX,
        texture_sampler,
        rock_layer_tex,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SOIL_WATER_LAYER_BUFFER_INDEX,
        texture_sampler,
        soil_water_layer_tex,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

er::WriteDescriptorList addCloudShadowTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& src_moisture_tex,
    const er::TextureInfo& dst_cloud_shadow_tex) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(2);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_MOISTURE_TEX_INDEX,
        texture_sampler,
        src_moisture_tex.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_CLOUD_SHADOW_TEX_INDEX,
        nullptr,
        dst_cloud_shadow_tex.view,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

er::WriteDescriptorList addCloudShadowMergeTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& dst_cloud_shadow_tex) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(1);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_CLOUD_SHADOW_TEX_INDEX,
        nullptr,
        dst_cloud_shadow_tex.view,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

static std::shared_ptr<er::DescriptorSetLayout> createTemperatureInitDescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings(3);
    bindings[0] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            DST_TEMP_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    bindings[1] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            DST_MOISTURE_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    bindings[2] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            DST_PRESSURE_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<er::DescriptorSetLayout> createAirflowUpdateDescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings(9);
    bindings[0] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            DST_TEMP_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    bindings[1] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            DST_MOISTURE_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    bindings[2] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            DST_PRESSURE_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    bindings[3] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_TEMP_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);

    bindings[4] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_MOISTURE_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);

    bindings[5] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_PRESSURE_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);

    bindings[6] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            DST_AIRFLOW_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    bindings[7] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            ROCK_LAYER_BUFFER_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);

    bindings[8] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            SOIL_WATER_LAYER_BUFFER_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<er::DescriptorSetLayout> createCloudShadowUpdateDescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings(2);
    bindings[0] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            SRC_MOISTURE_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);

    bindings[1] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            DST_CLOUD_SHADOW_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<er::DescriptorSetLayout> createCloudShadowMergeDescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    std::vector<er::DescriptorSetLayoutBinding> bindings(1);
    bindings[0] =
        erh::getTextureSamplerDescriptionSetLayoutBinding(
            DST_CLOUD_SHADOW_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

    return device->createDescriptorSetLayout(bindings);
}

} // namespace

namespace engine {
namespace scene_rendering {

WeatherSystem::WeatherSystem(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& rock_layer_tex,
    const std::vector<std::shared_ptr<renderer::ImageView>>& soil_water_layer_tex) {

    for (int i = 0; i < 2; i++) {
        renderer::Helper::create3DTextureImage(
            device,
            renderer::Format::R16_SFLOAT,
            glm::uvec3(
                kAirflowBufferWidth,
                kAirflowBufferWidth,
                kAirflowBufferHeight),
            temp_volume_[i],
            SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
            SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::source_location::current());

        renderer::Helper::create3DTextureImage(
            device,
            renderer::Format::R16_SFLOAT,
            glm::uvec3(
                kAirflowBufferWidth,
                kAirflowBufferWidth,
                kAirflowBufferHeight),
            moisture_volume_[i],
            SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
            SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::source_location::current());

        renderer::Helper::create3DTextureImage(
            device,
            renderer::Format::R16_SFLOAT,
            glm::uvec3(
                kAirflowBufferWidth,
                kAirflowBufferWidth,
                kAirflowBufferHeight),
            pressure_volume_[i],
            SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
            SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            std::source_location::current());
    }

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R32_UINT,
        glm::uvec2(
            kAirflowBufferWidth * 8,
            kAirflowBufferWidth * 8),
        1,
        temp_ground_airflow_info_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R32G32_UINT,
        glm::uvec2(
            kAirflowBufferWidth,
            kAirflowBufferWidth),
        1,
        ground_airflow_info_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    renderer::Helper::create3DTextureImage(
        device,
        renderer::Format::R8G8B8A8_UNORM,
        glm::uvec3(
            kAirflowBufferWidth,
            kAirflowBufferWidth,
            kAirflowBufferHeight),
        airflow_volume_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    uint32_t buffer_height = kAirflowBufferHeight;
    renderer::Helper::create3DTextureImage(
        device,
        renderer::Format::R16_SFLOAT,
        glm::uvec3(
            kAirflowBufferWidth,
            kAirflowBufferWidth,
            buffer_height),
        cloud_shadow_volume_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    temperature_init_desc_set_layout_ = createTemperatureInitDescSetLayout(device);
    airflow_desc_set_layout_ = createAirflowUpdateDescSetLayout(device);
    cloud_shadow_desc_set_layout_ = createCloudShadowUpdateDescSetLayout(device);
    cloud_shadow_merge_desc_set_layout_ = createCloudShadowMergeDescSetLayout(device);

    recreate(
        device,
        descriptor_pool,
        global_desc_set_layouts,
        texture_sampler,
        rock_layer_tex,
        soil_water_layer_tex);
}

void WeatherSystem::recreate(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& rock_layer_tex,
    const std::vector<std::shared_ptr<renderer::ImageView>>& soil_water_layer_tex) {

    erh::releasePipelineLayout(device, temperature_init_pipeline_layout_);
    erh::releasePipeline(device, temperature_init_pipeline_);
    erh::releasePipelineLayout(device, airflow_pipeline_layout_);
    erh::releasePipeline(device, airflow_pipeline_);
    erh::releasePipelineLayout(device, cloud_shadow_pipeline_layout_);
    erh::releasePipeline(device, cloud_shadow_init_pipeline_);
    erh::releasePipelineLayout(device, cloud_shadow_merge_pipeline_layout_);
    erh::releasePipeline(device, cloud_shadow_merge_pipeline_);

    temperature_init_tex_desc_set_ = nullptr;
    temperature_init_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            temperature_init_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto temperature_init_texture_descs = addTemperatureInitTextures(
        temperature_init_tex_desc_set_,
        temp_volume_[0],
        moisture_volume_[0],
        pressure_volume_[0]);
    device->updateDescriptorSets(temperature_init_texture_descs);

    for (int dbuf_idx = 0; dbuf_idx < 2; dbuf_idx++) {
        airflow_tex_desc_set_[dbuf_idx] = nullptr;
        airflow_tex_desc_set_[dbuf_idx] =
            device->createDescriptorSets(
                descriptor_pool,
                airflow_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto airflow_texture_descs = addAirflowTextures(
            airflow_tex_desc_set_[dbuf_idx],
            texture_sampler,
            temp_volume_[1-dbuf_idx],
            moisture_volume_[1 - dbuf_idx],
            pressure_volume_[1 - dbuf_idx],
            temp_volume_[dbuf_idx],
            moisture_volume_[dbuf_idx],
            pressure_volume_[dbuf_idx],
            airflow_volume_,
            rock_layer_tex,
            soil_water_layer_tex[dbuf_idx]);
        device->updateDescriptorSets(airflow_texture_descs);

        cloud_shadow_tex_desc_set_[dbuf_idx] = nullptr;
        cloud_shadow_tex_desc_set_[dbuf_idx] =
            device->createDescriptorSets(
                descriptor_pool,
                cloud_shadow_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto cloud_shadow_texture_descs = addCloudShadowTextures(
            cloud_shadow_tex_desc_set_[dbuf_idx],
            texture_sampler,
            moisture_volume_[dbuf_idx],
            cloud_shadow_volume_);
        device->updateDescriptorSets(cloud_shadow_texture_descs);
    }

    cloud_shadow_merge_tex_desc_set_ = nullptr;
    cloud_shadow_merge_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            cloud_shadow_merge_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto cloud_shadow_merge_tex_descs = addCloudShadowMergeTextures(
        cloud_shadow_merge_tex_desc_set_,
        texture_sampler,
        cloud_shadow_volume_);
    device->updateDescriptorSets(cloud_shadow_merge_tex_descs);

    temperature_init_pipeline_layout_ =
        erh::createComputePipelineLayout(
            device,
            { temperature_init_desc_set_layout_ },
            sizeof(glsl::AirflowUpdateParams));

    temperature_init_pipeline_ = erh::createComputePipeline(
        device,
        temperature_init_pipeline_layout_,
        "weather/temperature_init_comp.spv",
        std::source_location::current());

    airflow_pipeline_layout_ =
        erh::createComputePipelineLayout(
            device,
            { airflow_desc_set_layout_ },
            sizeof(glsl::AirflowUpdateParams));

    airflow_pipeline_ = erh::createComputePipeline(
        device,
        airflow_pipeline_layout_,
        "weather/airflow_update_comp.spv",
        std::source_location::current());

    cloud_shadow_pipeline_layout_ =
        erh::createComputePipelineLayout(
            device,
            { cloud_shadow_desc_set_layout_ },
            sizeof(glsl::CloudShadowParams));

    cloud_shadow_init_pipeline_ = erh::createComputePipeline(
        device,
        cloud_shadow_pipeline_layout_,
        "weather/cloud_shadow_init_comp.spv",
        std::source_location::current());

    cloud_shadow_merge_pipeline_layout_ =
        erh::createComputePipelineLayout(
            device,
            { cloud_shadow_merge_desc_set_layout_ },
            sizeof(glsl::CloudShadowParams));

    cloud_shadow_merge_pipeline_ = erh::createComputePipeline(
        device,
        cloud_shadow_merge_pipeline_layout_,
        "weather/cloud_shadow_merge_comp.spv",
        std::source_location::current());
}

// update air flow buffer.
void WeatherSystem::initTemperatureBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {

    erh::transitMapTextureToStoreImage(
        cmd_buf,
        { temp_volume_[0].image,
          moisture_volume_[0].image,
          pressure_volume_[0].image });

    auto w = static_cast<uint32_t>(kAirflowBufferWidth);
    auto h = static_cast<uint32_t>(kAirflowBufferHeight);
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, temperature_init_pipeline_);
    glsl::AirflowUpdateParams airflow_params = {};
    airflow_params.world_min =
        glm::vec3(-kCloudMapSize / 2.0f, -kCloudMapSize / 2.0f, kAirflowLowHeight);
    airflow_params.world_range =
        glm::vec3(kCloudMapSize / 2.0f, kCloudMapSize / 2.0f, kAirflowMaxHeight) - airflow_params.world_min;
    airflow_params.inv_size = glm::vec3(1.0f / w, 1.0f / w, 1.0f / h);
    airflow_params.size = glm::uvec3(w, w, h);
    airflow_params.controls = {0};
    airflow_params.controls.sea_level_temperature = 30.0f;
    airflow_params.current_time = 0;

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        temperature_init_pipeline_layout_,
        &airflow_params,
        sizeof(airflow_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        temperature_init_pipeline_layout_,
        { temperature_init_tex_desc_set_ });

    cmd_buf->dispatch(
        (w + 7) / 8,
        (w + 7) / 8,
        h);

    erh::transitMapTextureFromStoreImage(
        cmd_buf,
        { temp_volume_[0].image,
          moisture_volume_[0].image,
          pressure_volume_[0].image});
}

// update air flow buffer.
void WeatherSystem::updateAirflowBuffer(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glsl::WeatherControl& weather_controls,
    const float& global_flow_angle,
    const float& global_flow_speed,
    const float& moist_to_pressure_ratio,
    int dbuf_idx,
    float current_time) {

    erh::transitMapTextureToStoreImage(
        cmd_buf,
        { temp_volume_[dbuf_idx].image,
          moisture_volume_[dbuf_idx].image,
          pressure_volume_[dbuf_idx].image,
          airflow_volume_.image});

    auto w = static_cast<uint32_t>(kAirflowBufferWidth);
    auto h = static_cast<uint32_t>(kAirflowBufferHeight);
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, airflow_pipeline_);
    glsl::AirflowUpdateParams airflow_params = {};
    airflow_params.world_min =
        glm::vec3(-kCloudMapSize / 2.0f, -kCloudMapSize / 2.0f, kAirflowLowHeight);
    airflow_params.world_range =
        glm::vec3(kCloudMapSize / 2.0f, kCloudMapSize / 2.0f, kAirflowMaxHeight) - airflow_params.world_min;
    airflow_params.inv_size = glm::vec3(1.0f / w, 1.0f / w, 1.0f / h);
    airflow_params.size = glm::ivec3(w, w, h);
    airflow_params.controls = weather_controls;
    airflow_params.current_time = current_time;
    airflow_params.global_flow_angle = global_flow_angle;
    airflow_params.global_flow_scale = global_flow_speed / kCloudMapSize * w;
    airflow_params.moist_to_pressure_ratio = moist_to_pressure_ratio;

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        airflow_pipeline_layout_,
        &airflow_params,
        sizeof(airflow_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        airflow_pipeline_layout_,
        { airflow_tex_desc_set_[dbuf_idx] });

    cmd_buf->dispatch(
        (w + 7) / 8,
        (w + 7) / 8,
        (h + 15) / 16);

    erh::transitMapTextureFromStoreImage(
        cmd_buf,
        { temp_volume_[dbuf_idx].image,
          moisture_volume_[dbuf_idx].image,
          pressure_volume_[dbuf_idx].image,
          airflow_volume_.image });
}

void WeatherSystem::updateCloudShadow(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glm::vec3& sun_dir,
    const float& light_ext_factor,
    const int& dbuf_idx,
    const float& current_time) {

    auto w = static_cast<uint32_t>(kAirflowBufferWidth);
    auto h = static_cast<uint32_t>(kAirflowBufferHeight);
    // create light passing each layer.
    {
        erh::transitMapTextureToStoreImage(
            cmd_buf,
            { cloud_shadow_volume_.image });

        cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, cloud_shadow_init_pipeline_);
        glsl::CloudShadowParams params = {};
        params.world_min =
            glm::vec3(-kCloudMapSize / 2.0f, -kCloudMapSize / 2.0f, kAirflowLowHeight);
        params.world_range =
            glm::vec3(kCloudMapSize / 2.0f, kCloudMapSize / 2.0f, kAirflowMaxHeight) - params.world_min;
        params.inv_world_range = 1.0f / params.world_range;
        params.inv_size = glm::vec3(1.0f / w, 1.0f / w, 1.0f / h);
        params.size = glm::ivec3(w, w, h);
        params.current_time = current_time;
        params.sun_dir = sun_dir;
        params.light_ext_factor = light_ext_factor;

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            cloud_shadow_pipeline_layout_,
            { cloud_shadow_tex_desc_set_[dbuf_idx] });

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            cloud_shadow_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->dispatch(
            (w + 7) / 8,
            (w + 7) / 8,
            h);

        erh::transitMapTextureFromStoreImage(
            cmd_buf,
            { cloud_shadow_volume_.image });
    }

    // merge two layers to one combined layer.
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::COMPUTE, cloud_shadow_merge_pipeline_);
    for (uint32_t i_layer = 0; i_layer < kAirflowBufferBitCount-1; i_layer++) {
        erh::transitMapTextureToStoreImage(
            cmd_buf,
            { cloud_shadow_volume_.image });

        glsl::CloudShadowParams params = {};
        params.world_min =
            glm::vec3(-kCloudMapSize / 2.0f, -kCloudMapSize / 2.0f, kAirflowLowHeight);
        params.world_range =
            glm::vec3(kCloudMapSize / 2.0f, kCloudMapSize / 2.0f, kAirflowMaxHeight) - params.world_min;
        params.inv_world_range = 1.0f / params.world_range;
        params.inv_size = glm::vec3(1.0f / w, 1.0f / w, 1.0f / h);
        params.size = glm::ivec3(w, w, h);
        params.current_time = current_time;
        params.sun_dir = sun_dir;
        params.light_ext_factor = light_ext_factor;
        params.layer_idx = i_layer;

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            cloud_shadow_merge_pipeline_layout_,
            { cloud_shadow_merge_tex_desc_set_ });

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            cloud_shadow_merge_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->dispatch(
            (w + 7) / 8,
            (w + 7) / 8,
            h >> 1);

        erh::transitMapTextureFromStoreImage(
            cmd_buf,
            { cloud_shadow_volume_.image });
    }
}

void WeatherSystem::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    for (int i = 0; i < 2; i++) {
        temp_volume_[i].destroy(device);
        moisture_volume_[i].destroy(device);
        pressure_volume_[i].destroy(device);
    }
    temp_ground_airflow_info_.destroy(device);
    ground_airflow_info_.destroy(device);
    airflow_volume_.destroy(device);
    cloud_shadow_volume_.destroy(device);
    device->destroyDescriptorSetLayout(airflow_desc_set_layout_);
    device->destroyPipelineLayout(airflow_pipeline_layout_);
    device->destroyPipeline(airflow_pipeline_);
    device->destroyDescriptorSetLayout(temperature_init_desc_set_layout_);
    device->destroyPipelineLayout(temperature_init_pipeline_layout_);
    device->destroyPipeline(temperature_init_pipeline_);
    device->destroyDescriptorSetLayout(cloud_shadow_desc_set_layout_);
    device->destroyPipelineLayout(cloud_shadow_pipeline_layout_);
    device->destroyPipeline(cloud_shadow_init_pipeline_);
    device->destroyDescriptorSetLayout(cloud_shadow_merge_desc_set_layout_);
    device->destroyPipelineLayout(cloud_shadow_merge_pipeline_layout_);
    device->destroyPipeline(cloud_shadow_merge_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
