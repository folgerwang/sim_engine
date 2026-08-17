#include <vector>

#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"
#include "plugins/terrain_gen/terrain.h"
#include "shaders/global_definition.glsl.h"
#include "volume_cloud.h"

namespace {
namespace er = engine::renderer;

er::WriteDescriptorList addBlurImageTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& src_image,
    const std::shared_ptr<er::ImageView>& dst_image,
    const std::shared_ptr<er::ImageView>& depth_image) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(3);

    // envmap texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_TEX_INDEX,
        texture_sampler,
        src_image,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // envmap texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEX_INDEX,
        nullptr,
        dst_image,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_DEPTH_TEX_INDEX,
        texture_sampler,
        depth_image,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

er::WriteDescriptorList addCloudFogTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::Sampler>& point_clamp_texture_sampler,
    const std::shared_ptr<er::ImageView>& src_depth,
    const std::shared_ptr<er::ImageView>& cloud_fog_tex,
    const std::shared_ptr<er::ImageView>& detail_noise_tex,
    const std::shared_ptr<er::ImageView>& rough_noise_tex,
    const std::shared_ptr<er::ImageView>& volume_moist_tex,
    const std::shared_ptr<er::ImageView>& volume_temp_tex,
    const std::shared_ptr<er::ImageView>& cloud_lighting_tex,
    const std::shared_ptr<er::ImageView>& scattering_lut_tex,
    const std::shared_ptr<er::ImageView>& csm_shadow_tex,
    const std::shared_ptr<er::BufferInfo>& runtime_lights_buffer) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(16);

    // envmap texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_MOISTURE_TEX_INDEX,
        texture_sampler,
        volume_moist_tex,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_TEMP_TEX_INDEX,
        texture_sampler,
        volume_temp_tex,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_DEPTH_TEX_INDEX,
        texture_sampler,
        src_depth,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_CLOUD_LIGHTING_TEX_INDEX,
        texture_sampler,
        cloud_lighting_tex,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        DETAIL_NOISE_TEXTURE_INDEX,
        texture_sampler,
        detail_noise_tex,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ROUGH_NOISE_TEXTURE_INDEX,
        texture_sampler,
        rough_noise_tex,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_SCATTERING_LUT_INDEX,
        texture_sampler,
        scattering_lut_tex,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_FOG_CLOUD_INDEX,
        nullptr,        // STORAGE_IMAGE: sampler is ignored by Vulkan; pass
                        // nullptr to match the convention used elsewhere in
                        // the codebase (e.g. volume_cloud.cpp:37,
                        // application.cpp:505/850, conemap_test.cpp:248).
        cloud_fog_tex,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        PERMUTATION_TEXTURE_INDEX,
        point_clamp_texture_sampler,
        er::Helper::getPermutationTexture().view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        PERMUTATION_2D_TEXTURE_INDEX,
        point_clamp_texture_sampler,
        er::Helper::getPermutation2DTexture().view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        GRAD_TEXTURE_INDEX,
        point_clamp_texture_sampler,
        er::Helper::getGradTexture().view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        PERM_GRAD_TEXTURE_INDEX,
        point_clamp_texture_sampler,
        er::Helper::getPermGradTexture().view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        PERM_GRAD_4D_TEXTURE_INDEX,
        point_clamp_texture_sampler,
        er::Helper::getPermGrad4DTexture().view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        GRAD_4D_TEXTURE_INDEX,
        point_clamp_texture_sampler,
        er::Helper::getGrad4DTexture().view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // ── Ground fog: CSM shadow array + runtime-lights UBO ────────────
    // Same layout convention as the PBR global set's DIRECT_SHADOW
    // binding (application.cpp) — the depth array is sampled, not
    // compared, and sits in DEPTH_STENCIL_READ_ONLY between shadow
    // passes.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        VOLFOG_CSM_SHADOW_INDEX,
        texture_sampler,
        csm_shadow_tex,
        er::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);

    er::Helper::addOneBuffer(
        descriptor_writes,
        description_set,
        er::DescriptorType::UNIFORM_BUFFER,
        VOLFOG_LIGHTS_BUF_INDEX,
        runtime_lights_buffer->buffer,
        runtime_lights_buffer->buffer->getSize());

    return descriptor_writes;
}

std::shared_ptr<er::PipelineLayout>
    createBlurImagePipelineLayout(
        const std::shared_ptr<er::Device>& device,
        const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout,
        const std::shared_ptr<er::DescriptorSetLayout>& view_desc_set_layout) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::BlurImageParams);

    return device->createPipelineLayout(
        { desc_set_layout, view_desc_set_layout },
        { push_const_range },
        std::source_location::current());
}

std::shared_ptr<er::PipelineLayout> createRenderCloudFogPipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout,
    const std::shared_ptr<er::DescriptorSetLayout>& view_desc_set_layout) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::VolumeMoistrueParams);

    return device->createPipelineLayout(
        { desc_set_layout , view_desc_set_layout },
        { push_const_range },
        std::source_location::current());
}

} // namespace

namespace engine {
namespace scene_rendering {

VolumeCloud::VolumeCloud(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::Sampler>& point_clamp_texture_sampler,
    const std::shared_ptr<renderer::ImageView>& src_depth,
    const std::shared_ptr<renderer::ImageView>& hdr_color,
    const std::vector<std::shared_ptr<renderer::ImageView>>& moisture_texes,
    const std::vector<std::shared_ptr<renderer::ImageView>>& temp_texes,
    const std::shared_ptr<renderer::ImageView>& cloud_lighting_tex,
    const std::shared_ptr<renderer::ImageView>& scattering_lut_tex,
    const std::shared_ptr<renderer::ImageView>& detail_noise_tex,
    const std::shared_ptr<renderer::ImageView>& rough_noise_tex,
    const std::shared_ptr<renderer::ImageView>& csm_shadow_tex,
    const std::shared_ptr<renderer::BufferInfo>& runtime_lights_buffer,
    const glm::uvec2& display_size) {

    blur_image_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_DEPTH_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER) });

    render_cloud_fog_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_MOISTURE_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEMP_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_DEPTH_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_CLOUD_LIGHTING_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DETAIL_NOISE_TEXTURE_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                ROUGH_NOISE_TEXTURE_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_FOG_CLOUD_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_SCATTERING_LUT_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                PERMUTATION_TEXTURE_INDEX,
                SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                PERMUTATION_2D_TEXTURE_INDEX,
                SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                GRAD_TEXTURE_INDEX,
                SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                PERM_GRAD_TEXTURE_INDEX,
                SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                PERM_GRAD_4D_TEXTURE_INDEX,
                SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                GRAD_4D_TEXTURE_INDEX,
                SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                VOLFOG_CSM_SHADOW_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER),
              renderer::helper::getBufferDescriptionSetLayoutBinding(
                VOLFOG_LIGHTS_BUF_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::UNIFORM_BUFFER)});

    recreate(
        device,
        descriptor_pool,
        view_desc_set_layout,
        texture_sampler,
        point_clamp_texture_sampler,
        src_depth,
        hdr_color,
        moisture_texes,
        temp_texes,
        cloud_lighting_tex,
        scattering_lut_tex,
        detail_noise_tex,
        rough_noise_tex,
        csm_shadow_tex,
        runtime_lights_buffer,
        display_size);
}

void VolumeCloud::recreate(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::Sampler>& point_clamp_texture_sampler,
    const std::shared_ptr<renderer::ImageView>& src_depth,
    const std::shared_ptr<renderer::ImageView>& hdr_color,
    const std::vector<std::shared_ptr<renderer::ImageView>>& moisture_texes,
    const std::vector<std::shared_ptr<renderer::ImageView>>& temp_texes,
    const std::shared_ptr<renderer::ImageView>& cloud_lighting_tex,
    const std::shared_ptr<renderer::ImageView>& scattering_lut_tex,
    const std::shared_ptr<renderer::ImageView>& detail_noise_tex,
    const std::shared_ptr<renderer::ImageView>& rough_noise_tex,
    const std::shared_ptr<renderer::ImageView>& csm_shadow_tex,
    const std::shared_ptr<renderer::BufferInfo>& runtime_lights_buffer,
    const glm::uvec2& display_size) {

    fog_cloud_tex_.destroy(device);
    blurred_fog_cloud_tex_.destroy(device);

    // ── Half-resolution march targets ────────────────────────────────
    // The volumetric march is the most expensive pass in the frame and
    // clouds/fog are low-frequency, so the march and the X-blur run at
    // ceil(display/2).  The final blur_image_y_merge pass still runs at
    // FULL resolution (it writes every HDR pixel) and reads this target
    // through a LINEAR sampler with normalized UVs, which gives the
    // bilinear upsample for free.  Cuts both RGBA16F targets to 1/4
    // VRAM (56.3 -> 14.1 MiB at 1440p) and ~quarters the march cost.
    // Keep RGBA16F: RGB carries in-scattered light, A carries the
    // accumulated transmittance the merge pass composites with.
    const glm::uvec2 fog_tex_size = (display_size + 1u) / 2u;

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R16G16B16A16_SFLOAT,
        fog_tex_size,
        1,
        fog_cloud_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R16G16B16A16_SFLOAT,
        fog_tex_size,
        1,
        blurred_fog_cloud_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    if (blur_image_pipeline_layout_ != nullptr) {
        device->destroyPipelineLayout(blur_image_pipeline_layout_);
        blur_image_pipeline_layout_ = nullptr;
    }
    
    if (blur_image_x_pipeline_ != nullptr) {
        device->destroyPipeline(blur_image_x_pipeline_);
        blur_image_x_pipeline_ = nullptr;
    }

    if (blur_image_y_merge_pipeline_ != nullptr) {
        device->destroyPipeline(blur_image_y_merge_pipeline_);
        blur_image_y_merge_pipeline_ = nullptr;
    }

    // fog cloud
    // persistent pool: allocate once, reuse across resize
    if (!blur_image_x_tex_desc_set_)
        blur_image_x_tex_desc_set_ =
            device->createDescriptorSets(
                descriptor_pool,
                blur_image_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto cloud_texture_descs = addBlurImageTextures(
        blur_image_x_tex_desc_set_,
        texture_sampler,
        fog_cloud_tex_.view,
        blurred_fog_cloud_tex_.view,
        src_depth);
    device->updateDescriptorSets(cloud_texture_descs);

    // persistent pool: allocate once, reuse across resize
    if (!blur_image_y_merge_tex_desc_set_)
        blur_image_y_merge_tex_desc_set_ =
            device->createDescriptorSets(
                descriptor_pool,
                blur_image_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    cloud_texture_descs = addBlurImageTextures(
        blur_image_y_merge_tex_desc_set_,
        texture_sampler,
        blurred_fog_cloud_tex_.view,
        hdr_color,
        src_depth);
    device->updateDescriptorSets(cloud_texture_descs);

    assert(view_desc_set_layout);
    blur_image_pipeline_layout_ =
        createBlurImagePipelineLayout(
            device,
            blur_image_desc_set_layout_,
            view_desc_set_layout);

    blur_image_x_pipeline_ = 
        renderer::helper::createComputePipeline(
            device,
            blur_image_pipeline_layout_,
            "blur_image_x_comp.spv",
            std::source_location::current());

    blur_image_y_merge_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            blur_image_pipeline_layout_,
            "blur_image_y_merge_comp.spv",
            std::source_location::current());

    for (auto dbuf_idx = 0; dbuf_idx < 2; dbuf_idx++) {
        // persistent pool: allocate once, reuse across resize
        if (!render_cloud_fog_desc_set_[dbuf_idx])
            render_cloud_fog_desc_set_[dbuf_idx] =
                device->createDescriptorSets(
                    descriptor_pool,
                    render_cloud_fog_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto render_cloud_fog_texture_descs = addCloudFogTextures(
            render_cloud_fog_desc_set_[dbuf_idx],
            texture_sampler,
            point_clamp_texture_sampler,
            src_depth,
            fog_cloud_tex_.view,
            detail_noise_tex,
            rough_noise_tex,
            moisture_texes[dbuf_idx],
            temp_texes[dbuf_idx],
            cloud_lighting_tex,
            scattering_lut_tex,
            csm_shadow_tex,
            runtime_lights_buffer);
        device->updateDescriptorSets(render_cloud_fog_texture_descs);
    }

    render_cloud_fog_pipeline_layout_ =
        createRenderCloudFogPipelineLayout(
            device,
            render_cloud_fog_desc_set_layout_,
            view_desc_set_layout);

    render_cloud_fog_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            render_cloud_fog_pipeline_layout_,
            "weather/render_volume_cloud_comp.spv",
            std::source_location::current());
}

void VolumeCloud::renderVolumeCloud(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set,
    const std::shared_ptr<renderer::Image>& hdr_color,
    const std::shared_ptr<scene_rendering::Skydome>& skydome,
    const float& view_ext_factor,
    const float& view_ext_exponent,
    const float& ambient_intensity,
    const float& phase_intensity,
    const float& moist_to_pressure_ratio,
    const glm::vec4& noise_weights_0,
    const glm::vec4& noise_weights_1,
    const float& noise_thresold,
    const float& noise_scrolling_speed,
    const glm::vec2& noise_scale,
    const GroundFogParams& ground_fog,
    const glm::uvec2& display_size,
    int dbuf_idx,
    float current_time) {
    // The march and the X-blur run at the fog target's OWN resolution
    // (half display, see recreate) — derive extents from the actual
    // texture so dispatch counts can never drift from the allocation.
    // Only the final Y-blur+merge below runs at display resolution.
    const glm::uvec2 fog_tex_size = glm::uvec2(fog_cloud_tex_.size);

    // render moisture volume.
    {
        renderer::helper::transitMapTextureToStoreImage(
            cmd_buf,
            { fog_cloud_tex_.image });

        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            render_cloud_fog_pipeline_);
        glsl::VolumeMoistrueParams params = {};
        params.world_min = glm::vec2(-kCloudMapSize / 2.0f);
        params.inv_world_range = 1.0f / (glm::vec2(kCloudMapSize / 2.0f) - params.world_min);
        // The shader builds its ray UV as (pixel + 0.5) * inv_screen_size,
        // so feeding the half-res target size keeps UVs spanning the full
        // 0..1 screen; the full-res depth buffer is sampled with those
        // same normalized UVs, so no shader change is needed.
        params.size = fog_tex_size;
        params.inv_screen_size = 1.0f / glm::vec2(fog_tex_size);
        params.time = current_time;
        params.g = skydome->getG();
        params.inv_rayleigh_scale_height = 1.0f / skydome->getRayleighScaleHeight();
        params.inv_mie_scale_height = 1.0f / skydome->getMieScaleHeight();
        params.sun_pos = skydome->getSunDir();
        params.view_ext_factor = view_ext_factor;
        params.view_ext_exponent = view_ext_exponent;
        params.ambient_intensity = ambient_intensity;
        params.phase_intensity = phase_intensity;
        params.pressure_to_moist_ratio = 1.0f / moist_to_pressure_ratio;
        params.noise_weight_0 = noise_weights_0;
        params.noise_weight_1 = noise_weights_1;
        params.noise_thresold = noise_thresold;
        params.noise_speed_scale = noise_scrolling_speed;
        params.noise_scale = noise_scale;
        params.fog_density = ground_fog.density;
        params.fog_height_falloff = ground_fog.height_falloff;
        params.fog_base_y = ground_fog.base_y;
        params.fog_g = ground_fog.g;
        params.fog_sun_intensity = ground_fog.sun_intensity;
        params.fog_ambient_intensity = ground_fog.ambient_intensity;

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            render_cloud_fog_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            render_cloud_fog_pipeline_layout_,
            { render_cloud_fog_desc_set_[dbuf_idx],
              frame_desc_set });

        cmd_buf->dispatch(
            (fog_tex_size.x + 7) / 8,
            (fog_tex_size.y + 7) / 8,
            1);

        renderer::helper::transitMapTextureFromStoreImage(
            cmd_buf,
            { fog_cloud_tex_.image });
    }

    {
        renderer::helper::transitMapTextureToStoreImage(
            cmd_buf,
            { blurred_fog_cloud_tex_.image });

        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            blur_image_x_pipeline_);
        glsl::BlurImageParams params = {};
        // X-blur is half-res -> half-res (fog_cloud -> blurred), so
        // both the push-constant size and the dispatch follow the fog
        // target, not the display.
        params.size = fog_tex_size;
        params.inv_size = 1.0f / glm::vec2(fog_tex_size);

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            blur_image_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            blur_image_pipeline_layout_,
            { blur_image_x_tex_desc_set_,
              frame_desc_set });

        cmd_buf->dispatch(
            (fog_tex_size.x + 63) / 64,
            fog_tex_size.y,
            1);

        renderer::helper::transitMapTextureFromStoreImage(
            cmd_buf,
            { blurred_fog_cloud_tex_.image });
    }

    {
        renderer::helper::transitMapTextureToStoreImage(
            cmd_buf,
            { hdr_color });

        cmd_buf->bindPipeline(
            renderer::PipelineBindPoint::COMPUTE,
            blur_image_y_merge_pipeline_);
        glsl::BlurImageParams params = {};
        // Y-blur + merge WRITES the full-res HDR buffer, so it stays at
        // display resolution.  Its src (blurred_fog_cloud) is half-res:
        // sampled with normalized UVs through the LINEAR sampler bound
        // in recreate, this is exactly a bilinear upsample — and the
        // X/Y blur has already low-passed the signal, so no extra
        // filtering is needed.
        params.size = display_size;
        params.inv_size = 1.0f / glm::vec2(display_size);

        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            blur_image_pipeline_layout_,
            &params,
            sizeof(params));

        cmd_buf->bindDescriptorSets(
            renderer::PipelineBindPoint::COMPUTE,
            blur_image_pipeline_layout_,
            { blur_image_y_merge_tex_desc_set_,
              frame_desc_set });

        cmd_buf->dispatch(
            display_size.x,
            (display_size.y + 63) / 64,
            1);

        renderer::helper::transitMapTextureFromStoreImage(
            cmd_buf,
            { hdr_color },
            renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }
}

void VolumeCloud::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    fog_cloud_tex_.destroy(device);
    blurred_fog_cloud_tex_.destroy(device);
    device->destroyDescriptorSetLayout(blur_image_desc_set_layout_);
    device->destroyPipelineLayout(blur_image_pipeline_layout_);
    device->destroyPipeline(blur_image_x_pipeline_);
    device->destroyPipeline(blur_image_y_merge_pipeline_);
    device->destroyDescriptorSetLayout(render_cloud_fog_desc_set_layout_);
    device->destroyPipelineLayout(render_cloud_fog_pipeline_layout_);
    device->destroyPipeline(render_cloud_fog_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
