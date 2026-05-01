#include <vector>

#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "ibl_creator.h"

namespace {
namespace er = engine::renderer;
er::ShaderModuleList getIblShaderModules(
    const std::shared_ptr<er::Device>& device)
{
    er::ShaderModuleList shader_modules(5);
    shader_modules[0] =
        er::helper::loadShaderModule(
            device,
            "full_screen_vert.spv",
            er::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
    shader_modules[1] =
        er::helper::loadShaderModule(
            device,
            "panorama_to_cubemap_frag.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());
    shader_modules[2] =
        er::helper::loadShaderModule(
            device,
            "ibl_labertian_frag.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());
    shader_modules[3] =
        er::helper::loadShaderModule(
            device,
            "ibl_ggx_frag.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());
    shader_modules[4] =
        er::helper::loadShaderModule(
            device,
            "ibl_charlie_frag.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());

    return shader_modules;
}

std::shared_ptr<er::PipelineLayout> createCubemapPipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSetLayout>& ibl_desc_set_layout) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::IblParams);

    return device->createPipelineLayout(
        { ibl_desc_set_layout },
        { push_const_range },
        std::source_location::current());
}

er::WriteDescriptorList addPanoramaTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& panorama_tex) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(1);

    // envmap texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        PANORAMA_TEX_INDEX,
        texture_sampler,
        panorama_tex.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

er::WriteDescriptorList addIblTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& rt_envmap_tex)
{
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(1);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ENVMAP_TEX_INDEX,
        texture_sampler,
        rt_envmap_tex.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

er::WriteDescriptorList addIblComputeTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const er::TextureInfo& src_tex,
    const er::TextureInfo& dst_tex)
{
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(2);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        SRC_TEX_INDEX,
        texture_sampler,
        src_tex.view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEX_INDEX,
        texture_sampler,
        dst_tex.view,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

std::shared_ptr<er::PipelineLayout> createCubemapComputePipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSetLayout>& ibl_comp_desc_set_layout)
{
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::IblComputeParams);

    er::DescriptorSetLayoutList desc_set_layouts(1);
    desc_set_layouts[0] = ibl_comp_desc_set_layout;

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

// Pipeline layout for the mini-buffer IBL convolution compute shaders.
// Push constant carries IblMiniParams (roughness, mip metadata, dither
// stride/offset).  Single descriptor set has the source envmap (sampler)
// and the destination per-mip cubemap (storage image).
std::shared_ptr<er::PipelineLayout> createIblMiniPipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSetLayout>& ibl_mini_desc_set_layout)
{
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::IblMiniParams);

    return device->createPipelineLayout(
        { ibl_mini_desc_set_layout },
        { push_const_range },
        std::source_location::current());
}

// Bit-reversal Van der Corput permutation of [0, 64).  Same construction
// used in Skydome::ditherOffsetForFrame - we deliberately keep them in
// sync: Skydome and IblCreator both consume the same dither order so the
// envmap mip 0 and the IBL outputs converge over the same 64-frame window
// (though they advance their own counters independently, so the offsets
// align only at frame 0 - this is fine for steady-state quality).
glm::ivec2 iblDitherOffsetForFrame(uint32_t frame_index) {
    uint32_t i = frame_index & 63u;
    uint32_t r = 0u;
    for (uint32_t b = 0u; b < 6u; ++b) {
        r |= ((i >> b) & 1u) << (5u - b);
    }
    return glm::ivec2(static_cast<int>(r & 7u),
                      static_cast<int>((r >> 3u) & 7u));
}

// Write descriptor entries for one mini-buffer IBL pass:
//   - source envmap (combined image sampler)
//   - destination per-mip cubemap (storage image)
er::WriteDescriptorList addIblMiniTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& envmap_view,
    const std::shared_ptr<er::ImageView>& dst_per_mip_cube_view) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(2);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ENVMAP_TEX_INDEX,
        texture_sampler,
        envmap_view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEX_INDEX,
        nullptr,
        dst_per_mip_cube_view,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

}

namespace engine {
namespace scene_rendering {

IblCreator::IblCreator(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const uint32_t& cube_size) {

    createCubeTextures(
        device,
        cube_render_pass,
        cube_size);

    auto format = er::Format::R8G8B8A8_UNORM;
    helper::createTextureImage(
        device,
        "assets/environments/doge2.hdr",
        format,
        false,
        panorama_tex_,
        std::source_location::current());

    // ibl texture descriptor set layout.
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(1);
        bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(PANORAMA_TEX_INDEX);
        //bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(ENVMAP_TEX_INDEX);

        ibl_desc_set_layout_ =
            device->createDescriptorSetLayout(bindings);
    }

    // ibl compute texture descriptor set layout.
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(2);
        bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(SRC_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);
        bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(DST_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);

        ibl_comp_desc_set_layout_ =
            device->createDescriptorSetLayout(bindings);
    }

    createDescriptorSets(
        device,
        descriptor_pool,
        texture_sampler);

    ibl_comp_pipeline_layout_ =
        createCubemapComputePipelineLayout(
            device,
            ibl_comp_desc_set_layout_);

    blur_comp_pipeline_ =
        er::helper::createComputePipeline(
            device,
            ibl_comp_pipeline_layout_,
            "ibl_smooth_comp.spv",
            std::source_location::current());

    ibl_pipeline_layout_ =
        createCubemapPipelineLayout(
            device,
            ibl_desc_set_layout_);

    createIblGraphicsPipelines(
        device,
        cube_render_pass,
        cube_graphic_pipeline_info,
        cube_size);

    // ── Mini-buffer IBL convolution: descriptor set layout, pipeline
    // layout, and the three per-filter compute pipelines.  Per-mip
    // descriptor sets and per-mip cube views are bound in
    // bindIblMiniTargets, called next.
    {
        std::vector<er::DescriptorSetLayoutBinding> bindings(2);
        bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            ENVMAP_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::COMBINED_IMAGE_SAMPLER);
        bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
            DST_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
            er::DescriptorType::STORAGE_IMAGE);
        ibl_mini_desc_set_layout_ =
            device->createDescriptorSetLayout(bindings);
    }

    ibl_mini_pipeline_layout_ =
        createIblMiniPipelineLayout(device, ibl_mini_desc_set_layout_);

    lambertian_mini_pipeline_ =
        er::helper::createComputePipeline(
            device,
            ibl_mini_pipeline_layout_,
            "ibl_lambertian_mini_comp.spv",
            std::source_location::current());

    ggx_mini_pipeline_ =
        er::helper::createComputePipeline(
            device,
            ibl_mini_pipeline_layout_,
            "ibl_ggx_mini_comp.spv",
            std::source_location::current());

    charlie_mini_pipeline_ =
        er::helper::createComputePipeline(
            device,
            ibl_mini_pipeline_layout_,
            "ibl_charlie_mini_comp.spv",
            std::source_location::current());

    bindIblMiniTargets(device, descriptor_pool, texture_sampler, cube_size);
}

void IblCreator::bindIblMiniTargets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const uint32_t& cube_size) {

    const uint32_t num_mips =
        static_cast<uint32_t>(std::log2(cube_size) + 1);

    // Helper: build per-mip CUBE image views for one IBL output texture.
    // Each view targets a single mip across all 6 faces, so an `imageCube`
    // bound through it imageStore()s into that mip alone.
    auto rebuild_per_mip_cube_views = [&](
        std::vector<std::shared_ptr<er::ImageView>>& dst_views,
        const er::TextureInfo& tex,
        uint32_t mip_count) {
        dst_views.clear();
        dst_views.reserve(mip_count);
        for (uint32_t i = 0; i < mip_count; ++i) {
            dst_views.push_back(
                device->createImageView(
                    tex.image,
                    er::ImageViewType::VIEW_CUBE,
                    er::Format::R16G16B16A16_SFLOAT,
                    SET_FLAG_BIT(ImageAspect, COLOR_BIT),
                    std::source_location::current(),
                    i, 1, 0, 6));
        }
    };

    rebuild_per_mip_cube_views(
        rt_ibl_diffuse_per_mip_cube_views_, rt_ibl_diffuse_tex_, 1);
    rebuild_per_mip_cube_views(
        rt_ibl_specular_per_mip_cube_views_, rt_ibl_specular_tex_, num_mips);
    rebuild_per_mip_cube_views(
        rt_ibl_sheen_per_mip_cube_views_, rt_ibl_sheen_tex_, num_mips);

    // Allocate one descriptor set per (filter, mip) and write the source
    // envmap + per-mip destination view into each.
    auto build_per_mip_desc_sets = [&](
        std::vector<std::shared_ptr<er::DescriptorSet>>& dst_sets,
        const std::vector<std::shared_ptr<er::ImageView>>& dst_views) {
        dst_sets.clear();
        dst_sets.reserve(dst_views.size());
        for (const auto& view : dst_views) {
            auto desc_set = device->createDescriptorSets(
                descriptor_pool, ibl_mini_desc_set_layout_, 1)[0];
            auto writes = addIblMiniTextures(
                desc_set,
                texture_sampler,
                rt_envmap_tex_.view,
                view);
            device->updateDescriptorSets(writes);
            dst_sets.push_back(desc_set);
        }
    };

    build_per_mip_desc_sets(diffuse_mini_desc_sets_,
                            rt_ibl_diffuse_per_mip_cube_views_);
    build_per_mip_desc_sets(specular_mini_desc_sets_,
                            rt_ibl_specular_per_mip_cube_views_);
    build_per_mip_desc_sets(sheen_mini_desc_sets_,
                            rt_ibl_sheen_per_mip_cube_views_);
}

void IblCreator::createCubeTextures(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const uint32_t& cube_size) {
    uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);
    std::vector<renderer::BufferImageCopyInfo> dump_copies;

    renderer::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        cube_size,
        cube_size,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_envmap_tex_,
        std::source_location::current());

    renderer::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        cube_size,
        cube_size,
        1,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_diffuse_tex_,
        std::source_location::current());

    renderer::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        cube_size,
        cube_size,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_specular_tex_,
        std::source_location::current());

    renderer::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        cube_size,
        cube_size,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        tmp_ibl_sheen_tex_,
        std::source_location::current());

    renderer::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        cube_size,
        cube_size,
        1,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_diffuse_tex_,
        std::source_location::current());

    renderer::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        cube_size,
        cube_size,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_specular_tex_,
        std::source_location::current());

    renderer::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        cube_size,
        cube_size,
        num_mips,
        renderer::Format::R16G16B16A16_SFLOAT,
        dump_copies,
        rt_ibl_sheen_tex_,
        std::source_location::current());
}

void IblCreator::createIblGraphicsPipelines(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
    const uint32_t& cube_size) {
    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;
    {
        auto ibl_shader_modules = getIblShaderModules(device);

        envmap_pipeline_ = device->createPipeline(
            cube_render_pass,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            cube_graphic_pipeline_info,
            { ibl_shader_modules[0], ibl_shader_modules[1] },
            glm::uvec2(cube_size, cube_size),
            std::source_location::current());

        lambertian_pipeline_ = device->createPipeline(
            cube_render_pass,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            cube_graphic_pipeline_info,
            { ibl_shader_modules[0], ibl_shader_modules[2] },
            glm::uvec2(cube_size, cube_size),
            std::source_location::current());

        ggx_pipeline_ = device->createPipeline(
            cube_render_pass,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            cube_graphic_pipeline_info,
            { ibl_shader_modules[0], ibl_shader_modules[3] },
            glm::uvec2(cube_size, cube_size),
            std::source_location::current());

        charlie_pipeline_ = device->createPipeline(
            cube_render_pass,
            ibl_pipeline_layout_,
            {}, {},
            input_assembly,
            cube_graphic_pipeline_info,
            { ibl_shader_modules[0], ibl_shader_modules[4] },
            glm::uvec2(cube_size, cube_size),
            std::source_location::current());
    }
}

void IblCreator::createDescriptorSets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {
    // envmap
    {
        // only one descriptor layout.
        envmap_tex_desc_set_ =
            device->createDescriptorSets(
                descriptor_pool, ibl_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addPanoramaTextures(
            envmap_tex_desc_set_,
            texture_sampler,
            panorama_tex_);
        device->updateDescriptorSets(ibl_texture_descs);
    }

    // ibl
    {
        // only one descriptor layout.
        ibl_tex_desc_set_ = device->createDescriptorSets(
            descriptor_pool, ibl_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addIblTextures(
            ibl_tex_desc_set_,
            texture_sampler,
            rt_envmap_tex_);
        device->updateDescriptorSets(ibl_texture_descs);
    }

    // ibl diffuse compute
    {
        // only one descriptor layout.
        ibl_diffuse_tex_desc_set_ = device->createDescriptorSets(
            descriptor_pool, ibl_comp_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addIblComputeTextures(
            ibl_diffuse_tex_desc_set_,
            texture_sampler,
            tmp_ibl_diffuse_tex_,
            rt_ibl_diffuse_tex_);
        device->updateDescriptorSets(ibl_texture_descs);
    }

    // ibl specular compute
    {
        // only one descriptor layout.
        ibl_specular_tex_desc_set_ = device->createDescriptorSets(
            descriptor_pool, ibl_comp_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addIblComputeTextures(
            ibl_specular_tex_desc_set_,
            texture_sampler,
            tmp_ibl_specular_tex_,
            rt_ibl_specular_tex_);
        device->updateDescriptorSets(ibl_texture_descs);
    }

    // ibl sheen compute
    {
        // only one descriptor layout.
        ibl_sheen_tex_desc_set_ = device->createDescriptorSets(
            descriptor_pool, ibl_comp_desc_set_layout_, 1)[0];

        // create a global ibl texture descriptor set.
        auto ibl_texture_descs = addIblComputeTextures(
            ibl_sheen_tex_desc_set_,
            texture_sampler,
            tmp_ibl_sheen_tex_,
            rt_ibl_sheen_tex_);
        device->updateDescriptorSets(ibl_texture_descs);
    }
}

void IblCreator::addToGlobalTextures(
    renderer::WriteDescriptorList& descriptor_writes,
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        LAMBERTIAN_ENV_TEX_INDEX,
        texture_sampler,
        rt_ibl_diffuse_tex_.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        GGX_ENV_TEX_INDEX,
        texture_sampler,
        rt_ibl_specular_tex_.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        CHARLIE_ENV_TEX_INDEX,
        texture_sampler,
        rt_ibl_sheen_tex_.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
}

// generate envmap cubemap from panorama hdr image.
void IblCreator::drawEnvmapFromPanoramaImage(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size) {

    cmd_buf->addImageBarrier(
        rt_envmap_tex_.image,
        er::Helper::getImageAsSource(),
        er::Helper::getImageAsColorAttachment(),
        0, 1, 0, 6);

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::GRAPHICS,
        envmap_pipeline_);

    std::vector<er::ClearValue> envmap_clear_values(6, clear_values[0]);
    cmd_buf->beginRenderPass(
        cube_render_pass,
        rt_envmap_tex_.framebuffers[0],
        glm::uvec2(cube_size, cube_size),
        envmap_clear_values);

    glsl::IblParams ibl_params = {};
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        ibl_pipeline_layout_,
        &ibl_params,
        sizeof(ibl_params));

    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        ibl_pipeline_layout_,
        { envmap_tex_desc_set_ });

    cmd_buf->draw(3);

    cmd_buf->endRenderPass();

    uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);

    er::Helper::generateMipmapLevels(
        cmd_buf,
        rt_envmap_tex_.image,
        num_mips,
        cube_size,
        cube_size,
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
}

// generate ibl diffuse texture.
void IblCreator::createIblDiffuseMap(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size)
{
    cmd_buf->bindPipeline(
        er::PipelineBindPoint::GRAPHICS,
        lambertian_pipeline_);

    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        er::Helper::getImageAsSource(),
        er::Helper::getImageAsColorAttachment(),
        0, 1, 0, 6);

    std::vector<er::ClearValue> envmap_clear_values(6, clear_values[0]);
    cmd_buf->beginRenderPass(cube_render_pass,
        rt_ibl_diffuse_tex_.framebuffers[0],
        glm::uvec2(cube_size, cube_size),
        envmap_clear_values);

    glsl::IblParams ibl_params = {};
    ibl_params.roughness = 1.0f;
    ibl_params.currentMipLevel = 0;
    ibl_params.width = cube_size;
    ibl_params.lodBias = 0;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        ibl_pipeline_layout_,
        &ibl_params,
        sizeof(ibl_params));

    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::GRAPHICS,
        ibl_pipeline_layout_,
        { ibl_tex_desc_set_ });

    cmd_buf->draw(3);
    cmd_buf->endRenderPass();

    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        er::Helper::getImageAsColorAttachment(),
        er::Helper::getImageAsShaderSampler(),
        0, 1, 0, 6);
}

// generate ibl specular texture.
void IblCreator::createIblSpecularMap(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size)
{
    uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);
    cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, ggx_pipeline_);

    for (int i_mip = num_mips - 1; i_mip >= 0; i_mip--) {
        cmd_buf->addImageBarrier(
            rt_ibl_specular_tex_.image,
            er::Helper::getImageAsSource(),
            er::Helper::getImageAsColorAttachment(),
            i_mip, 1, 0, 6);

        uint32_t width = std::max(static_cast<uint32_t>(cube_size) >> i_mip, 1u);
        uint32_t height = std::max(static_cast<uint32_t>(cube_size) >> i_mip, 1u);

        std::vector<er::ClearValue> envmap_clear_values(6, clear_values[0]);
        cmd_buf->beginRenderPass(
            cube_render_pass,
            rt_ibl_specular_tex_.framebuffers[i_mip],
            glm::uvec2(width, height),
            envmap_clear_values);

        glsl::IblParams ibl_params = {};
        ibl_params.roughness = static_cast<float>(i_mip) / static_cast<float>(num_mips - 1);
        ibl_params.currentMipLevel = i_mip;
        ibl_params.width = cube_size;
        ibl_params.lodBias = 0;
        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            ibl_pipeline_layout_,
            &ibl_params,
            sizeof(ibl_params));

        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::GRAPHICS,
            ibl_pipeline_layout_,
            { ibl_tex_desc_set_ });

        cmd_buf->draw(3);

        cmd_buf->endRenderPass();
    }

    cmd_buf->addImageBarrier(
        rt_ibl_specular_tex_.image,
        er::Helper::getImageAsColorAttachment(),
        er::Helper::getImageAsShaderSampler(),
        0, num_mips, 0, 6);
}

// generate ibl sheen texture.
void IblCreator::createIblSheenMap(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size) {
    uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);
    cmd_buf->bindPipeline(er::PipelineBindPoint::GRAPHICS, charlie_pipeline_);

    for (int i_mip = num_mips - 1; i_mip >= 0; i_mip--) {
        uint32_t width = std::max(static_cast<uint32_t>(cube_size) >> i_mip, 1u);
        uint32_t height = std::max(static_cast<uint32_t>(cube_size) >> i_mip, 1u);

        cmd_buf->addImageBarrier(
            rt_ibl_sheen_tex_.image,
            er::Helper::getImageAsSource(),
            er::Helper::getImageAsColorAttachment(),
            i_mip, 1, 0, 6);

        std::vector<er::ClearValue> envmap_clear_values(6, clear_values[0]);
        cmd_buf->beginRenderPass(
            cube_render_pass,
            rt_ibl_sheen_tex_.framebuffers[i_mip],
            glm::uvec2(width, height),
            envmap_clear_values);

        glsl::IblParams ibl_params = {};
        ibl_params.roughness = static_cast<float>(i_mip) / static_cast<float>(num_mips - 1);
        ibl_params.currentMipLevel = i_mip;
        ibl_params.width = cube_size;
        ibl_params.lodBias = 0;
        cmd_buf->pushConstants(
            SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            ibl_pipeline_layout_,
            &ibl_params,
            sizeof(ibl_params));

        cmd_buf->bindDescriptorSets(
            er::PipelineBindPoint::GRAPHICS,
            ibl_pipeline_layout_,
            { ibl_tex_desc_set_ });

        cmd_buf->draw(3);

        cmd_buf->endRenderPass();
    }

    cmd_buf->addImageBarrier(
        rt_ibl_sheen_tex_.image,
        er::Helper::getImageAsColorAttachment(),
        er::Helper::getImageAsShaderSampler(),
        0, num_mips, 0, 6);
}

namespace {

// EMA blend weight applied to each touched texel's new partial estimate.
// Lower = smoother (more samples averaged) but slower to react to sun /
// atmosphere parameter changes.
//
//   alpha = 1/8   ->  ~16 effective touches averaged  (~240 effective samples)
//                     convergence lag ~ 512 frames = 8.5 s @ 60 fps
//                     IBL visibly lags the sky; appears frozen at slow TOD speeds.
//
//   alpha = 1/2   ->  ~3 effective touches averaged   (~48 effective samples)
//                     90% convergence in ~192 frames = 3.2 s @ 60 fps
//
//   alpha = 1/4   ->  ~5 effective touches averaged  (~112 effective samples)
//                     90% convergence in ~320 frames = 5.3 s @ 60 fps
//                     At tod_advance_speed=100 the sun moves ~2.2° during that
//                     window — imperceptible at midday, very small near horizon.
//                     Enough temporal smoothing to suppress 16-sample variance.
//
//   alpha = 1.0   ->  no EMA; immediate replacement each touch
//                     IBL converges in one 64-frame dither cycle (≈1 s @ 60 fps)
//                     Causes visible per-pixel flicker on the diffuse buffer:
//                     the Lambertian kernel draws 16 random hemisphere samples
//                     whose variance is directly visible with no smoothing.
//
// alpha = 1/8 caused "IBL never changes" at tod_advance_speed=100: the sun only
// moves ~3.5° during the 512-frame lag, which is invisible at midday and barely
// perceptible near the horizon.
// alpha = 1.0 caused diffuse flicker: 16 samples/touch have high variance with
// no temporal smoothing.  1/4 is the middle ground: fast enough to track the
// sky, smooth enough to suppress noise.
constexpr float kIblTemporalAlpha = 1.0f / 4.0f;

// Dispatch one compute pass for a single mip of one IBL filter.  Handles
// the dither-stride decision (8 for big mips, 1 for small ones), the
// per-mip layout transitions, and pushing the per-frame stratification
// parameters (frame_index for sample-index advance, temporal_alpha for
// the EMA blend).  The image is left in SHADER_READ_ONLY_OPTIMAL on
// completion so the IBL fragment shaders can sample it next frame.
void dispatchIblMiniMip(
    const std::shared_ptr<er::CommandBuffer>& cmd_buf,
    const std::shared_ptr<er::PipelineLayout>& pipeline_layout,
    const std::shared_ptr<er::DescriptorSet>& desc_set,
    const std::shared_ptr<er::Image>& dst_image,
    uint32_t cube_size,
    uint32_t mip_level,
    uint32_t num_mips,
    const glm::ivec2& dither_offset,
    uint32_t frame_index) {

    const uint32_t mip_face_size =
        std::max(cube_size >> mip_level, 1u);

    // Sparse 8x8 dither only when the mip is at least one block tall.
    // Smaller mips fall back to a full update each frame - they have at
    // most 64 + 16 + 4 + 1 = 85 texels per face, which is rounding error.
    const bool sparse = mip_face_size >= 8u;
    const uint32_t dither_stride = sparse ? 8u : 1u;
    const uint32_t mini_size =
        sparse ? (mip_face_size / 8u) : mip_face_size;

    // Roughness is identical to the existing fragment-shader path: 0 at
    // the top mip, 1 at the smallest.  See createIblSpecularMap.
    const float roughness =
        (num_mips > 1)
            ? (static_cast<float>(mip_level) /
               static_cast<float>(num_mips - 1))
            : 0.0f;

    // First-touch detection: on the very first call (frame_index == 0)
    // the IBL image is still in UNDEFINED and holds garbage.  Tell the
    // shader to use its block-fill path (broadcast one estimate to every
    // texel in the dither block) instead of the EMA-blend path that
    // would otherwise blend against uninitialized memory.  This replaces
    // the heavyweight `createIbl*Map` graphics-pipeline bootstrap.
    const bool is_first_touch = (frame_index == 0u);

    glsl::IblMiniParams params = {};
    params.roughness        = roughness;
    params.currentMipLevel  = static_cast<int>(mip_level);
    params.width            = static_cast<int>(cube_size);
    params.lodBias          = 0.0f;
    params.mip_face_size    = static_cast<int>(mip_face_size);
    params.mini_size        = static_cast<int>(mini_size);
    params.dither_stride    = static_cast<int>(dither_stride);
    params.is_first_touch   = is_first_touch ? 1 : 0;
    params.dither_offset    = sparse ? dither_offset : glm::ivec2(0, 0);
    params.frame_index      = static_cast<int>(frame_index);
    params.temporal_alpha   = kIblTemporalAlpha;

    // Transition the destination mip to GENERAL so we can imageStore() into
    // it.  On first touch the image is in UNDEFINED (just allocated) and
    // we'll overwrite every texel via the block-fill path, so we use
    // getImageAsSource() (which encodes UNDEFINED) as the src layout.
    // After that, the previous frame's contents must be preserved (the
    // shader reads each touched texel for EMA blending), so we come from
    // SHADER_READ_ONLY_OPTIMAL.
    //
    // IMPORTANT: the non-first-touch path does imageLoad + imageStore (EMA).
    // Using getImageAsStore() (dstAccess = SHADER_WRITE only) does NOT make
    // prior writes visible to imageLoad reads — use getImageAsLoadStore()
    // (dstAccess = SHADER_READ | SHADER_WRITE) so the GPU flushes its shader
    // read cache before the compute dispatch.  Without this, imageLoad returns
    // stale/undefined values on frames where the cache was evicted, causing
    // sudden IBL intensity jumps.
    cmd_buf->addImageBarrier(
        dst_image,
        is_first_touch
            ? er::Helper::getImageAsSource()
            : er::Helper::getImageAsShaderSampler(),
        is_first_touch
            ? er::Helper::getImageAsStore()      // first touch: write-only block fill
            : er::Helper::getImageAsLoadStore(), // EMA path: imageLoad + imageStore
        mip_level, 1, 0, 6);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        pipeline_layout,
        &params,
        sizeof(params));

    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        { desc_set });

    const uint32_t group_xy = (mini_size + 7u) / 8u;
    cmd_buf->dispatch(group_xy, group_xy, 6u);

    cmd_buf->addImageBarrier(
        dst_image,
        er::Helper::getImageAsStore(),
        er::Helper::getImageAsShaderSampler(),
        mip_level, 1, 0, 6);
}
} // namespace

void IblCreator::updateIblDiffuseMapMini(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const uint32_t& cube_size) {
    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE,
        lambertian_mini_pipeline_);

    const glm::ivec2 dither = iblDitherOffsetForFrame(mini_frame_index_);

    // Diffuse only convolves mip 0; total per-frame cost is ~1/64 of the
    // old full-res Lambertian fragment-shader pass.
    dispatchIblMiniMip(
        cmd_buf,
        ibl_mini_pipeline_layout_,
        diffuse_mini_desc_sets_[0],
        rt_ibl_diffuse_tex_.image,
        cube_size,
        /*mip_level*/ 0u,
        /*num_mips */ 1u,
        dither,
        mini_frame_index_);
}

void IblCreator::updateIblSpecularMapMini(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const uint32_t& cube_size) {
    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE,
        ggx_mini_pipeline_);

    const uint32_t num_mips =
        static_cast<uint32_t>(std::log2(cube_size) + 1);
    const glm::ivec2 dither = iblDitherOffsetForFrame(mini_frame_index_);

    // Only mip 0 is convolved.  We pass num_mips=1 so the dispatch helper
    // uses roughness = 0 (sharp / mirror-like).  Higher-roughness response
    // comes from the box-filter mipgen below: mip i becomes a blurred
    // version of mip 0, which approximates - but is NOT identical to -
    // the per-mip GGX convolution the original path produced.  This trades
    // a small amount of physical accuracy at high roughness for a
    // ~num_mips-fold cost reduction per frame.
    dispatchIblMiniMip(
        cmd_buf,
        ibl_mini_pipeline_layout_,
        specular_mini_desc_sets_[0],
        rt_ibl_specular_tex_.image,
        cube_size,
        /*mip_level*/ 0u,
        /*num_mips */ 1u,
        dither,
        mini_frame_index_);

    // Box-filter mipgen for mips 1..N-1 ("downsample buffer will be
    // merged nearby pixel in downsample pass").  Mip 0 is left in
    // SHADER_READ_ONLY_OPTIMAL by the dispatch helper, which is exactly
    // what generateMipmapLevels expects as the input mip's current layout.
    if (num_mips > 1) {
        renderer::Helper::generateMipmapLevels(
            cmd_buf,
            rt_ibl_specular_tex_.image,
            num_mips,
            cube_size,
            cube_size,
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }
}

void IblCreator::updateIblSheenMapMini(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const uint32_t& cube_size) {
    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE,
        charlie_mini_pipeline_);

    const uint32_t num_mips =
        static_cast<uint32_t>(std::log2(cube_size) + 1);
    const glm::ivec2 dither = iblDitherOffsetForFrame(mini_frame_index_);

    // See updateIblSpecularMapMini for the rationale: only mip 0 is
    // Charlie-convolved; higher mips come from box-filter mipgen.
    dispatchIblMiniMip(
        cmd_buf,
        ibl_mini_pipeline_layout_,
        sheen_mini_desc_sets_[0],
        rt_ibl_sheen_tex_.image,
        cube_size,
        /*mip_level*/ 0u,
        /*num_mips */ 1u,
        dither,
        mini_frame_index_);

    if (num_mips > 1) {
        renderer::Helper::generateMipmapLevels(
            cmd_buf,
            rt_ibl_sheen_tex_.image,
            num_mips,
            cube_size,
            cube_size,
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    // Advance the dither + sample-stratification frame counter once per
    // "frame's IBL update".  Bumped here (after the last filter) so all
    // three filters - diffuse / specular / sheen - use the same per-frame
    // dither offset and sample stratum.  Caller expects this method to
    // run last in the IBL update sequence (see application.cpp).
    ++mini_frame_index_;
}

void IblCreator::blurIblMaps(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const uint32_t& cube_size) {

    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        er::Helper::getImageAsSource(),
        er::Helper::getImageAsStore(),
        0, 1, 0, 6);

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE,
        blur_comp_pipeline_);

    glsl::IblComputeParams ibl_comp_params = {};
    ibl_comp_params.size = glm::ivec4(cube_size, cube_size, 0, 0);
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        ibl_comp_pipeline_layout_,
        &ibl_comp_params,
        sizeof(ibl_comp_params));

    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::COMPUTE,
        ibl_comp_pipeline_layout_,
        { ibl_diffuse_tex_desc_set_ });

    cmd_buf->dispatch((cube_size + 7) / 8, (cube_size + 7) / 8, 6);

    uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);
    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        er::Helper::getImageAsStore(),
        er::Helper::getImageAsShaderSampler(),
        0, 1, 0, 6);
    cmd_buf->addImageBarrier(
        rt_ibl_specular_tex_.image,
        er::Helper::getImageAsSource(),
        er::Helper::getImageAsShaderSampler(),
        0, num_mips, 0, 6);
    cmd_buf->addImageBarrier(
        rt_ibl_sheen_tex_.image,
        er::Helper::getImageAsSource(),
        er::Helper::getImageAsShaderSampler(),
        0, num_mips, 0, 6);
}

void IblCreator::recreate(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler) {

    createDescriptorSets(
        device,
        descriptor_pool,
        texture_sampler);

    // The descriptor pool was rebuilt by the application (swap-chain
    // recreate); our per-(filter, mip) mini-buffer descriptor sets are
    // therefore stale.  Reallocate them.  Per-mip image views are still
    // valid (they reference the IBL Image objects, which persist), but
    // bindIblMiniTargets() rebuilds them too for symmetry.
    if (ibl_mini_desc_set_layout_ != nullptr) {
        const uint32_t cube_size =
            static_cast<uint32_t>(rt_ibl_specular_tex_.size.x);
        bindIblMiniTargets(device, descriptor_pool, texture_sampler, cube_size);
    }
}

void IblCreator::destroy(
    const std::shared_ptr<renderer::Device>& device)
{
    device->destroyPipeline(blur_comp_pipeline_);
    device->destroyPipelineLayout(ibl_comp_pipeline_layout_);

    device->destroyPipeline(envmap_pipeline_);
    device->destroyPipeline(lambertian_pipeline_);
    device->destroyPipeline(ggx_pipeline_);
    device->destroyPipeline(charlie_pipeline_);
    device->destroyPipelineLayout(ibl_pipeline_layout_);

    device->destroyDescriptorSetLayout(ibl_desc_set_layout_);
    device->destroyDescriptorSetLayout(ibl_comp_desc_set_layout_);

    // Mini-buffer IBL convolution.
    device->destroyPipeline(lambertian_mini_pipeline_);
    device->destroyPipeline(ggx_mini_pipeline_);
    device->destroyPipeline(charlie_mini_pipeline_);
    device->destroyPipelineLayout(ibl_mini_pipeline_layout_);
    device->destroyDescriptorSetLayout(ibl_mini_desc_set_layout_);
    // Per-mip cube views are owned ImageView shared_ptrs - clearing the
    // vectors releases them.  The underlying Images are owned by the
    // TextureInfo objects below.
    rt_ibl_diffuse_per_mip_cube_views_.clear();
    rt_ibl_specular_per_mip_cube_views_.clear();
    rt_ibl_sheen_per_mip_cube_views_.clear();

    rt_envmap_tex_.destroy(device);
    panorama_tex_.destroy(device);
    tmp_ibl_diffuse_tex_.destroy(device);
    tmp_ibl_specular_tex_.destroy(device);
    tmp_ibl_sheen_tex_.destroy(device);
    rt_ibl_diffuse_tex_.destroy(device);
    rt_ibl_specular_tex_.destroy(device);
    rt_ibl_sheen_tex_.destroy(device);

}

}//namespace scene_rendering
}//namespace engine
