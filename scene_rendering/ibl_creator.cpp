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

    // ibl_smooth.comp binds both src_img and dst_img as `imageCube`
    // (STORAGE_IMAGE).  Vulkan ignores VkDescriptorImageInfo::sampler for
    // STORAGE_IMAGE descriptors, so pass nullptr here to match the
    // convention used elsewhere in the codebase
    // (e.g. application.cpp:505/850, conemap_test.cpp:248).  The
    // texture_sampler parameter is retained on the function signature in
    // case a future caller wants to repurpose this helper for a
    // SAMPLED_IMAGE/COMBINED_IMAGE_SAMPLER variant.
    (void)texture_sampler;
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        SRC_TEX_INDEX,
        nullptr,
        src_tex.view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEX_INDEX,
        nullptr,
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

// Bit-reversal Van der Corput permutation of [0, 1024).  Sweeps the 32x32
// dither block in low-discrepancy order: 10-bit reversed index, lower 5 bits
// → x, upper 5 bits → y.  Skydome::ditherOffsetForFrame uses the smaller
// 8x8 block; the two are intentionally NOT in sync any more (the IBL
// convolution gets the slower 1024-frame fill window while the sky envmap
// stays on its tighter 64-frame cycle so sun motion shows up promptly in
// the visible sky).
glm::ivec2 iblDitherOffsetForFrame(uint32_t frame_index) {
    uint32_t i = frame_index & 1023u;
    uint32_t r = 0u;
    for (uint32_t b = 0u; b < 10u; ++b) {
        r |= ((i >> b) & 1u) << (9u - b);
    }
    return glm::ivec2(static_cast<int>(r & 31u),
                      static_cast<int>((r >> 5u) & 31u));
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

    // Diffuse accumulator at 4× linear / 16× area of the consumer-facing
    // diffuse cube.  The mini-buffer Lambertian convolution writes mip 0
    // each frame at the larger size; we then run box-filter mipgen to
    // populate mips 1 and 2 — mip 2 is a proper 4×4-averaged 1× cube,
    // and we blit that to tmp_ibl_diffuse_tex_ for consumers.
    //
    // Why 3 mips and not just LINEAR-blit 4×→1× directly:
    //   Vulkan's LINEAR blit filter uses a 2×2 footprint regardless of
    //   the downsample ratio, so a 4:1 LINEAR blit misses 12 out of every
    //   16 source pixels per output cell — the 4 it does read are biased
    //   toward whatever dither phase the cycle happens to land on, which
    //   shows up as systematic darkening of the consumer-facing cube.
    //   Two sequential 2:1 LINEAR blits via mipgen each cover their full
    //   2×2 source neighbourhood and chain into a true 4×4 box average,
    //   so every accumulator texel contributes to the consumer cube.
    renderer::Helper::createCubemapTexture(
        device,
        cube_render_pass,
        cube_size * 4,
        cube_size * 4,
        3,                      // mips 0 (4×), 1 (2×), 2 (1×)
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

    // ibl diffuse blur compute.
    //
    // Layout decision:
    //   src = rt_ibl_diffuse_tex_   (the EMA accumulator the mini-buffer
    //                                Lambertian convolution reads/writes
    //                                each frame — its full history)
    //   dst = tmp_ibl_diffuse_tex_  (the consumer-facing blurred output
    //                                bound at LAMBERTIAN_ENV_TEX_INDEX
    //                                — see addToGlobalTextures())
    //
    // The blur is therefore a *read-only side-channel* on the EMA: it
    // never writes back into rt_, so successive frames' Monte-Carlo
    // averages don't compound the spatial blur into themselves.  The
    // earlier "blur made the buffer darker every frame" symptom was
    // exactly that feedback loop: the previous descriptor binding had
    // src=tmp_, dst=rt_, so each frame the EMA picked up the blurred
    // result as its starting point and the sequence converged toward a
    // doubly-low-pass-filtered, increasingly desaturated steady state.
    // Swapping src/dst breaks that loop entirely.
    {
        ibl_diffuse_tex_desc_set_ = device->createDescriptorSets(
            descriptor_pool, ibl_comp_desc_set_layout_, 1)[0];

        auto ibl_texture_descs = addIblComputeTextures(
            ibl_diffuse_tex_desc_set_,
            texture_sampler,
            rt_ibl_diffuse_tex_,    // src = EMA history
            tmp_ibl_diffuse_tex_);  // dst = blurred consumer-facing
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
    // LAMBERTIAN_ENV_TEX_INDEX → tmp_ibl_diffuse_tex_ (post-blur output).
    // The on-disk-loaded fallback that some assets use still binds rt_
    // directly (not blurred), but for the live, runtime-convolved cube
    // the consumer reads the blurred side-channel that blurIblMaps()
    // produces each frame from the EMA accumulator in rt_.  See the
    // src/dst comment in createDescriptorSets() for why the blur output
    // doesn't feed back into the accumulator.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        LAMBERTIAN_ENV_TEX_INDEX,
        texture_sampler,
        tmp_ibl_diffuse_tex_.view,
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

    // Sparse 32x32 dither only when the mip is at least one block tall.
    // Smaller mips fall back to a full update each frame - they have at
    // most 16x16 + 8x8 + 4x4 + 2x2 + 1 = 341 texels per face, which is
    // rounding error compared to the convolution work elsewhere.
    const bool sparse = mip_face_size >= 32u;
    const uint32_t dither_stride = sparse ? 32u : 1u;
    const uint32_t mini_size =
        sparse ? (mip_face_size / 32u) : mip_face_size;

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

    // The diffuse accumulator (rt_ibl_diffuse_tex_) is allocated at 4×
    // the consumer-facing size (16× area) with a 3-mip chain.  Each
    // frame:
    //   1. Convolve mip 0 at the larger size (more dither coverage per
    //      cycle).
    //   2. Box-filter mipgen mips 1 and 2 — two sequential 2:1 LINEAR
    //      blits, equivalent to a true 4×4 box filter on mip 0.  This
    //      replaces a single 4:1 LINEAR blit, which only reads a 2×2
    //      footprint per output pixel and so misses 12/16 of the source
    //      data — the cause of the systematic darkening we saw with
    //      direct rt_(4×) → tmp_(1×) blits.
    //   3. Copy mip 2 (now exactly cube_size sized) → tmp_ for consumers.
    const uint32_t accum_size = cube_size * 4;

    dispatchIblMiniMip(
        cmd_buf,
        ibl_mini_pipeline_layout_,
        diffuse_mini_desc_sets_[0],
        rt_ibl_diffuse_tex_.image,
        accum_size,
        /*mip_level*/ 0u,
        /*num_mips */ 1u,
        dither,
        mini_frame_index_);
    // dispatchIblMiniMip leaves rt_ mip 0 in SHADER_READ_ONLY_OPTIMAL.
    // Mips 1 and 2 are still in UNDEFINED on first frame (just allocated)
    // and SHADER_READ_ONLY_OPTIMAL on subsequent frames (left there by
    // the previous frame's mipgen tail).  generateMipmapLevels handles
    // both cases — it uses cur_image_layout for the source mip and
    // transitions destination mips internally.

    // Step 2: box-filter mipgen across mips 0 → 1 → 2.
    renderer::Helper::generateMipmapLevels(
        cmd_buf,
        rt_ibl_diffuse_tex_.image,
        /*mip_count*/ 3,
        accum_size, accum_size,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    // After mipgen, every mip of rt_ is in SHADER_READ_ONLY_OPTIMAL.

    // Step 3: copy mip 2 (1× size) → tmp_ mip 0 for consumers.
    er::ImageResourceInfo as_xfer_src = {
        er::ImageLayout::TRANSFER_SRC_OPTIMAL,
        SET_FLAG_BIT(Access, TRANSFER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };
    er::ImageResourceInfo as_xfer_dst = {
        er::ImageLayout::TRANSFER_DST_OPTIMAL,
        SET_FLAG_BIT(Access, TRANSFER_WRITE_BIT),
        SET_FLAG_BIT(PipelineStage, TRANSFER_BIT) };

    // Only mip 2 of rt_ needs to flip to TRANSFER_SRC (mips 0/1 stay
    // in SHADER_READ_ONLY).
    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        er::Helper::getImageAsShaderSampler(),
        as_xfer_src,
        /*baseMip*/ 2, /*mipCount*/ 1, 0, 6);
    cmd_buf->addImageBarrier(
        tmp_ibl_diffuse_tex_.image,
        er::Helper::getImageAsSource(),
        as_xfer_dst,
        0, 1, 0, 6);

    er::ImageCopyInfo copy_region{};
    copy_region.src_subresource.aspect_mask      = SET_FLAG_BIT(ImageAspect, COLOR_BIT);
    copy_region.src_subresource.mip_level        = 2;
    copy_region.src_subresource.base_array_layer = 0;
    copy_region.src_subresource.layer_count      = 6;
    copy_region.src_offset = glm::ivec3(0, 0, 0);
    copy_region.dst_subresource = copy_region.src_subresource;
    copy_region.dst_subresource.mip_level = 0;
    copy_region.dst_offset = glm::ivec3(0, 0, 0);
    copy_region.extent     = glm::uvec3(cube_size, cube_size, 1);

    cmd_buf->copyImage(
        rt_ibl_diffuse_tex_.image,  er::ImageLayout::TRANSFER_SRC_OPTIMAL,
        tmp_ibl_diffuse_tex_.image, er::ImageLayout::TRANSFER_DST_OPTIMAL,
        { copy_region });

    // tmp_ → SHADER_READ_ONLY for consumers (cluster bindless ambient,
    // base.frag IBL ambient, glass OIT).  rt_ mip 2 → SHADER_READ_ONLY
    // for next-frame state symmetry.
    cmd_buf->addImageBarrier(
        tmp_ibl_diffuse_tex_.image,
        as_xfer_dst,
        er::Helper::getImageAsShaderSampler(),
        0, 1, 0, 6);
    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        as_xfer_src,
        er::Helper::getImageAsShaderSampler(),
        /*baseMip*/ 2, /*mipCount*/ 1, 0, 6);
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

    // ── Shallow Gaussian on the diffuse cubemap ────────────────────────
    // Reads the EMA accumulator (rt_ibl_diffuse_tex_) and writes the
    // consumer-facing blurred cubemap (tmp_ibl_diffuse_tex_, bound at
    // LAMBERTIAN_ENV_TEX_INDEX in addToGlobalTextures).  The blur is a
    // pure side-channel: it never writes back into rt_, so successive
    // frames' EMA accumulations don't pick up the spatial blur as their
    // own starting point — see ibl_diffuse_tex_desc_set_ wiring in
    // createDescriptorSets() for the rationale.
    //
    // Specular is intentionally NOT blurred: at the call site's
    // num_mips=1 the GGX importance sampler degenerates to envmap(N),
    // which already matches its consumer's expectation, and adding a
    // blur to the specular side-channel would require either a second
    // mipgen pass or a parallel blurred-specular allocation.

    er::ImageResourceInfo rt_for_compute_read = {
        er::ImageLayout::GENERAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, COMPUTE_SHADER_BIT) };

    // 1. rt_diffuse: SHADER_READ_ONLY → GENERAL (storage-image read).
    //    The mini-buffer convolution left it in SHADER_READ_ONLY at end.
    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        er::Helper::getImageAsShaderSampler(),
        rt_for_compute_read,
        0, 1, 0, 6);
    // 2. tmp_diffuse: any layout → GENERAL (storage-image write).
    //    On first frame this comes from UNDEFINED; subsequently from
    //    SHADER_READ_ONLY left by the previous frame's tail barrier.
    //    Using getImageAsSource() (which encodes UNDEFINED) is safe in
    //    both cases — we overwrite every texel anyway.
    cmd_buf->addImageBarrier(
        tmp_ibl_diffuse_tex_.image,
        er::Helper::getImageAsSource(),
        er::Helper::getImageAsStore(),
        0, 1, 0, 6);

    // 3. Bind, push constants, dispatch.
    glsl::IblComputeParams ibl_comp_params = {};
    ibl_comp_params.size = glm::ivec4(cube_size, cube_size, 0, 0);

    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE,
        blur_comp_pipeline_);
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        ibl_comp_pipeline_layout_,
        &ibl_comp_params,
        sizeof(ibl_comp_params));
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::COMPUTE,
        ibl_comp_pipeline_layout_,
        { ibl_diffuse_tex_desc_set_ });
    cmd_buf->dispatch(
        (cube_size + 7) / 8, (cube_size + 7) / 8, 6);

    // 4. tmp_diffuse → SHADER_READ_ONLY for consumers (cluster bindless
    //    ambient, base.frag IBL ambient, glass OIT).
    cmd_buf->addImageBarrier(
        tmp_ibl_diffuse_tex_.image,
        er::Helper::getImageAsStore(),
        er::Helper::getImageAsShaderSampler(),
        0, 1, 0, 6);
    // 5. rt_diffuse → SHADER_READ_ONLY so the next frame's mini-buffer
    //    EMA path (which transitions GENERAL→GENERAL via getImageAsLoad
    //    Store before its dispatch) starts from a known well-defined
    //    layout, AND so the IBL Debug viewer's SAMPLED descriptors over
    //    rt_'s per-mip face views remain valid for ImGui::Image reads.
    cmd_buf->addImageBarrier(
        rt_ibl_diffuse_tex_.image,
        rt_for_compute_read,
        er::Helper::getImageAsShaderSampler(),
        0, 1, 0, 6);
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
