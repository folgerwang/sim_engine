#include <iostream>
#include <vector>
#include <chrono>

#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"
#include "shaders/global_definition.glsl.h"
#include "skydome.h"

namespace {
namespace er = engine::renderer;
struct SkyBoxVertex {
    glm::vec3 pos;

    static std::vector<er::VertexInputBindingDescription> getBindingDescription() {
        std::vector<er::VertexInputBindingDescription> binding_description(1);
        binding_description[0].binding = 0;
        binding_description[0].stride = sizeof(SkyBoxVertex);
        binding_description[0].input_rate = er::VertexInputRate::VERTEX;
        return binding_description;
    }

    static std::vector<er::VertexInputAttributeDescription> getAttributeDescriptions() {
        std::vector<er::VertexInputAttributeDescription> attribute_descriptions(1);
        attribute_descriptions[0].binding = 0;
        attribute_descriptions[0].location = 0;
        attribute_descriptions[0].format = er::Format::R32G32B32_SFLOAT;
        attribute_descriptions[0].offset = offsetof(SkyBoxVertex, pos);
        return attribute_descriptions;
    }
};

er::ShaderModuleList getSkyboxShaderModules(
    std::shared_ptr<er::Device> device)
{
    er::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        er::helper::loadShaderModule(
            device,
            "skybox_vert.spv",
            er::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
    shader_modules[1] =
        er::helper::loadShaderModule(
            device,
            "skybox_frag.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());
    return shader_modules;
}

er::ShaderModuleList getCubeSkyboxShaderModules(
    std::shared_ptr<er::Device> device)
{
    er::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        er::helper::loadShaderModule(
            device,
            "full_screen_vert.spv",
            er::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
    shader_modules[1] =
        er::helper::loadShaderModule(
            device,
            "cube_skybox.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());
    return shader_modules;
}

er::BufferInfo createVertexBuffer(
    const std::shared_ptr<er::Device>& device) {
    const std::vector<SkyBoxVertex> vertices = {
        {{-1.0f, -1.0f, -1.0f}},
        {{1.0f, -1.0f, -1.0f}},
        {{-1.0f, 1.0f, -1.0f}},
        {{1.0f, 1.0f, -1.0f}},
        {{-1.0f, -1.0f, 1.0f}},
        {{1.0f, -1.0f, 1.0f}},
        {{-1.0f, 1.0f, 1.0f}},
        {{1.0f, 1.0f, 1.0f}},
    };

    uint64_t buffer_size = sizeof(vertices[0]) * vertices.size();

    er::BufferInfo buffer;
    er::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, VERTEX_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        0,
        buffer.buffer,
        buffer.memory,
        std::source_location::current(),
        buffer_size,
        vertices.data());

    return buffer;
}

er::BufferInfo createIndexBuffer(
    const std::shared_ptr<er::Device>& device) {
    const std::vector<uint16_t> indices = {
        0, 1, 2, 2, 1, 3,
        4, 6, 5, 5, 6, 7,
        0, 4, 1, 1, 4, 5,
        2, 3, 6, 6, 3, 7,
        1, 5, 3, 3, 5, 7,
        0, 2, 4, 4, 2, 6 };

    uint64_t buffer_size =
        sizeof(indices[0]) * indices.size();

    er::BufferInfo buffer;
    er::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, INDEX_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, DEVICE_LOCAL_BIT),
        0,
        buffer.buffer,
        buffer.memory,
        std::source_location::current(),
        buffer_size,
        indices.data());

    return buffer;
}

er::WriteDescriptorList addSkyboxTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& scattering_lut_tex) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(1);

    // envmap texture.
    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_SCATTERING_LUT_INDEX,
        texture_sampler,
        scattering_lut_tex,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

er::WriteDescriptorList addSkyScatteringLutFirstPassTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::ImageView>& sky_scattering_lut_tex) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(1);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_SCATTERING_LUT_INDEX,
        nullptr,
        sky_scattering_lut_tex,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

er::WriteDescriptorList addSkyScatteringLutSumPassTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::ImageView>& sky_scattering_lut_sum_tex,
    const std::shared_ptr<er::ImageView>& sky_scattering_lut_tex) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(2);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_SCATTERING_LUT_SUM_INDEX,
        nullptr,
        sky_scattering_lut_sum_tex,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        SRC_SCATTERING_LUT_INDEX,
        nullptr,
        sky_scattering_lut_tex,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

er::WriteDescriptorList addSkyScatteringLutFinalPassTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::ImageView>& sky_scattering_lut_tex,
    const std::shared_ptr<er::ImageView>& sky_scattering_lut_sum_tex) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(2);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_SCATTERING_LUT_INDEX,
        nullptr,
        sky_scattering_lut_tex,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        SRC_SCATTERING_LUT_SUM_INDEX,
        nullptr,
        sky_scattering_lut_sum_tex,
        er::ImageLayout::GENERAL);

    return descriptor_writes;
}

std::shared_ptr<er::PipelineLayout>
    createSkydomePipelineLayout(
        const std::shared_ptr<er::Device>& device,
        const er::DescriptorSetLayoutList& desc_set_layouts) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::SunSkyParams);

    return device->createPipelineLayout(
        desc_set_layouts,
        { push_const_range },
        std::source_location::current());
}

std::shared_ptr<er::Pipeline> createGraphicsPipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::RenderPass>& render_pass,
    const std::shared_ptr<er::PipelineLayout>& pipeline_layout,
    const er::GraphicPipelineInfo& graphic_pipeline_info,
    const glm::uvec2& display_size) {
#if 0
    VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_LINE_WIDTH
    };

    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;
#endif
    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    auto shader_modules = getSkyboxShaderModules(device);
    std::shared_ptr<er::Pipeline> skybox_pipeline =
        device->createPipeline(
        render_pass,
        pipeline_layout,
        SkyBoxVertex::getBindingDescription(),
        SkyBoxVertex::getAttributeDescriptions(),
        input_assembly,
        graphic_pipeline_info,
        shader_modules,
        display_size,
        std::source_location::current());

    return skybox_pipeline;
}

std::shared_ptr<er::Pipeline> createCubeGraphicsPipeline(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::RenderPass>& render_pass,
    const std::shared_ptr<er::PipelineLayout>& pipeline_layout,
    const er::GraphicPipelineInfo& graphic_pipeline_info,
    const uint32_t& cube_size) {
    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;
    auto cube_shader_modules = getCubeSkyboxShaderModules(device);
    return device->createPipeline(
        render_pass,
        pipeline_layout,
        {}, {},
        input_assembly,
        graphic_pipeline_info,
        cube_shader_modules,
        glm::uvec2(cube_size, cube_size),
        std::source_location::current());
}

std::shared_ptr<er::PipelineLayout> createSkyScatteringLutFirstPassPipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::VolumeMoistrueParams);

    return device->createPipelineLayout(
        { desc_set_layout },
        { push_const_range },
        std::source_location::current());
}

std::shared_ptr<er::PipelineLayout> createSkyScatteringLutPipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout) {
    return device->createPipelineLayout(
        { desc_set_layout},
        { },
        std::source_location::current());
}

// Pipeline layout for the dithered "mini-buffer" sky compute shader.  Push
// constant carries SunSkyMiniParams (sun + atmosphere coefs + dither offset
// + sizes + frame index).  Single descriptor set has the mini-buffer storage
// image (binding SRC_TEX_INDEX), the full-res envmap storage image (binding
// DST_TEX_INDEX), and the precomputed sky scattering LUT sampler used by
// atmosphereLut() (binding SRC_SCATTERING_LUT_INDEX).
std::shared_ptr<er::PipelineLayout> createCubeSkyBoxMiniPipelineLayout(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSetLayout>& desc_set_layout) {
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, COMPUTE_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::SunSkyMiniParams);

    return device->createPipelineLayout(
        { desc_set_layout },
        { push_const_range },
        std::source_location::current());
}

// Bit-reversal Van der Corput permutation of [0, 64).  Yields a low-
// discrepancy ("blue-noise-like") sweep over the 64 (dx, dy) positions
// inside the 8x8 dither block, so the partially-converged envmap looks
// well-distributed instead of marching in scanline order.
glm::ivec2 ditherOffsetForFrame(uint32_t frame_index) {
    uint32_t i = frame_index & 63u;
    uint32_t r = 0u;
    for (uint32_t b = 0u; b < 6u; ++b) {
        r |= ((i >> b) & 1u) << (5u - b);
    }
    return glm::ivec2(static_cast<int>(r & 7u),
                      static_cast<int>((r >> 3u) & 7u));
}

er::WriteDescriptorList addCubeSkyBoxMiniTextures(
    const std::shared_ptr<er::DescriptorSet>& description_set,
    const std::shared_ptr<er::Sampler>& texture_sampler,
    const std::shared_ptr<er::ImageView>& mini_envmap_view,
    const std::shared_ptr<er::ImageView>& rt_envmap_view,
    const std::shared_ptr<er::ImageView>& scattering_lut_view) {
    er::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(3);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        SRC_TEX_INDEX,
        nullptr,
        mini_envmap_view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::STORAGE_IMAGE,
        DST_TEX_INDEX,
        nullptr,
        rt_envmap_view,
        er::ImageLayout::GENERAL);

    er::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_SCATTERING_LUT_INDEX,
        texture_sampler,
        scattering_lut_view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    return descriptor_writes;
}

} // namespace

namespace engine {
namespace scene_rendering {

Skydome::Skydome(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::RenderPass>& cube_render_pass,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::GraphicPipelineInfo& cube_graphic_pipeline_info,
    const renderer::TextureInfo& rt_envmap_tex,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const glm::uvec2& display_size,
    const uint32_t& cube_size) {

    vertex_buffer_ = createVertexBuffer(device);
    index_buffer_ = createIndexBuffer(device);

    // Both LUTs use R32G32_SFLOAT to match the `rg32f` storage-image
    // qualifier on every binding in the LUT shaders.  R16G16 was the
    // previous format and caused two issues:
    //   (a) Vulkan storage-image format mismatch with the shader's
    //       declared rg32f - undefined behaviour without the
    //       shaderStorageImage{Read,Write}WithoutFormat features.
    //   (b) The cross-group cumulative optical depth peaks near 130 km
    //       for chords skimming the planet, overflowing R16F's 65504 max
    //       in the sum-pass output before the final `log2()` compresses
    //       the range.
    // R32G32_SFLOAT covers the full range with full f32 precision.
    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R32G32_SFLOAT,
        glm::uvec2(kAtmosphereScatteringLutWidth, kAtmosphereScatteringLutHeight),
        1,
        sky_scattering_lut_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    renderer::Helper::create2DTextureImage(
        device,
        renderer::Format::R32G32_SFLOAT,
        glm::uvec2(kAtmosphereScatteringLutWidth, kAtmosphereScatteringLutHeight / 64),
        1,
        sky_scattering_lut_sum_tex_,
        SET_FLAG_BIT(ImageUsage, SAMPLED_BIT) |
        SET_FLAG_BIT(ImageUsage, STORAGE_BIT),
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    // Mini-buffer cubemap: cube_size/8 per face, 1 mip, no framebuffers
    // (compute-only target).  Format matches rt_envmap_tex_ so a future
    // copy/resolve path stays trivial.  STORAGE_BIT is required for
    // imageStore(); SAMPLED_BIT is included so the mini-buffer can also be
    // visualized or reused as a coarse approximation.
    {
        const uint32_t mini_size = std::max(1u, cube_size / 8u);
        std::vector<renderer::BufferImageCopyInfo> dump_copies;
        renderer::Helper::createCubemapTexture(
            device,
            cube_render_pass,
            mini_size,
            mini_size,
            1,
            renderer::Format::R16G16B16A16_SFLOAT,
            dump_copies,
            mini_envmap_tex_,
            std::source_location::current());
    }

    skybox_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_SCATTERING_LUT_INDEX) });

    sky_scattering_lut_first_pass_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_SCATTERING_LUT_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE) });

    sky_scattering_lut_sum_pass_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_SCATTERING_LUT_SUM_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_SCATTERING_LUT_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE) });

    sky_scattering_lut_final_pass_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_SCATTERING_LUT_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_SCATTERING_LUT_SUM_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE) });

    // Layout for the dithered mini-buffer sky compute shader: mini-buffer
    // (storage), full-res envmap (storage), scattering LUT (sampled).
    cube_skybox_mini_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                DST_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::STORAGE_IMAGE),
              renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
                SRC_SCATTERING_LUT_INDEX,
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER) });

    recreate(
        device,
        descriptor_pool,
        render_pass,
        view_desc_set_layout,
        graphic_pipeline_info,
        rt_envmap_tex,
        texture_sampler,
        display_size);

    cube_skybox_pipeline_layout_ =
        createSkydomePipelineLayout(
            device,
            { skybox_desc_set_layout_ });

    cube_skybox_pipeline_ = createCubeGraphicsPipeline(
        device,
        cube_render_pass,
        cube_skybox_pipeline_layout_,
        cube_graphic_pipeline_info,
        cube_size);

    sky_scattering_lut_first_pass_pipeline_layout_ =
        createSkyScatteringLutFirstPassPipelineLayout(
            device,
            sky_scattering_lut_first_pass_desc_set_layout_);

    sky_scattering_lut_sum_pass_pipeline_layout_ =
        createSkyScatteringLutPipelineLayout(
            device,
            sky_scattering_lut_sum_pass_desc_set_layout_);

    sky_scattering_lut_final_pass_pipeline_layout_ =
        createSkyScatteringLutPipelineLayout(
            device,
            sky_scattering_lut_final_pass_desc_set_layout_);

    sky_scattering_lut_first_pass_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            sky_scattering_lut_first_pass_pipeline_layout_,
            "sky_scattering_lut_first_pass_comp.spv",
            std::source_location::current());

    sky_scattering_lut_sum_pass_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            sky_scattering_lut_sum_pass_pipeline_layout_,
            "sky_scattering_lut_sum_pass_comp.spv",
            std::source_location::current());

    sky_scattering_lut_final_pass_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            sky_scattering_lut_final_pass_pipeline_layout_,
            "sky_scattering_lut_final_pass_comp.spv",
            std::source_location::current());

    // Dithered mini-buffer sky pipeline.
    cube_skybox_mini_pipeline_layout_ =
        createCubeSkyBoxMiniPipelineLayout(
            device,
            cube_skybox_mini_desc_set_layout_);

    cube_skybox_mini_pipeline_ =
        renderer::helper::createComputePipeline(
            device,
            cube_skybox_mini_pipeline_layout_,
            "cube_skybox_mini_comp.spv",
            std::source_location::current());

    // The descriptor set is allocated/written here; the destination view
    // (rt_envmap_tex.view) is captured now.  bindMiniSkyBoxTargets() may be
    // called again after construction if the envmap texture is recreated.
    bindMiniSkyBoxTargets(device, descriptor_pool, rt_envmap_tex);
}

void Skydome::bindMiniSkyBoxTargets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::TextureInfo& rt_envmap_tex) {
    cube_skybox_mini_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            cube_skybox_mini_desc_set_layout_, 1)[0];

    // We need a sampler for the scattering LUT entry.  Reuse the device's
    // default linear sampler kept by Skydome's caller; here we pass nullptr
    // to addOneTexture for the storage images, but the LUT entry must have
    // a valid sampler.  Vulkan allows COMBINED_IMAGE_SAMPLER's sampler to
    // be drawn from an immutable sampler embedded in the layout binding -
    // however addOneTexture takes a sampler directly, so we need one here.
    //
    // The scattering LUT was created with SHADER_READ_ONLY_OPTIMAL layout
    // and is sampled in the shader through `texture(...)`, which needs a
    // sampler.  We use a quick clamp sampler created on demand.
    static std::shared_ptr<renderer::Sampler> s_clamp_sampler;
    if (!s_clamp_sampler) {
        s_clamp_sampler = device->createSampler(
            er::Filter::LINEAR,
            er::SamplerAddressMode::CLAMP_TO_EDGE,
            er::SamplerMipmapMode::LINEAR, 1.0f,
            std::source_location::current());
    }

    auto descs = addCubeSkyBoxMiniTextures(
        cube_skybox_mini_desc_set_,
        s_clamp_sampler,
        mini_envmap_tex_.view,
        rt_envmap_tex.view,
        sky_scattering_lut_tex_.view);
    device->updateDescriptorSets(descs);
}

void Skydome::recreate(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::TextureInfo& rt_envmap_tex,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const glm::uvec2& display_size) {

    if (skybox_pipeline_layout_ != nullptr) {
        device->destroyPipelineLayout(skybox_pipeline_layout_);
        skybox_pipeline_layout_ = nullptr;
    }
    
    if (skybox_pipeline_ != nullptr) {
        device->destroyPipeline(skybox_pipeline_);
        skybox_pipeline_ = nullptr;
    }

    skybox_tex_desc_set_ = nullptr;

    // skybox
    skybox_tex_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            skybox_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    auto skybox_texture_descs = addSkyboxTextures(
        skybox_tex_desc_set_,
        texture_sampler,
        sky_scattering_lut_tex_.view);
    device->updateDescriptorSets(skybox_texture_descs);

    sky_scattering_lut_first_pass_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            sky_scattering_lut_first_pass_desc_set_layout_, 1)[0];
    auto sky_scattering_lut_first_pass_texture_descs =
        addSkyScatteringLutFirstPassTextures(
            sky_scattering_lut_first_pass_desc_set_,
            sky_scattering_lut_tex_.view);
    device->updateDescriptorSets(
        sky_scattering_lut_first_pass_texture_descs);

    sky_scattering_lut_sum_pass_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            sky_scattering_lut_sum_pass_desc_set_layout_, 1)[0];
    auto sky_scattering_lut_sum_pass_texture_descs =
        addSkyScatteringLutSumPassTextures(
            sky_scattering_lut_sum_pass_desc_set_,
            sky_scattering_lut_sum_tex_.view,
            sky_scattering_lut_tex_.view);
    device->updateDescriptorSets(
        sky_scattering_lut_sum_pass_texture_descs);

    sky_scattering_lut_final_pass_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            sky_scattering_lut_final_pass_desc_set_layout_, 1)[0];
    auto sky_scattering_lut_final_pass_texture_descs =
        addSkyScatteringLutFinalPassTextures(
            sky_scattering_lut_final_pass_desc_set_,
            sky_scattering_lut_tex_.view,
            sky_scattering_lut_sum_tex_.view);
    device->updateDescriptorSets(
        sky_scattering_lut_final_pass_texture_descs);

    assert(view_desc_set_layout);
    skybox_pipeline_layout_ =
        createSkydomePipelineLayout(
            device,
            { skybox_desc_set_layout_,
              view_desc_set_layout });

    skybox_pipeline_ = createGraphicsPipeline(
        device,
        render_pass,
        skybox_pipeline_layout_,
        graphic_pipeline_info,
        display_size);

    // The descriptor pool was recreated by the caller (swap-chain rebuild),
    // so any descriptor set we allocated against the old pool is now stale.
    // Rebind the mini-buffer compute descriptor set against the new pool.
    // The compute pipeline / layout themselves do not depend on display_size
    // and are kept across recreate() calls.
    if (cube_skybox_mini_desc_set_layout_ != nullptr) {
        bindMiniSkyBoxTargets(device, descriptor_pool, rt_envmap_tex);
    }
}

// render skybox.
void Skydome::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::DescriptorSet>& view_desc_set) {
    cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, skybox_pipeline_);
    std::vector<std::shared_ptr<renderer::Buffer>> buffers(1);
    std::vector<uint64_t> offsets(1);
    buffers[0] = vertex_buffer_.buffer;
    offsets[0] = 0;

    cmd_buf->bindVertexBuffers(0, buffers, offsets);
    cmd_buf->bindIndexBuffer(index_buffer_.buffer, 0, renderer::IndexType::UINT16);

    glsl::SunSkyParams sun_sky_params = {};
    sun_sky_params.sun_pos = sun_dir_;
    sun_sky_params.inv_rayleigh_scale_height = 1.0f / rayleigh_scale_height_;
    sun_sky_params.inv_mie_scale_height = 1.0f / mie_scale_height_;
    sun_sky_params.g = g_;

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        skybox_pipeline_layout_,
        &sun_sky_params,
        sizeof(sun_sky_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        skybox_pipeline_layout_,
        {skybox_tex_desc_set_, view_desc_set });

    cmd_buf->drawIndexed(36);
}

void Skydome::drawCubeSkyBox(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const renderer::TextureInfo& rt_envmap_tex,
    const std::vector<er::ClearValue>& clear_values,
    const uint32_t& cube_size)
{
    // generate envmap from skybox.
    cmd_buf->addImageBarrier(
        rt_envmap_tex.image,
        renderer::Helper::getImageAsSource(),
        renderer::Helper::getImageAsColorAttachment(),
        0, 1, 0, 6);

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::GRAPHICS,
        cube_skybox_pipeline_);

    std::vector<renderer::ClearValue> envmap_clear_values(6, clear_values[0]);
    cmd_buf->beginRenderPass(
        render_pass,
        rt_envmap_tex.framebuffers[0],
        glm::uvec2(cube_size),
        envmap_clear_values);

    glsl::SunSkyParams sun_sky_params = {};
    sun_sky_params.sun_pos = sun_dir_;
    sun_sky_params.inv_rayleigh_scale_height = 1.0f / rayleigh_scale_height_;
    sun_sky_params.inv_mie_scale_height = 1.0f / mie_scale_height_;
    sun_sky_params.g = g_;

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        cube_skybox_pipeline_layout_,
        &sun_sky_params,
        sizeof(sun_sky_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        cube_skybox_pipeline_layout_,
        { skybox_tex_desc_set_ });

    cmd_buf->draw(3);

    cmd_buf->endRenderPass();

    uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);

    renderer::Helper::generateMipmapLevels(
        cmd_buf,
        rt_envmap_tex.image,
        num_mips,
        cube_size,
        cube_size,
        renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
}

void Skydome::updateCubeSkyBoxMini(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::TextureInfo& rt_envmap_tex,
    const uint32_t& cube_size) {

    // First-touch handling: on the very first call the envmap image is
    // still in UNDEFINED layout and contains garbage memory, so we use
    // the shader's "fill the whole 8x8 block with this thread's sample"
    // path to initialize every texel in one cheap dispatch.  Every later
    // call assumes mip 0 holds a valid prior frame's contents and uses
    // the regular sparse 1-of-64 dither + scatter path.
    const bool is_first_touch = (mini_frame_index_ == 0u);
    const uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);

    // Full-res envmap mip 0: layout transition to GENERAL.  On first
    // touch the image is in UNDEFINED and we'll overwrite every texel
    // anyway, so getImageAsSource() (UNDEFINED) is the correct src.
    // After that, mip 0 holds the previous frame's data we want to
    // preserve (the shader only overwrites a 1/64 sparse subset), so we
    // come from SHADER_READ_ONLY_OPTIMAL.
    cmd_buf->addImageBarrier(
        rt_envmap_tex.image,
        is_first_touch
            ? renderer::Helper::getImageAsSource()
            : renderer::Helper::getImageAsShaderSampler(),
        renderer::Helper::getImageAsStore(),
        0, 1, 0, 6);

    // Mini-buffer is fully overwritten every frame, so discarding the old
    // contents (UNDEFINED via getImageAsSource) is fine and avoids
    // requiring a known prior layout.
    cmd_buf->addImageBarrier(
        mini_envmap_tex_.image,
        renderer::Helper::getImageAsSource(),
        renderer::Helper::getImageAsStore(),
        0, 1, 0, 6);

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE,
        cube_skybox_mini_pipeline_);

    glsl::SunSkyMiniParams params = {};
    params.sun_pos = sun_dir_;
    params.g = g_;
    params.inv_rayleigh_scale_height = 1.0f / rayleigh_scale_height_;
    params.inv_mie_scale_height = 1.0f / mie_scale_height_;
    params.cube_size = static_cast<int>(cube_size);
    params.mini_size = static_cast<int>(std::max(1u, cube_size / 8u));

    const glm::ivec2 dither = ditherOffsetForFrame(mini_frame_index_);
    params.dither_offset = dither;
    params.frame_index = static_cast<int>(mini_frame_index_);
    params.is_first_touch = is_first_touch ? 1 : 0;

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        cube_skybox_mini_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        cube_skybox_mini_pipeline_layout_,
        { cube_skybox_mini_desc_set_ });

    // Dispatch one thread per mini-buffer texel.  local_size is (8, 8, 1).
    const uint32_t mini_size = static_cast<uint32_t>(params.mini_size);
    const uint32_t group_xy = (mini_size + 7u) / 8u;
    cmd_buf->dispatch(group_xy, group_xy, 6u);

    // Transition the mini-buffer to a sampleable state so the IBL Debug
    // window (and any future passes) can read it.
    cmd_buf->addImageBarrier(
        mini_envmap_tex_.image,
        renderer::Helper::getImageAsStore(),
        renderer::Helper::getImageAsShaderSampler(),
        0, 1, 0, 6);

    // Mip 0 has been partially refreshed in GENERAL layout.  Hand it
    // straight to generateMipmapLevels with cur_image_layout=GENERAL so
    // it issues a proper GENERAL -> TRANSFER_SRC_OPTIMAL barrier on mip 0
    // before reading.  (The earlier code did GENERAL -> UNDEFINED -> ...
    // which is *invalid* per the Vulkan spec - newLayout cannot be
    // UNDEFINED - and on strict drivers leaves mip 0 logically discarded,
    // producing concentric-ring banding patterns in the regenerated
    // mip chain.)
    renderer::Helper::generateMipmapLevels(
        cmd_buf,
        rt_envmap_tex.image,
        num_mips,
        cube_size,
        cube_size,
        renderer::ImageLayout::GENERAL);

    ++mini_frame_index_;
}

void Skydome::update(float latitude, float longtitude, int d, int th, int tm, int ts) {
    float latitude_r = glm::radians(latitude);
    float decline_angle = glm::radians(
        -23.44f * cos(2.0f * PI / 365.0f * (d + 10.0f)));
    float t = th + tm / 60.0f + ts / 3600.0f;
    float h = -15.0f * (t - 12.0f);
    float delta_h = glm::radians(h - longtitude);

    sun_dir_.x = cos(latitude_r) * sin(decline_angle) - sin(latitude_r) * cos(decline_angle) * cos(delta_h);
    sun_dir_.z = cos(decline_angle) * sin(delta_h);
    sun_dir_.y = sin(latitude_r) * sin(decline_angle) + cos(latitude_r) * cos(decline_angle) * cos(delta_h);
}

void Skydome::updateSkyScatteringLut(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {

    if (rayleigh_scale_height_ != lut_rayleigh_scale_height_ ||
        mie_scale_height_ != lut_mie_scale_height_) {
        renderer::helper::transitMapTextureToStoreImage(
            cmd_buf,
            { sky_scattering_lut_tex_.image,
                sky_scattering_lut_sum_tex_.image });

        const uint32_t scattering_lut_height_group =
            (kAtmosphereScatteringLutHeight + kAtmosphereScatteringLutGroupSize - 1) /
            kAtmosphereScatteringLutGroupSize;

        {
            cmd_buf->bindPipeline(
                renderer::PipelineBindPoint::COMPUTE,
                sky_scattering_lut_first_pass_pipeline_);

            glsl::SkyScatteringParams params = {};
            params.inv_rayleigh_scale_height = 1.0f / rayleigh_scale_height_;
            params.inv_mie_scale_height = 1.0f / mie_scale_height_;

            cmd_buf->pushConstants(
                SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
                sky_scattering_lut_first_pass_pipeline_layout_,
                &params,
                sizeof(params));

            cmd_buf->bindDescriptorSets(
                renderer::PipelineBindPoint::COMPUTE,
                sky_scattering_lut_first_pass_pipeline_layout_,
                { sky_scattering_lut_first_pass_desc_set_ });

            cmd_buf->dispatch(
                kAtmosphereScatteringLutWidth,
                scattering_lut_height_group,
                1);
        }

        {
            cmd_buf->bindPipeline(
                renderer::PipelineBindPoint::COMPUTE,
                sky_scattering_lut_sum_pass_pipeline_);

            cmd_buf->bindDescriptorSets(
                renderer::PipelineBindPoint::COMPUTE,
                sky_scattering_lut_sum_pass_pipeline_layout_,
                { sky_scattering_lut_sum_pass_desc_set_ });

            cmd_buf->dispatch(
                kAtmosphereScatteringLutWidth,
                1,
                1);
        }

        {
            cmd_buf->bindPipeline(
                renderer::PipelineBindPoint::COMPUTE,
                sky_scattering_lut_final_pass_pipeline_);

            cmd_buf->bindDescriptorSets(
                renderer::PipelineBindPoint::COMPUTE,
                sky_scattering_lut_final_pass_pipeline_layout_,
                { sky_scattering_lut_final_pass_desc_set_ });

            cmd_buf->dispatch(
                kAtmosphereScatteringLutWidth,
                scattering_lut_height_group,
                1);
        }

        renderer::helper::transitMapTextureFromStoreImage(
            cmd_buf,
            { sky_scattering_lut_tex_.image,
              sky_scattering_lut_sum_tex_.image });

        lut_rayleigh_scale_height_ = rayleigh_scale_height_;
        lut_mie_scale_height_ = mie_scale_height_;
    }
}

void Skydome::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    vertex_buffer_.destroy(device);
    index_buffer_.destroy(device);

    sky_scattering_lut_tex_.destroy(device);
    sky_scattering_lut_sum_tex_.destroy(device);
    mini_envmap_tex_.destroy(device);

    device->destroyDescriptorSetLayout(skybox_desc_set_layout_);
    device->destroyPipelineLayout(skybox_pipeline_layout_);
    device->destroyPipeline(skybox_pipeline_);
    device->destroyPipelineLayout(cube_skybox_pipeline_layout_);
    device->destroyPipeline(cube_skybox_pipeline_);

    device->destroyDescriptorSetLayout(sky_scattering_lut_first_pass_desc_set_layout_);
    device->destroyDescriptorSetLayout(sky_scattering_lut_sum_pass_desc_set_layout_);
    device->destroyDescriptorSetLayout(sky_scattering_lut_final_pass_desc_set_layout_);
    device->destroyPipelineLayout(sky_scattering_lut_first_pass_pipeline_layout_);
    device->destroyPipelineLayout(sky_scattering_lut_sum_pass_pipeline_layout_);
    device->destroyPipelineLayout(sky_scattering_lut_final_pass_pipeline_layout_);
    device->destroyPipeline(sky_scattering_lut_first_pass_pipeline_);
    device->destroyPipeline(sky_scattering_lut_sum_pass_pipeline_);
    device->destroyPipeline(sky_scattering_lut_final_pass_pipeline_);

    // Mini-buffer compute pipeline.
    device->destroyDescriptorSetLayout(cube_skybox_mini_desc_set_layout_);
    device->destroyPipelineLayout(cube_skybox_mini_pipeline_layout_);
    device->destroyPipeline(cube_skybox_mini_pipeline_);
}

}//namespace scene_rendering
}//namespace engine
