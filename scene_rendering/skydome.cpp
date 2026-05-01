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

er::ShaderModuleList getSkyboxEnvmapShaderModules(
    std::shared_ptr<er::Device> device)
{
    er::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        er::helper::loadShaderModule(
            device,
            "skybox_envmap_vert.spv",
            er::ShaderStageFlagBits::VERTEX_BIT,
            std::source_location::current());
    shader_modules[1] =
        er::helper::loadShaderModule(
            device,
            "skybox_envmap_frag.spv",
            er::ShaderStageFlagBits::FRAGMENT_BIT,
            std::source_location::current());
    return shader_modules;
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
    bindMiniSkyBoxTargets(device, descriptor_pool, rt_envmap_tex);
}

void Skydome::updateCubeSkyBoxMini(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::TextureInfo& rt_envmap_tex,
    const uint32_t& cube_size) {

    const uint32_t mini_size     = std::max(1u, cube_size / 8u);
    const uint32_t num_mips      = static_cast<uint32_t>(std::log2(cube_size) + 1);
    const glm::ivec2 dither      = ditherOffsetForFrame(mini_frame_index_);
    const bool is_first_touch    = (mini_frame_index_ == 0u);

    // ── Push-constant payload ─────────────────────────────────────────────
    glsl::SunSkyMiniParams params = {};
    params.sun_pos                    = sun_dir_;
    params.g                          = g_;
    params.inv_rayleigh_scale_height  = 1.0f / rayleigh_scale_height_;
    params.inv_mie_scale_height       = 1.0f / mie_scale_height_;
    params.cube_size                  = static_cast<int>(cube_size);
    params.mini_size                  = static_cast<int>(mini_size);
    params.dither_offset              = dither;
    params.frame_index                = static_cast<int>(mini_frame_index_);
    params.is_first_touch             = is_first_touch ? 1 : 0;

    // ── Transition mini-buffer → GENERAL (imageStore target) ─────────────
    // On first touch the image is in UNDEFINED (just allocated); after that
    // the previous frame's dispatch left it in SHADER_READ_ONLY_OPTIMAL.
    cmd_buf->addImageBarrier(
        mini_envmap_tex_.image,
        is_first_touch
            ? er::Helper::getImageAsSource()        // UNDEFINED
            : er::Helper::getImageAsShaderSampler(),// SHADER_READ_ONLY
        er::Helper::getImageAsStore(),              // → GENERAL
        0, 1, 0, 6);

    // ── Transition full-res envmap mip 0 → GENERAL ───────────────────────
    // The compute shader writes only mip 0 of the envmap (imageStore on the
    // base mip of the view). On first touch the mip is UNDEFINED; on
    // subsequent frames it was left in SHADER_READ_ONLY_OPTIMAL by the
    // previous call's mip-chain generation.
    cmd_buf->addImageBarrier(
        rt_envmap_tex.image,
        is_first_touch
            ? er::Helper::getImageAsSource()
            : er::Helper::getImageAsShaderSampler(),
        er::Helper::getImageAsStore(),
        0, 1, 0, 6);

    // ── Dispatch ──────────────────────────────────────────────────────────
    cmd_buf->bindPipeline(
        er::PipelineBindPoint::COMPUTE,
        cube_skybox_mini_pipeline_);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        cube_skybox_mini_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::COMPUTE,
        cube_skybox_mini_pipeline_layout_,
        { cube_skybox_mini_desc_set_ });

    // Each workgroup covers one 8x8 block of mini texels; z=6 for cube faces.
    const uint32_t groups = (mini_size + 7u) / 8u;
    cmd_buf->dispatch(groups, groups, 6u);

    // ── Transition mini-buffer → SHADER_READ_ONLY ─────────────────────────
    cmd_buf->addImageBarrier(
        mini_envmap_tex_.image,
        er::Helper::getImageAsStore(),
        er::Helper::getImageAsShaderSampler(),
        0, 1, 0, 6);

    // ── Transition envmap mip 0 → SHADER_READ_ONLY, regenerate mip chain ──
    // generateMipmapLevels expects mip 0 in SHADER_READ_ONLY_OPTIMAL as its
    // blit source; it transitions each destination mip from UNDEFINED and
    // blits down, leaving the full chain in SHADER_READ_ONLY_OPTIMAL when
    // done. The IBL mini compute shaders sample the envmap via textureLod
    // and need the full chain to be readable from the next frame onward.
    cmd_buf->addImageBarrier(
        rt_envmap_tex.image,
        er::Helper::getImageAsStore(),
        er::Helper::getImageAsShaderSampler(),
        0, 1, 0, 6);

    if (num_mips > 1) {
        renderer::Helper::generateMipmapLevels(
            cmd_buf,
            rt_envmap_tex.image,
            num_mips,
            cube_size,
            cube_size,
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    // Advance the dither frame counter so the next call uses a fresh
    // (dx, dy) offset.  Stays in sync with IblCreator::mini_frame_index_
    // which is advanced at the end of updateIblSheenMapMini.
    ++mini_frame_index_;
}


// ─── draw ────────────────────────────────────────────────────────────────────
// Renders the skybox into the current render pass using the full-resolution
// atmospheric scattering shader.  Called once per frame inside the main
// forward render pass.
void Skydome::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::DescriptorSet>& frame_desc_set) {

    glsl::SunSkyParams params = {};
    params.sun_pos                  = sun_dir_;
    params.g                        = g_;
    params.inv_rayleigh_scale_height = 1.0f / rayleigh_scale_height_;
    params.inv_mie_scale_height     = 1.0f / mie_scale_height_;

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, skybox_pipeline_);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        skybox_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        skybox_pipeline_layout_,
        { skybox_tex_desc_set_, frame_desc_set });

    std::vector<std::shared_ptr<renderer::Buffer>> buffers = { vertex_buffer_.buffer };
    std::vector<uint64_t> offsets = { 0 };
    cmd_buf->bindVertexBuffers(0, buffers, offsets);

    cmd_buf->bindIndexBuffer(
        index_buffer_.buffer,
        0,
        renderer::IndexType::UINT16);

    // 6 cube faces × 6 indices each = 36 indices.
    cmd_buf->drawIndexed(36);
}

// ─── initEnvmapBackgroundPipeline ────────────────────────────────────────────
// Creates the descriptor set layout, pipeline layout, pipeline, and descriptor
// set for the fullscreen sky envmap background pass.
//
// Uses the dynamic-rendering pipeline variant (no VkRenderPass) so it works
// with the engine's main forward pass which uses vkCmdBeginRenderingKHR.
// Depth state: LESS_OR_EQUAL / no write — the sky quad is at z=1.0 (far
// plane) so it only draws where the depth buffer still holds 1.0 (background).
void Skydome::initEnvmapBackgroundPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const renderer::TextureInfo& rt_envmap_tex,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::PipelineRenderbufferFormats& renderbuffer_formats) {

    // ── Tear down any previous instance (swap-chain rebuild) ─────────────────
    if (skybox_envmap_pipeline_layout_ != nullptr) {
        device->destroyPipelineLayout(skybox_envmap_pipeline_layout_);
        skybox_envmap_pipeline_layout_ = nullptr;
    }
    if (skybox_envmap_pipeline_ != nullptr) {
        device->destroyPipeline(skybox_envmap_pipeline_);
        skybox_envmap_pipeline_ = nullptr;
    }
    // Descriptor set and layout are pool-allocated; the pool is recreated on
    // swap-chain rebuild, so simply release the shared_ptrs.
    skybox_envmap_desc_set_ = nullptr;
    skybox_envmap_desc_set_layout_ = nullptr;

    // ── Descriptor set layout: envmap cubemap sampler at binding 0 ───────────
    skybox_envmap_desc_set_layout_ =
        device->createDescriptorSetLayout(
            { er::helper::getTextureSamplerDescriptionSetLayoutBinding(
                ENVMAP_TEX_INDEX,
                SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
                er::DescriptorType::COMBINED_IMAGE_SAMPLER) });

    // ── Pipeline layout: push constant carries SkyboxEnvmapParams (mat4) ─────
    er::PushConstantRange push_const_range{};
    push_const_range.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset      = 0;
    push_const_range.size        = sizeof(glsl::SkyboxEnvmapParams);

    skybox_envmap_pipeline_layout_ = device->createPipelineLayout(
        { skybox_envmap_desc_set_layout_ },
        { push_const_range },
        std::source_location::current());

    // ── Pipeline state ────────────────────────────────────────────────────────
    // Depth test ON (LESS_OR_EQUAL), depth write OFF — sky at z=1.0 only
    // draws on pixels where no geometry has been rasterized.
    auto depth_stencil = std::make_shared<er::PipelineDepthStencilStateCreateInfo>(
        er::helper::fillPipelineDepthStencilStateCreateInfo(
            true,   // depth_test_enable
            false,  // depth_write_enable
            er::CompareOp::LESS_OR_EQUAL));

    // No blending, no back-face culling (fullscreen triangle has no culling issue).
    auto blend_state = std::make_shared<er::PipelineColorBlendStateCreateInfo>(
        er::helper::fillPipelineColorBlendStateCreateInfo(
            { er::helper::fillPipelineColorBlendAttachmentState() }));
    auto raster_info = std::make_shared<er::PipelineRasterizationStateCreateInfo>(
        er::helper::fillPipelineRasterizationStateCreateInfo(
            false, false,
            er::PolygonMode::FILL,
            SET_FLAG_BIT(CullMode, NONE)));
    auto ms_info = std::make_shared<er::PipelineMultisampleStateCreateInfo>(
        er::helper::fillPipelineMultisampleStateCreateInfo());

    er::GraphicPipelineInfo pipeline_info;
    pipeline_info.blend_state_info   = blend_state;
    pipeline_info.rasterization_info = raster_info;
    pipeline_info.ms_info            = ms_info;
    pipeline_info.depth_stencil_info = depth_stencil;

    er::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology       = er::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    skybox_envmap_pipeline_ = device->createPipeline(
        skybox_envmap_pipeline_layout_,
        {}, {},        // no vertex bindings — vertices generated in the shader
        input_assembly,
        pipeline_info,
        getSkyboxEnvmapShaderModules(device),
        renderbuffer_formats,
        er::RasterizationStateOverride{},
        std::source_location::current());

    // ── Descriptor set: bind rt_envmap_tex as the source cubemap sampler ─────
    skybox_envmap_desc_set_ =
        device->createDescriptorSets(
            descriptor_pool,
            skybox_envmap_desc_set_layout_, 1)[0];

    er::WriteDescriptorList desc_writes;
    er::Helper::addOneTexture(
        desc_writes,
        skybox_envmap_desc_set_,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ENVMAP_TEX_INDEX,
        texture_sampler,
        rt_envmap_tex.view,
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    device->updateDescriptorSets(desc_writes);
}

// ─── drawEnvmap ──────────────────────────────────────────────────────────────
// Draws the sky envmap as a fullscreen background triangle into the currently
// active dynamic rendering pass.  The pipeline depth state (LESS_OR_EQUAL,
// no write) ensures sky only fills pixels where no geometry was rasterized
// (depth buffer == 1.0 = far-plane cleared value).
//
// Must be called after initEnvmapBackgroundPipeline() and inside a dynamic
// rendering pass that exposes the forward color + depth attachments.
void Skydome::drawEnvmap(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const glm::mat4& inv_view_proj_relative) {

    glsl::SkyboxEnvmapParams params = {};
    params.inv_view_proj_relative = inv_view_proj_relative;

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::GRAPHICS,
        skybox_envmap_pipeline_);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        skybox_envmap_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        skybox_envmap_pipeline_layout_,
        { skybox_envmap_desc_set_ });

    // 3 vertices: fullscreen triangle generated entirely in the vertex shader.
    cmd_buf->draw(3);
}

// ─── drawCubeSkyBox ──────────────────────────────────────────────────────────
// One-shot high-quality bake of the full-resolution sky cubemap using a
// fragment shader that outputs all 6 faces in a single draw (layered
// framebuffer attachment).  Generates the full mip chain afterwards.
//
// Used as the initial bootstrap before the dithered mini-buffer path takes
// over.  In the current app flow this is no longer called (the mini-buffer
// path replaces it with is_first_touch block-fill on frame 0), but the method
// is retained for debug/tooling use and as a fallback.
void Skydome::drawCubeSkyBox(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const renderer::TextureInfo& rt_envmap_tex,
    const std::vector<renderer::ClearValue>& clear_values,
    const uint32_t& cube_size) {

    // Transition mip 0 of the envmap from whatever layout it is in to
    // COLOR_ATTACHMENT_OPTIMAL so the render pass can write to it.
    cmd_buf->addImageBarrier(
        rt_envmap_tex.image,
        renderer::Helper::getImageAsSource(),
        renderer::Helper::getImageAsColorAttachment(),
        0, 1, 0, 6);

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::GRAPHICS,
        cube_skybox_pipeline_);

    glsl::SunSkyParams params = {};
    params.sun_pos                   = sun_dir_;
    params.g                         = g_;
    params.inv_rayleigh_scale_height = 1.0f / rayleigh_scale_height_;
    params.inv_mie_scale_height      = 1.0f / mie_scale_height_;

    // The framebuffers[0] entry is the full-resolution layered framebuffer
    // covering all 6 cube faces at mip 0, created by createCubemapTexture().
    std::vector<renderer::ClearValue> cube_clear_values(6, clear_values[0]);
    cmd_buf->beginRenderPass(
        render_pass,
        rt_envmap_tex.framebuffers[0],
        glm::uvec2(cube_size, cube_size),
        cube_clear_values);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        cube_skybox_pipeline_layout_,
        &params,
        sizeof(params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        cube_skybox_pipeline_layout_,
        { skybox_tex_desc_set_ });

    // Full-screen triangle — the vertex shader produces clip-space positions
    // from gl_VertexIndex with no vertex buffer.  The fragment shader loops
    // over all 6 faces and writes to layered output attachments.
    cmd_buf->draw(3);

    cmd_buf->endRenderPass();

    const uint32_t num_mips = static_cast<uint32_t>(std::log2(cube_size) + 1);
    if (num_mips > 1) {
        renderer::Helper::generateMipmapLevels(
            cmd_buf,
            rt_envmap_tex.image,
            num_mips,
            cube_size,
            cube_size,
            renderer::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }
}

// ─── update ──────────────────────────────────────────────────────────────────
// Computes the sun direction from geographic coordinates and time of day using
// the Spencer (1971) / Michalsky (1988) solar position approximation.
//
// Coordinate system: world-space Y-up (matches the engine convention used
// throughout the atmospheric scattering shaders).  The returned vector points
// FROM the earth TOWARD the sun and is not normalized — the shader normalizes
// it via normalize(params.sun_pos).
//
// Parameters
//   latitude   — degrees north (positive) / south (negative)
//   longtitude — degrees east (positive) / west (negative)  [sic: typo in API]
//   d          — day of year (0 = Jan 1)
//   h/m/s      — local solar time (hour, minute, second)
void Skydome::update(
    float latitude,
    float longtitude,
    int d,
    int h,
    int m,
    int s) {

    constexpr float kDegToRad = glm::pi<float>() / 180.0f;

    // Fractional hour in local solar time.
    const float t_hours = static_cast<float>(h)
                        + static_cast<float>(m) / 60.0f
                        + static_cast<float>(s) / 3600.0f;

    // Solar declination (degrees) — Spencer 1971.
    // B = 360/365 × (d - 81), with d = day of year starting at 1.
    const float B = (360.0f / 365.0f) * (static_cast<float>(d + 1) - 81.0f) * kDegToRad;
    const float decl_deg = 23.45f * std::sin(B);
    const float decl     = decl_deg * kDegToRad;

    // Hour angle: 0 at solar noon, negative in the morning.
    // Each hour = 15°; we add the longitude correction assuming the time
    // provided is already local solar time (no UTC offset needed here —
    // application.cpp passes the menu's time-of-day slider directly).
    const float hour_angle = (t_hours - 12.0f) * 15.0f * kDegToRad;

    const float lat = latitude * kDegToRad;

    // Solar altitude above horizon.
    const float sin_alt = std::sin(lat)  * std::sin(decl)
                        + std::cos(lat)  * std::cos(decl) * std::cos(hour_angle);
    const float altitude = std::asin(glm::clamp(sin_alt, -1.0f, 1.0f));

    // Solar azimuth (measured clockwise from north in the horizontal plane).
    const float cos_alt = std::cos(altitude);
    float azimuth = 0.0f;
    if (cos_alt > 1e-6f) {
        const float sin_az = (std::cos(decl) * std::sin(hour_angle)) / cos_alt;
        const float cos_az = (std::sin(lat) * sin_alt - std::sin(decl))
                           / (std::cos(lat) * cos_alt);
        azimuth = std::atan2(sin_az, cos_az);
    }

    // Convert to world-space Cartesian (Y-up, Z-south, X-east convention).
    //   x =  sin(azimuth) * cos(altitude)   — east component
    //   y =  sin(altitude)                   — up component
    //   z = -cos(azimuth) * cos(altitude)   — north component (Z points south)
    sun_dir_ = glm::vec3(
         std::sin(azimuth) * cos_alt,
         sin_alt,
        -std::cos(azimuth) * cos_alt);
}

// ─── dumpScatteringLut ───────────────────────────────────────────────────────
// Debug helper: copies the LUT to a staging buffer and prints sample values.
// Must be called after device->waitIdle() so the LUT compute is complete and
// the image is in SHADER_READ_ONLY_OPTIMAL layout.
void Skydome::dumpScatteringLut(const std::shared_ptr<renderer::Device>& device) {
    constexpr uint32_t kW = kAtmosphereScatteringLutWidth;
    constexpr uint32_t kH = kAtmosphereScatteringLutHeight;
    constexpr uint64_t kPixelBytes = 8;                    // rg32f = 2 × float32
    constexpr uint64_t kBufBytes   = kW * kH * kPixelBytes;

    // ── Create a host-visible staging buffer ─────────────────────────────────
    std::shared_ptr<er::Buffer>       staging_buf;
    std::shared_ptr<er::DeviceMemory> staging_mem;
    er::Helper::createBuffer(
        device,
        SET_FLAG_BIT(BufferUsage, TRANSFER_DST_BIT),
        SET_2_FLAG_BITS(MemoryProperty, HOST_VISIBLE_BIT, HOST_COHERENT_BIT),
        0,
        staging_buf,
        staging_mem,
        std::source_location::current(),
        kBufBytes);

    // ── Transient cmd buffer: barrier → copy → barrier ───────────────────────
    auto cmd = device->setupTransientCommandBuffer();

    // SHADER_READ_ONLY_OPTIMAL → TRANSFER_SRC_OPTIMAL
    er::ImageResourceInfo from_sampler = {
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        SET_2_FLAG_BITS(Access, SHADER_READ_BIT, SHADER_WRITE_BIT),
        SET_2_FLAG_BITS(PipelineStage, FRAGMENT_SHADER_BIT, COMPUTE_SHADER_BIT)
    };
    er::ImageResourceInfo to_transfer_src = {
        er::ImageLayout::TRANSFER_SRC_OPTIMAL,
        SET_FLAG_BIT(Access, TRANSFER_READ_BIT),
        SET_FLAG_BIT(PipelineStage, TRANSFER_BIT)
    };
    cmd->addImageBarrier(sky_scattering_lut_tex_.image,
                         from_sampler, to_transfer_src, 0, 1, 0, 1);

    // Copy entire LUT mip0 to staging buffer
    er::BufferImageCopyInfo region;
    region.buffer_offset      = 0;
    region.buffer_row_length  = 0;
    region.buffer_image_height = 0;
    region.image_subresource  = { SET_FLAG_BIT(ImageAspect, COLOR_BIT), 0, 0, 1 };
    region.image_offset       = { 0, 0, 0 };
    region.image_extent       = { kW, kH, 1 };
    cmd->copyImageToBuffer(sky_scattering_lut_tex_.image,
                           staging_buf, { region },
                           er::ImageLayout::TRANSFER_SRC_OPTIMAL);

    // TRANSFER_SRC_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
    er::ImageResourceInfo back_to_sampler = {
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        SET_FLAG_BIT(Access, SHADER_READ_BIT),
        SET_2_FLAG_BITS(PipelineStage, FRAGMENT_SHADER_BIT, COMPUTE_SHADER_BIT)
    };
    cmd->addImageBarrier(sky_scattering_lut_tex_.image,
                         to_transfer_src, back_to_sampler, 0, 1, 0, 1);

    device->submitAndWaitTransientCommandBuffer();

    // ── Read back and print a coarse sample grid ──────────────────────────────
    std::vector<float> pixels(kW * kH * 2);
    device->dumpBufferMemory(staging_mem, kBufBytes, pixels.data());

    std::cout << "\n=== sky_scattering_lut dump (log2-optical-depth, rg32f) ===\n";
    std::cout << "Format: [x][y] = (log2_rlh, log2_mie)  exp2→ (rlh_od, mie_od)\n";
    // Print a 8×8 coarse grid and a dense strip at x=256 (mid-perpendicular)
    static const uint32_t kXSamples[] = { 0, 64, 128, 192, 256, 320, 384, 448, 511 };
    static const uint32_t kYSamples[] = { 0, 64, 128, 192, 256, 320, 384, 448, 511 };
    for (uint32_t xi = 0; xi < 9; xi++) {
        uint32_t x = kXSamples[xi];
        for (uint32_t yi = 0; yi < 9; yi++) {
            uint32_t y = kYSamples[yi];
            uint32_t base = (y * kW + x) * 2;
            float lr = pixels[base + 0];   // log2(rlh_od)
            float lm = pixels[base + 1];   // log2(mie_od)
            float rlh = std::exp2(lr);
            float mie = std::exp2(lm);
            std::cout << "  [" << x << "][" << y << "]"
                      << " log2=(" << lr << ", " << lm << ")"
                      << " od=(" << rlh << ", " << mie << ")\n";
        }
    }

    // Also print the raw first/last value to catch zeros/NaN/Inf early.
    {
        float r0 = pixels[0], m0 = pixels[1];
        float rE = pixels[(kW*kH-1)*2], mE = pixels[(kW*kH-1)*2+1];
        std::cout << "  [0][0]   raw log2=(" << r0 << ", " << m0 << ")\n";
        std::cout << "  [511][511] raw log2=(" << rE << ", " << mE << ")\n";
    }
    std::cout << "=== end of LUT dump ===\n" << std::flush;

    // ── Free staging buffer ───────────────────────────────────────────────────
    device->destroyBuffer(staging_buf);
    device->freeMemory(staging_mem);
}

// ─── updateSkyScatteringLut ───────────────────────────────────────────────────
// Runs the 3-pass atmospheric scattering LUT bake when the Rayleigh or Mie
// scale heights have changed.  The LUT is read by both the skybox fragment
// shader and the mini-buffer compute shader; it must be current before either
// of those runs.
//
// Pass layout:
//   first_pass  — writes per-column partial optical depth to sky_scattering_lut_tex_
//                 Dispatch(512, 8, 1); push SkyScatteringParams (COMPUTE_BIT).
//   sum_pass    — accumulates column sums into sky_scattering_lut_sum_tex_
//                 Dispatch(512, 1, 1); no push constants.
//   final_pass  — normalises back into sky_scattering_lut_tex_ using sum
//                 Dispatch(512, 8, 1); no push constants.
void Skydome::updateSkyScatteringLut(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf) {

    // Early-out: LUT is already up to date for the current scale heights.
    if (lut_rayleigh_scale_height_ == rayleigh_scale_height_ &&
        lut_mie_scale_height_      == mie_scale_height_) {
        return;
    }
    lut_rayleigh_scale_height_ = rayleigh_scale_height_;
    lut_mie_scale_height_      = mie_scale_height_;

    constexpr uint32_t kLutWidth   = kAtmosphereScatteringLutWidth;
    constexpr uint32_t kLutHeight  = kAtmosphereScatteringLutHeight;
    constexpr uint32_t kGroupSize  = kAtmosphereScatteringLutGroupSize;

    // ── Transition LUT textures to GENERAL for storage-image writes ──────────
    cmd_buf->addImageBarrier(
        sky_scattering_lut_tex_.image,
        renderer::Helper::getImageAsShaderSampler(),
        renderer::Helper::getImageAsStore(),
        0, 1, 0, 1);

    cmd_buf->addImageBarrier(
        sky_scattering_lut_sum_tex_.image,
        renderer::Helper::getImageAsShaderSampler(),
        renderer::Helper::getImageAsStore(),
        0, 1, 0, 1);

    // ── First pass: per-column partial optical depth ─────────────────────────
    glsl::SkyScatteringParams lut_params = {};
    lut_params.inv_rayleigh_scale_height = 1.0f / rayleigh_scale_height_;
    lut_params.inv_mie_scale_height      = 1.0f / mie_scale_height_;

    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE,
        sky_scattering_lut_first_pass_pipeline_);

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        sky_scattering_lut_first_pass_pipeline_layout_,
        &lut_params,
        sizeof(lut_params));

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        sky_scattering_lut_first_pass_pipeline_layout_,
        { sky_scattering_lut_first_pass_desc_set_ });

    cmd_buf->dispatch(kLutWidth, kLutHeight / kGroupSize, 1);

    // ── Barrier: first_pass writes → sum_pass reads (LUT tex) ───────────────
    // dst must be getImageAsLoadStore() (SHADER_READ|WRITE in GENERAL layout).
    // A Store→Store barrier (dst=getImageAsStore(), dstAccess=SHADER_WRITE only)
    // does NOT make the first_pass writes visible to sum_pass's imageLoad calls.
    cmd_buf->addImageBarrier(
        sky_scattering_lut_tex_.image,
        renderer::Helper::getImageAsStore(),
        renderer::Helper::getImageAsLoadStore(),
        0, 1, 0, 1);

    // ── Sum pass: accumulate column totals ───────────────────────────────────
    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE,
        sky_scattering_lut_sum_pass_pipeline_);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        sky_scattering_lut_sum_pass_pipeline_layout_,
        { sky_scattering_lut_sum_pass_desc_set_ });

    cmd_buf->dispatch(kLutWidth, 1, 1);

    // ── Barrier: sum_pass writes → final_pass reads (sum tex) ───────────────
    // Same reasoning: final_pass reads src_sum_lut via imageLoad, so dstAccess
    // must include SHADER_READ_BIT, not just SHADER_WRITE_BIT.
    cmd_buf->addImageBarrier(
        sky_scattering_lut_sum_tex_.image,
        renderer::Helper::getImageAsStore(),
        renderer::Helper::getImageAsLoadStore(),
        0, 1, 0, 1);

    // ── Final pass: normalise using accumulated sums ─────────────────────────
    cmd_buf->bindPipeline(
        renderer::PipelineBindPoint::COMPUTE,
        sky_scattering_lut_final_pass_pipeline_);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::COMPUTE,
        sky_scattering_lut_final_pass_pipeline_layout_,
        { sky_scattering_lut_final_pass_desc_set_ });

    cmd_buf->dispatch(kLutWidth, kLutHeight / kGroupSize, 1);

    // ── Transition both LUT textures back to SHADER_READ_ONLY ────────────────
    cmd_buf->addImageBarrier(
        sky_scattering_lut_tex_.image,
        renderer::Helper::getImageAsStore(),
        renderer::Helper::getImageAsShaderSampler(),
        0, 1, 0, 1);

    cmd_buf->addImageBarrier(
        sky_scattering_lut_sum_tex_.image,
        renderer::Helper::getImageAsStore(),
        renderer::Helper::getImageAsShaderSampler(),
        0, 1, 0, 1);
}

// ─── destroy ─────────────────────────────────────────────────────────────────
void Skydome::destroy(const std::shared_ptr<renderer::Device>& device) {
    // Geometry buffers.
    if (vertex_buffer_.buffer) vertex_buffer_.destroy(device);
    if (index_buffer_.buffer)  index_buffer_.destroy(device);

    // Textures.
    if (sky_scattering_lut_tex_.image)     sky_scattering_lut_tex_.destroy(device);
    if (sky_scattering_lut_sum_tex_.image) sky_scattering_lut_sum_tex_.destroy(device);
    if (mini_envmap_tex_.image)            mini_envmap_tex_.destroy(device);

    // Skybox graphics pipeline.
    if (skybox_pipeline_)        device->destroyPipeline(skybox_pipeline_);
    if (skybox_pipeline_layout_) device->destroyPipelineLayout(skybox_pipeline_layout_);
    if (skybox_desc_set_layout_) device->destroyDescriptorSetLayout(skybox_desc_set_layout_);

    // Cube skybox graphics pipeline (used by drawCubeSkyBox).
    if (cube_skybox_pipeline_)        device->destroyPipeline(cube_skybox_pipeline_);
    if (cube_skybox_pipeline_layout_) device->destroyPipelineLayout(cube_skybox_pipeline_layout_);

    // Mini-buffer compute pipeline.
    if (cube_skybox_mini_pipeline_)        device->destroyPipeline(cube_skybox_mini_pipeline_);
    if (cube_skybox_mini_pipeline_layout_) device->destroyPipelineLayout(cube_skybox_mini_pipeline_layout_);
    if (cube_skybox_mini_desc_set_layout_) device->destroyDescriptorSetLayout(cube_skybox_mini_desc_set_layout_);

    // Sky scattering LUT compute pipelines.
    if (sky_scattering_lut_first_pass_pipeline_)
        device->destroyPipeline(sky_scattering_lut_first_pass_pipeline_);
    if (sky_scattering_lut_sum_pass_pipeline_)
        device->destroyPipeline(sky_scattering_lut_sum_pass_pipeline_);
    if (sky_scattering_lut_final_pass_pipeline_)
        device->destroyPipeline(sky_scattering_lut_final_pass_pipeline_);

    if (sky_scattering_lut_first_pass_pipeline_layout_)
        device->destroyPipelineLayout(sky_scattering_lut_first_pass_pipeline_layout_);
    if (sky_scattering_lut_sum_pass_pipeline_layout_)
        device->destroyPipelineLayout(sky_scattering_lut_sum_pass_pipeline_layout_);
    if (sky_scattering_lut_final_pass_pipeline_layout_)
        device->destroyPipelineLayout(sky_scattering_lut_final_pass_pipeline_layout_);

    if (sky_scattering_lut_first_pass_desc_set_layout_)
        device->destroyDescriptorSetLayout(sky_scattering_lut_first_pass_desc_set_layout_);
    if (sky_scattering_lut_sum_pass_desc_set_layout_)
        device->destroyDescriptorSetLayout(sky_scattering_lut_sum_pass_desc_set_layout_);
    if (sky_scattering_lut_final_pass_desc_set_layout_)
        device->destroyDescriptorSetLayout(sky_scattering_lut_final_pass_desc_set_layout_);
}

}// namespace scene_rendering
}// namespace engine
