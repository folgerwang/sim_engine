#include "conemap_test.h"
#include "engine_helper.h"
#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "shaders/global_definition.glsl.h"

namespace engine {
namespace {

static float getKValueFast(int l, int m)
{
    const float s_factorial_serials[9] =
    { 1.0f, 1.0f, 2.0f, 6.0f, 24.0f, 120.0f, 720.0f, 5040.0f, 40320.0f };
    float factor = sqrt((2 * l + 1) / (4.0f * glm::pi<float>()));
    return sqrt(factor * s_factorial_serials[l - abs(m)] / s_factorial_serials[l + abs(m)]);
}

static void fillPVauleTablle(float p_value[15], float x) {
    float x2 = x * x;
    float x2_s7 = 7.0f * x2;
    float x2_1 = 1.0f - x2;
    float x2_1sqrt = sqrt(x2_1);
    float x2_1_2_3rd = x2_1sqrt * x2_1;
    float inv_x2_1_2_3rd = 1.0f / x2_1_2_3rd;
    float x2_1sqr = x2_1 * x2_1;

    // l = 0, m = 0
    p_value[0] = 1.0f;
    // l = 1, m = 0
    p_value[1] = x;
    // l = 1, m = 1
    p_value[2] = -x2_1sqrt;
    float a00 = x * x2_1sqrt;
    // l = 2, m = 0
    p_value[3] = 0.5f * (3.0f * x2 - 1.0f);
    // l = 2, m = 1
    p_value[4] = -3.0f * a00;
    // l = 2, m = 2
    p_value[5] = 3.0f * x2_1;
    float a01 = (-5.0f * x2 + 1.0f) * x2_1sqrt;
    // l = 3, m = 0
    p_value[6] = 0.5f * x * (5.0f * x2 - 3.0f);
    // l = 3, m = 1
    p_value[7] = 1.5f * a01;
    // l = 3, m = 2
    p_value[8] = 15.0f * x * x2_1;
    // l = 3, m = 3
    p_value[9] = -15.0f * x2_1_2_3rd;
    float a02 = (-x2_s7 + 3.0f) * a00;
    // l = 4, m = 0
    p_value[10] = 0.125f * ((35.0f * x2 - 30.0f) * x2 + 3.0f);
    // l = 4, m = 1
    p_value[11] = 2.5f * a02;
    // l = 4, m = 2
    p_value[12] = 7.5f * (x2_s7 - 1.0f) * x2_1;
    // l = 4, m = 3
    p_value[13] = -105.0f * x * x2_1_2_3rd;
    // l = 4, m = 4
    p_value[14] = 105.0f * x2_1sqr;
}

static void fillYVauleTablle(float y_value[25], float theta, float phi) {
    const float sqrt2 = sqrt(2.0f);

    float cos_theta = cos(theta);
    float sin_phi = sin(phi);
    float cos_phi = cos(phi);
    float sin_2phi = sin(2.0f * phi);
    float cos_2phi = cos(2.0f * phi);
    float sin_3phi = sin(3.0f * phi);
    float cos_3phi = cos(3.0f * phi);
    float sin_4phi = sin(4.0f * phi);
    float cos_4phi = cos(4.0f * phi);

    float p_value[15];
    fillPVauleTablle(p_value, cos_theta);

    float a11 = sqrt2 * getKValueFast(1, 1) * p_value[2];
    float a22 = sqrt2 * getKValueFast(2, 2) * p_value[5];
    float a21 = sqrt2 * getKValueFast(2, 1) * p_value[4];
    float a33 = sqrt2 * getKValueFast(3, 3) * p_value[9];
    float a32 = sqrt2 * getKValueFast(3, 2) * p_value[8];
    float a31 = sqrt2 * getKValueFast(3, 1) * p_value[7];
    float a44 = sqrt2 * getKValueFast(4, 4) * p_value[14];
    float a43 = sqrt2 * getKValueFast(4, 3) * p_value[13];
    float a42 = sqrt2 * getKValueFast(4, 2) * p_value[12];
    float a41 = sqrt2 * getKValueFast(4, 1) * p_value[11];

    // l = 0, m = 0
    y_value[0] = getKValueFast(0, 0) * p_value[0];
    // l = 1, m = -1
    y_value[1] = sin_phi * a11;
    // l = 1, m = 0
    y_value[2] = getKValueFast(1, 0) * p_value[1];
    // l = 1, m = 1
    y_value[3] = cos_phi * a11;
    // l = 2, m = -2
    y_value[4] = sin_2phi * a22;
    // l = 2, m = -1
    y_value[5] = sin_phi * a21;
    // l = 2, m = 0
    y_value[6] = getKValueFast(2, 0) * p_value[3];
    // l = 2, m = 1
    y_value[7] = cos_phi * a21;
    // l = 2, m = 2
    y_value[8] = cos_2phi * a22;
    // l = 3, m = -3
    y_value[9] = sin_3phi * a33;
    // l = 3, m = -2
    y_value[10] = sin_2phi * a32;
    // l = 3, m = -1
    y_value[11] = sin_phi * a31;
    // l = 3, m = 0
    y_value[12] = getKValueFast(3, 0) * p_value[6];
    // l = 3, m = 1
    y_value[13] = cos_phi * a31;
    // l = 3, m = 2
    y_value[14] = cos_2phi * a32;
    // l = 3, m = 3
    y_value[15] = cos_3phi * a33;
    // l = 4, m = -4
    y_value[16] = sin_4phi * a44;
    // l = 4, m = -3
    y_value[17] = sin_3phi * a43;
    // l = 4, m = -2
    y_value[18] = sin_2phi * a42;
    // l = 4, m = -1
    y_value[19] = sin_phi * a41;
    // l = 4, m = 0
    y_value[20] = getKValueFast(4, 0) * p_value[10];
    // l = 4, m = 1
    y_value[21] = cos_phi * a41;
    // l = 4, m = 2
    y_value[22] = cos_2phi * a42;
    // l = 4, m = 3
    y_value[23] = cos_3phi * a43;
    // l = 4, m = 4
    y_value[24] = cos_4phi * a44;
}

static std::shared_ptr<renderer::DescriptorSetLayout> createPrtDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(10);

    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(ALBEDO_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(NORMAL_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(METAL_ROUGHNESS_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(CONEMAP_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(EMISSIVE_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(OCCLUSION_TEX_INDEX));
    bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(THIN_FILM_LUT_INDEX));
/*    for (int i = 0; i < 7; ++i)
        bindings.push_back(renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(PRT_TEX_INDEX_0 + i));*/
    bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            PRT_PACK_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, VERTEX_BIT) | SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            renderer::DescriptorType::STORAGE_IMAGE));
    bindings.push_back(
        renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
            PRT_PACK_INFO_TEX_INDEX,
            SET_FLAG_BIT(ShaderStage, VERTEX_BIT) | SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
            renderer::DescriptorType::STORAGE_IMAGE));

    renderer::DescriptorSetLayoutBinding ubo_pbr_layout_binding{};
    ubo_pbr_layout_binding.binding = PBR_CONSTANT_INDEX;
    ubo_pbr_layout_binding.descriptor_count = 1;
    ubo_pbr_layout_binding.descriptor_type = renderer::DescriptorType::UNIFORM_BUFFER;
    ubo_pbr_layout_binding.stage_flags = SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    ubo_pbr_layout_binding.immutable_samplers = nullptr; // Optional
    bindings.push_back(ubo_pbr_layout_binding);

    return device->createDescriptorSetLayout(bindings);
}

static renderer::WriteDescriptorList addPrtTestTextures(
    const std::shared_ptr<renderer::DescriptorSet>& desc_set,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& base_tex,
    const renderer::TextureInfo& normal_tex,
    const renderer::TextureInfo& orh_tex,
    const std::shared_ptr<renderer::TextureInfo>& conemap_tex,
    const std::shared_ptr<renderer::TextureInfo>& prt_pack_texture,
    const std::shared_ptr<renderer::TextureInfo>& prt_pack_info_texture,
    const std::shared_ptr<renderer::BufferInfo>& uniform_buffer) {

    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(10);

    // diffuse.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ALBEDO_TEX_INDEX,
        texture_sampler,
        base_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // normal.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        NORMAL_TEX_INDEX,
        texture_sampler,
        normal_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // orh.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        METAL_ROUGHNESS_TEX_INDEX,
        texture_sampler,
        orh_tex.view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // conemap.
    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        CONEMAP_TEX_INDEX,
        texture_sampler,
        conemap_tex->view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    // prt.
    /*
    for (int i = 0; i < prt_texes.size(); ++i) {
        renderer::Helper::addOneTexture(
            descriptor_writes,
            desc_set,
            renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
            PRT_TEX_INDEX_0 + i,
            texture_sampler,
            prt_texes[i]->view,
            renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }*/
    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        PRT_PACK_TEX_INDEX,
        nullptr,
        prt_pack_texture->view,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        EMISSIVE_TEX_INDEX,
        texture_sampler,
        renderer::Helper::getBlackTexture().view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        OCCLUSION_TEX_INDEX,
        texture_sampler,
        renderer::Helper::getBlackTexture().view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        THIN_FILM_LUT_INDEX,
        texture_sampler,
        renderer::Helper::getBlackTexture().view,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::STORAGE_IMAGE,
        PRT_PACK_INFO_TEX_INDEX,
        nullptr,
        prt_pack_info_texture->view,
        renderer::ImageLayout::GENERAL);

    renderer::Helper::addOneBuffer(
        descriptor_writes,
        desc_set,
        renderer::DescriptorType::UNIFORM_BUFFER,
        PBR_CONSTANT_INDEX,
        uniform_buffer->buffer,
        uniform_buffer->buffer->getSize());

    return descriptor_writes;
}

static std::shared_ptr<renderer::PipelineLayout> createPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::DescriptorSetLayout>& prt_desc_set_layout) {

    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::PrtLightParams);

    renderer::DescriptorSetLayoutList desc_set_layouts = global_desc_set_layouts;
    desc_set_layouts.push_back(prt_desc_set_layout);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

} // namespace

namespace game_object {

ConemapTest::ConemapTest(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const renderer::TextureInfo& prt_base_tex,
    const renderer::TextureInfo& prt_normal_tex,
    const renderer::TextureInfo& prt_orh_tex,
    const std::shared_ptr<game_object::ConemapObj>& conemap_obj,
    const glm::uvec2& display_size,
    std::shared_ptr<Plane> unit_plane) {

    const glm::uvec2& buffer_size =
        glm::uvec2(prt_base_tex.size);

    prt_desc_set_layout_ = createPrtDescriptorSetLayout(
        device);

    prt_desc_set_ = device->createDescriptorSets(
        descriptor_pool, prt_desc_set_layout_, 1)[0];

    uniform_buffer_ = std::make_shared<renderer::BufferInfo>();
    device->createBuffer(
        sizeof(glsl::PbrMaterialParams),
        SET_FLAG_BIT(BufferUsage, STORAGE_BUFFER_BIT) |
        SET_FLAG_BIT(BufferUsage, UNIFORM_BUFFER_BIT),
        SET_FLAG_BIT(MemoryProperty, HOST_VISIBLE_BIT) |
        SET_FLAG_BIT(MemoryProperty, HOST_CACHED_BIT),
        0,
        uniform_buffer_->buffer,
        uniform_buffer_->memory);

    // create a global ibl texture descriptor set.
    auto prt_test_material_descs =
        addPrtTestTextures(
            prt_desc_set_,
            texture_sampler,
            prt_base_tex,
            prt_normal_tex,
            prt_orh_tex,
            conemap_obj->getConemapTexture(),
            conemap_obj->getPackTexture(),
            conemap_obj->getPackInfoTexture(),
            uniform_buffer_);

    device->updateDescriptorSets(prt_test_material_descs);

    prt_pipeline_layout_ = createPipelineLayout(
        device,
        global_desc_set_layouts,
        prt_desc_set_layout_);

    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device,
            "conemap_test_vert.spv",
            renderer::ShaderStageFlagBits::VERTEX_BIT);
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device,
            "conemap_test_frag.spv",
            renderer::ShaderStageFlagBits::FRAGMENT_BIT);

    prt_pipeline_ = device->createPipeline(
        render_pass,
        prt_pipeline_layout_,
        unit_plane->getBindingDescs(),
        unit_plane->getAttribDescs(),
        input_assembly,
        graphic_pipeline_info,
        shader_modules,
        display_size);
}

void ConemapTest::draw(
    const std::shared_ptr<renderer::Device>& device,
    std::shared_ptr<renderer::CommandBuffer> cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    std::shared_ptr<Plane> unit_plane,
    const std::shared_ptr<game_object::ConemapObj>& conemap_obj) {

    const auto buffer_size =
        glm::uvec2(conemap_obj->getPackTexture()->size);

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, prt_pipeline_);

    renderer::DescriptorSetList desc_sets = desc_set_list;
    desc_sets.push_back(prt_desc_set_);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        prt_pipeline_layout_,
        desc_sets);

    glsl::PrtLightParams params{};
    params.model_mat =
        glm::mat4(
            glm::vec4(float(buffer_size.x) / float(buffer_size.y) * 1.0f, 0, 0, 0),
            glm::vec4(0, 1, 0, 0),
            glm::vec4(0, 0, 1, 0),
            glm::vec4(0, 0, 0, 1));

    float y_value[25];
    static float s_theta = glm::pi<float>() / 3.0f;
    static float s_phi = glm::pi<float>() / 4.0f;
    glm::vec2 ray_2d = glm::normalize(glm::vec2(std::sinf(s_theta), std::cos(s_theta) / conemap_obj->getDepthScale()));
    
    fillYVauleTablle(y_value, std::acosf(ray_2d.y), s_phi);

    glm::vec3 light_ray =
        glm::vec3(std::sin(s_theta) * std::cos(s_phi),
            std::cos(s_theta),
            std::sin(s_theta) * std::sin(s_phi));

    for (int i = 0; i < 25; i++) {
        params.coeffs[i] = y_value[i];
    }

    //s_theta += 0.001f;
    s_phi += 0.003f;

    params.height_scale = conemap_obj->getDepthScale() * (conemap_obj->isHeightMap() ? -1.0f : 1.0f);
    params.buffer_size = glm::vec2(buffer_size);
    params.test_color = light_ray * 0.5f + 0.5f;

    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        prt_pipeline_layout_,
        &params,
        sizeof(params));

    {
        glsl::PbrMaterialParams ubo{};
        ubo.base_color_factor = glm::vec4(1.0f);
        ubo.glossiness_factor = 1.0f;
        ubo.metallic_roughness_specular_factor = 0.3f;
        ubo.metallic_factor = static_cast<float>(1.0f);
        ubo.roughness_factor = static_cast<float>(1.0f);
        ubo.alpha_cutoff = static_cast<float>(1.0f);
        ubo.mip_count = 11;
        ubo.normal_scale = static_cast<float>(1.0f);
        ubo.occlusion_strength = static_cast<float>(1.0f);

        ubo.emissive_factor = glm::vec3(0.0f, 0.0f, 0.0f);
        ubo.uv_set_flags = glm::vec4(0, 0, 0, 0);
        ubo.exposure = 1.0f;
        ubo.material_features = FEATURE_HAS_METALLIC_ROUGHNESS_MAP | FEATURE_MATERIAL_METALLICROUGHNESS;
        ubo.material_features |= FEATURE_HAS_BASE_COLOR_MAP;
        ubo.material_features |= FEATURE_HAS_OCCLUSION_MAP;
        ubo.material_features |= FEATURE_HAS_NORMAL_MAP;
        ubo.tonemap_type = TONEMAP_DEFAULT;
        ubo.specular_factor = glm::vec3(1.0f, 1.0f, 1.0f);
        for (int l = 0; l < LIGHT_COUNT; l++) {
            ubo.lights[l].type = glsl::LightType_Directional;
            ubo.lights[l].color = glm::vec3(1, 0, 0);
            ubo.lights[l].direction = light_ray;
            ubo.lights[l].intensity = 100.0f;
            ubo.lights[l].position = glm::vec3(0, 0, 0);
        }

        device->updateBufferMemory(
            uniform_buffer_->memory,
            sizeof(ubo),
            &ubo);
    }


    if (unit_plane) {
        unit_plane->draw(cmd_buf);
    }
}

void ConemapTest::destroy(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(prt_desc_set_layout_);
    device->destroyPipelineLayout(prt_pipeline_layout_);
    device->destroyPipeline(prt_pipeline_);
    uniform_buffer_->destroy(device);
}

} // game_object
} // engine
