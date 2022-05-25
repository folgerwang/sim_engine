// Mountains. By David Hoskins - 2013
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

// https://www.shadertoy.com/view/4slGD4
// A ray-marched version of my terrain renderer which uses
// streaming texture normals for speed:-
// http://www.youtube.com/watch?v=qzkBnCBpQAM

// It uses binary subdivision to accurately find the height map.
// Lots of thanks to Inigo and his noise functions!

// Video of my OpenGL version that 
// http://www.youtube.com/watch?v=qzkBnCBpQAM


#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "renderer/renderer.h"
#include "renderer/renderer_helper.h"
#include "engine_helper.h"
#include "terrain.h"
#include "debug_draw.h"
#include "shaders/global_definition.glsl.h"

//#define STATIC_CAMERA
#define LOWQUALITY

namespace engine {
namespace game_object {
namespace {

renderer::WriteDescriptorList addDebugDrawBuffers(
    const std::shared_ptr<renderer::DescriptorSet>& description_set,
    const std::shared_ptr<renderer::Sampler>&texture_sampler,
    const std::shared_ptr<renderer::ImageView>& temp_volume_tex,
    const std::shared_ptr<renderer::ImageView>& moisture_volume_tex,
    const std::shared_ptr<renderer::ImageView>& airflow_tex) {
    renderer::WriteDescriptorList descriptor_writes;
    descriptor_writes.reserve(3);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_TEMP_TEX_INDEX,
        texture_sampler,
        temp_volume_tex,
        renderer::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    renderer::Helper::addOneTexture(
        descriptor_writes,
        description_set,
        renderer::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SRC_MOISTURE_TEX_INDEX,
        texture_sampler,
        moisture_volume_tex,
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

static renderer::ShaderModuleList getDebugDrawShaderModules(
    std::shared_ptr<renderer::Device> device) {
    renderer::ShaderModuleList shader_modules(2);
    shader_modules[0] =
        renderer::helper::loadShaderModule(
            device,
            "debug_draw_vert.spv",
            renderer::ShaderStageFlagBits::VERTEX_BIT);
    shader_modules[1] =
        renderer::helper::loadShaderModule(
            device,
            "debug_draw_frag.spv",
            renderer::ShaderStageFlagBits::FRAGMENT_BIT);

    return shader_modules;
}

static std::shared_ptr<renderer::DescriptorSetLayout> CreateDebugDrawDescriptorSetLayout(
    const std::shared_ptr<renderer::Device>& device) {
    std::vector<renderer::DescriptorSetLayoutBinding> bindings(3);

    bindings[0] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SRC_TEMP_TEX_INDEX,
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, VERTEX_BIT));

    bindings[1] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SRC_MOISTURE_TEX_INDEX,
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, VERTEX_BIT));

    bindings[2] = renderer::helper::getTextureSamplerDescriptionSetLayoutBinding(
        SRC_AIRFLOW_INDEX,
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT) | SET_FLAG_BIT(ShaderStage, VERTEX_BIT));
    
    return device->createDescriptorSetLayout(bindings);
}

static std::shared_ptr<renderer::PipelineLayout> createDebugDrawPipelineLayout(
    const std::shared_ptr<renderer::Device>& device,
    const renderer::DescriptorSetLayoutList& desc_set_layouts) {
    renderer::PushConstantRange push_const_range{};
    push_const_range.stage_flags =
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) |
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT);
    push_const_range.offset = 0;
    push_const_range.size = sizeof(glsl::DebugDrawParams);

    return device->createPipelineLayout(desc_set_layouts, { push_const_range });
}

static std::shared_ptr<renderer::Pipeline> createDebugDrawPipeline(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const std::shared_ptr<renderer::PipelineLayout>& pipeline_layout,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const glm::uvec2& display_size) {
    renderer::PipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.topology = renderer::PrimitiveTopology::TRIANGLE_LIST;
    input_assembly.restart_enable = false;

    renderer::GraphicPipelineInfo new_graphic_pipeline_info = graphic_pipeline_info;
    new_graphic_pipeline_info.rasterization_info = 
        std::make_shared<renderer::PipelineRasterizationStateCreateInfo>(
            renderer::helper::fillPipelineRasterizationStateCreateInfo(
                false,
                false,
                renderer::PolygonMode::FILL,
                SET_FLAG_BIT(CullMode, NONE)));

    auto shader_modules = getDebugDrawShaderModules(device);
    auto pipeline = device->createPipeline(
        render_pass,
        pipeline_layout,
        {},
        {},
        input_assembly,
        new_graphic_pipeline_info,
        shader_modules,
        display_size);

    return pipeline;
}

} // namespace

// static member definition.
std::shared_ptr<renderer::PipelineLayout> DebugDrawObject::debug_draw_pipeline_layout_;
std::shared_ptr<renderer::Pipeline> DebugDrawObject::debug_draw_pipeline_;
std::shared_ptr<renderer::DescriptorSetLayout> DebugDrawObject::debug_draw_desc_set_layout_;
std::shared_ptr<renderer::DescriptorSet> DebugDrawObject::debug_draw_desc_set_;

DebugDrawObject::DebugDrawObject(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::DescriptorPool> descriptor_pool,
    const glm::vec2& min,
    const glm::vec2& max) :
    device_info_(device_info),
    min_(min),
    max_(max){
    assert(debug_draw_desc_set_layout_);
}

void DebugDrawObject::createStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const glm::uvec2& display_size) {

    auto desc_set_layouts = global_desc_set_layouts;
    desc_set_layouts.push_back(debug_draw_desc_set_layout_);

    if (debug_draw_pipeline_layout_ == nullptr) {
        debug_draw_pipeline_layout_ =
            createDebugDrawPipelineLayout(
                device,
                desc_set_layouts);
    }

    if (debug_draw_pipeline_ == nullptr) {
        assert(debug_draw_pipeline_layout_);
        debug_draw_pipeline_ =
            createDebugDrawPipeline(
                device,
                render_pass,
                debug_draw_pipeline_layout_,
                graphic_pipeline_info,
                display_size);
    }
}

void DebugDrawObject::initStaticMembers(
    const renderer::DeviceInfo& device_info,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const glm::uvec2& display_size) {
    auto& device = device_info.device;

    if (debug_draw_desc_set_layout_ == nullptr) {
        debug_draw_desc_set_layout_ =
            CreateDebugDrawDescriptorSetLayout(device);
    }

    createStaticMembers(
        device,
        render_pass,
        graphic_pipeline_info,
        global_desc_set_layouts,
        display_size);
}

void DebugDrawObject::recreateStaticMembers(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::RenderPass>& render_pass,
    const renderer::GraphicPipelineInfo& graphic_pipeline_info,
    const renderer::DescriptorSetLayoutList& global_desc_set_layouts,
    const glm::uvec2& display_size) {

    if (debug_draw_pipeline_layout_) {
        device->destroyPipelineLayout(debug_draw_pipeline_layout_);
        debug_draw_pipeline_layout_ = nullptr;
    }

    if (debug_draw_pipeline_) {
        device->destroyPipeline(debug_draw_pipeline_);
        debug_draw_pipeline_ = nullptr;
    }

    createStaticMembers(
        device,
        render_pass,
        graphic_pipeline_info,
        global_desc_set_layouts,
        display_size);
}

void DebugDrawObject::destoryStaticMembers(
    const std::shared_ptr<renderer::Device>& device) {
    device->destroyDescriptorSetLayout(debug_draw_desc_set_layout_);
    device->destroyPipelineLayout(debug_draw_pipeline_layout_);
    device->destroyPipeline(debug_draw_pipeline_);
}

void DebugDrawObject::generateStaticDescriptorSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& temp_volume_tex,
    const std::shared_ptr<renderer::ImageView>& moisture_volume_tex,
    const std::shared_ptr<renderer::ImageView>& airflow_tex) {
    // tile creator buffer set.
    debug_draw_desc_set_ = device->createDescriptorSets(
        descriptor_pool, debug_draw_desc_set_layout_, 1)[0];

    // create a global ibl texture descriptor set.
    // all world map buffer only create once, so always pick the first one.
    auto texture_descs = addDebugDrawBuffers(
        debug_draw_desc_set_,
        texture_sampler,
        temp_volume_tex,
        moisture_volume_tex,
        airflow_tex);
    device->updateDescriptorSets(texture_descs);
}

void DebugDrawObject::updateStaticDescriptorSet(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& temp_volume_tex,
    const std::shared_ptr<renderer::ImageView>& moisture_volume_tex,
    const std::shared_ptr<renderer::ImageView>& airflow_tex) {

    if (debug_draw_desc_set_ == nullptr) {
        generateStaticDescriptorSet(
            device,
            descriptor_pool,
            texture_sampler,
            temp_volume_tex,
            moisture_volume_tex,
            airflow_tex);
    }
}

void DebugDrawObject::draw(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const renderer::DescriptorSetList& desc_set_list,
    const glm::vec3& camera_pos,
    uint32_t debug_type) {

    cmd_buf->bindPipeline(renderer::PipelineBindPoint::GRAPHICS, debug_draw_pipeline_);

    glm::vec3 aligned_camera_pos = glm::vec3(glm::ivec3(camera_pos / 128.0f)) * 128.0f;

    glsl::DebugDrawParams debug_params = {};
    debug_params.world_min = glm::vec2(-kWorldMapSize / 2.0f);
    debug_params.inv_world_range = 1.0f / (glm::vec2(kWorldMapSize / 2.0f) - debug_params.world_min);
    debug_params.debug_min = aligned_camera_pos - glm::vec3(4096.0f, 2048.0f, 4096.0f);
    debug_params.debug_range = glm::vec3(8192.0f, 8192.0f, 8192.0f);
    debug_params.size = glm::uvec3(32, 32, 32);
    debug_params.inv_size = 1.0f / glm::vec3(debug_params.size);
    debug_params.debug_type = debug_type;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, VERTEX_BIT) | 
        SET_FLAG_BIT(ShaderStage, FRAGMENT_BIT),
        debug_draw_pipeline_layout_,
        &debug_params,
        sizeof(debug_params));

    auto new_desc_sets = desc_set_list;
    new_desc_sets.push_back(debug_draw_desc_set_);

    cmd_buf->bindDescriptorSets(
        renderer::PipelineBindPoint::GRAPHICS,
        debug_draw_pipeline_layout_,
        new_desc_sets);

    cmd_buf->draw(64 * 32 * 64 * 3);
}

void DebugDrawObject::generateAllDescriptorSets(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& temp_volume_tex,
    const std::shared_ptr<renderer::ImageView>& moisture_volume_tex,
    const std::shared_ptr<renderer::ImageView>& airflow_tex) {
    generateStaticDescriptorSet(
        device,
        descriptor_pool,
        texture_sampler,
        temp_volume_tex,
        moisture_volume_tex,
        airflow_tex);
}

} // namespace game_object
} // namespace engine