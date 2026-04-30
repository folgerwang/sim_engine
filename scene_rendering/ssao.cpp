#include "ssao.h"
#include "renderer/renderer_helper.h"
#include "helper/engine_helper.h"
#include "shaders/global_definition.glsl.h"

#include <random>
#include <cmath>
#include <source_location>

namespace er = engine::renderer;

namespace engine {
namespace scene_rendering {

// ── Push-constant structs matching the shaders ──────────────────

struct SSAOPushConstants {
    glm::vec2 inv_screen_size;
    float     radius;
    float     bias;
    float     power;
    float     intensity;
    int32_t   kernel_size;
    float     pad;
};

struct SSAOBlurPushConstants {
    glm::vec2 inv_screen_size;
    glm::vec2 blur_dir;
};

struct SSAOApplyPushConstants {
    glm::vec2 inv_screen_size;
    float     ao_strength;
    float     pad;
};

// ── 4×4 random rotation noise texture ───────────────────────────

void SSAO::createNoiseTex(
    const std::shared_ptr<renderer::Device>& device) {

    // 4×4 random unit vectors in the tangent plane (z = 0).
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int N = 4;
    std::vector<uint8_t> pixels(N * N * 4);
    for (int i = 0; i < N * N; ++i) {
        float x = dist(rng);
        float y = dist(rng);
        float len = std::sqrt(x * x + y * y);
        if (len > 0.001f) { x /= len; y /= len; }
        pixels[i * 4 + 0] = uint8_t((x * 0.5f + 0.5f) * 255.0f);
        pixels[i * 4 + 1] = uint8_t((y * 0.5f + 0.5f) * 255.0f);
        pixels[i * 4 + 2] = 0;
        pixels[i * 4 + 3] = 255;
    }

    // Use the simple overload: (device, format, w, h, mips, buf_size, pixels, image&, memory&, loc)
    er::Helper::create2DTextureImage(
        device,
        er::Format::R8G8B8A8_UNORM,
        N, N,
        1,
        (uint64_t)pixels.size(),
        pixels.data(),
        noise_tex_.image,
        noise_tex_.memory,
        std::source_location::current());

    noise_tex_.size = glm::uvec3(N, N, 1);
    noise_tex_.mip_levels = 1;

    noise_tex_.view = device->createImageView(
        noise_tex_.image,
        er::ImageViewType::VIEW_2D,
        er::Format::R8G8B8A8_UNORM,
        SET_FLAG_BIT(ImageAspect, COLOR_BIT),
        std::source_location::current());
}

// ── Helpers for descriptor set layout creation ──────────────────

namespace {

std::shared_ptr<er::DescriptorSetLayout> createSSAODescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    // binding 0: depth (sampler2D)
    // binding 1: noise (sampler2D)
    // binding 2: ao output (image2D, r16f)
    std::vector<er::DescriptorSetLayoutBinding> bindings(3);
    bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        0, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        1, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[2] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        2, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::STORAGE_IMAGE);
    return device->createDescriptorSetLayout(bindings);
}

std::shared_ptr<er::DescriptorSetLayout> createBlurDescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    // binding 0: ao input (sampler2D)
    // binding 1: depth (sampler2D)
    // binding 2: ao output (image2D, r16f)
    std::vector<er::DescriptorSetLayoutBinding> bindings(3);
    bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        0, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        1, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[2] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        2, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::STORAGE_IMAGE);
    return device->createDescriptorSetLayout(bindings);
}

std::shared_ptr<er::DescriptorSetLayout> createApplyDescSetLayout(
    const std::shared_ptr<er::Device>& device) {
    // binding 0: blurred AO (sampler2D)
    // binding 1: HDR color (image2D, r/w)
    std::vector<er::DescriptorSetLayoutBinding> bindings(2);
    bindings[0] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        0, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::COMBINED_IMAGE_SAMPLER);
    bindings[1] = er::helper::getTextureSamplerDescriptionSetLayoutBinding(
        1, SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        er::DescriptorType::STORAGE_IMAGE);
    return device->createDescriptorSetLayout(bindings);
}

void writeSSAODescriptors(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSet>& desc_set,
    const std::shared_ptr<er::Sampler>& sampler,
    const std::shared_ptr<er::ImageView>& depth_view,
    const std::shared_ptr<er::ImageView>& noise_view,
    const std::shared_ptr<er::ImageView>& ao_out_view) {
    er::WriteDescriptorList writes;
    writes.reserve(3);
    er::Helper::addOneTexture(writes, desc_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER, 0,
        sampler, depth_view, er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(writes, desc_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER, 1,
        sampler, noise_view, er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(writes, desc_set,
        er::DescriptorType::STORAGE_IMAGE, 2,
        nullptr, ao_out_view, er::ImageLayout::GENERAL);
    device->updateDescriptorSets(writes);
}

void writeBlurDescriptors(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSet>& desc_set,
    const std::shared_ptr<er::Sampler>& sampler,
    const std::shared_ptr<er::ImageView>& ao_in_view,
    const std::shared_ptr<er::ImageView>& depth_view,
    const std::shared_ptr<er::ImageView>& ao_out_view) {
    er::WriteDescriptorList writes;
    writes.reserve(3);
    er::Helper::addOneTexture(writes, desc_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER, 0,
        sampler, ao_in_view, er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(writes, desc_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER, 1,
        sampler, depth_view, er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(writes, desc_set,
        er::DescriptorType::STORAGE_IMAGE, 2,
        nullptr, ao_out_view, er::ImageLayout::GENERAL);
    device->updateDescriptorSets(writes);
}

void writeApplyDescriptors(
    const std::shared_ptr<er::Device>& device,
    const std::shared_ptr<er::DescriptorSet>& desc_set,
    const std::shared_ptr<er::Sampler>& sampler,
    const std::shared_ptr<er::ImageView>& ao_view,
    const std::shared_ptr<er::ImageView>& color_view) {
    er::WriteDescriptorList writes;
    writes.reserve(2);
    er::Helper::addOneTexture(writes, desc_set,
        er::DescriptorType::COMBINED_IMAGE_SAMPLER, 0,
        sampler, ao_view, er::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    er::Helper::addOneTexture(writes, desc_set,
        er::DescriptorType::STORAGE_IMAGE, 1,
        nullptr, color_view, er::ImageLayout::GENERAL);
    device->updateDescriptorSets(writes);
}

} // anonymous namespace

// ── Constructor ─────────────────────────────────────────────────

SSAO::SSAO(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& depth_view,
    const std::shared_ptr<renderer::ImageView>& hdr_color_view,
    const glm::uvec2& display_size) {

    // One-time: noise texture + descriptor set layouts + pipelines.
    createNoiseTex(device);

    ssao_desc_set_layout_  = createSSAODescSetLayout(device);
    blur_desc_set_layout_  = createBlurDescSetLayout(device);
    apply_desc_set_layout_ = createApplyDescSetLayout(device);

    ssao_pipeline_layout_ = er::helper::createComputePipelineLayout(
        device, { ssao_desc_set_layout_, view_desc_set_layout },
        sizeof(SSAOPushConstants));

    blur_pipeline_layout_ = er::helper::createComputePipelineLayout(
        device, { blur_desc_set_layout_ },
        sizeof(SSAOBlurPushConstants));

    apply_pipeline_layout_ = er::helper::createComputePipelineLayout(
        device, { apply_desc_set_layout_ },
        sizeof(SSAOApplyPushConstants));

    ssao_pipeline_ = er::helper::createComputePipeline(
        device, ssao_pipeline_layout_,
        "ssao_compute_comp.spv",
        std::source_location::current());

    blur_pipeline_ = er::helper::createComputePipeline(
        device, blur_pipeline_layout_,
        "ssao_blur_comp.spv",
        std::source_location::current());

    apply_pipeline_ = er::helper::createComputePipeline(
        device, apply_pipeline_layout_,
        "ssao_apply_comp.spv",
        std::source_location::current());

    // Per-size resources.
    recreate(device, descriptor_pool, view_desc_set_layout,
             texture_sampler, depth_view, hdr_color_view, display_size);
}

// ── Recreate (swap chain resize) ────────────────────────────────

void SSAO::recreate(
    const std::shared_ptr<renderer::Device>& device,
    const std::shared_ptr<renderer::DescriptorPool>& descriptor_pool,
    const std::shared_ptr<renderer::DescriptorSetLayout>& view_desc_set_layout,
    const std::shared_ptr<renderer::Sampler>& texture_sampler,
    const std::shared_ptr<renderer::ImageView>& depth_view,
    const std::shared_ptr<renderer::ImageView>& hdr_color_view,
    const glm::uvec2& display_size) {

    // Destroy old textures if they exist.
    if (ao_raw_tex_.image)     ao_raw_tex_.destroy(device);
    if (ao_blurred_tex_.image) ao_blurred_tex_.destroy(device);

    // Create AO textures at screen resolution.
    er::Helper::create2DTextureImage(
        device,
        er::Format::R16_SFLOAT,
        display_size,
        1,
        ao_raw_tex_,
        SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, STORAGE_BIT),
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    er::Helper::create2DTextureImage(
        device,
        er::Format::R16_SFLOAT,
        display_size,
        1,
        ao_blurred_tex_,
        SET_2_FLAG_BITS(ImageUsage, SAMPLED_BIT, STORAGE_BIT),
        er::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        std::source_location::current());

    // ── SSAO descriptor set ──
    ssao_desc_set_ = device->createDescriptorSets(
        descriptor_pool, ssao_desc_set_layout_, 1)[0];
    writeSSAODescriptors(device, ssao_desc_set_, texture_sampler,
                         depth_view, noise_tex_.view, ao_raw_tex_.view);

    // ── Blur H: raw → blurred ──
    blur_h_desc_set_ = device->createDescriptorSets(
        descriptor_pool, blur_desc_set_layout_, 1)[0];
    writeBlurDescriptors(device, blur_h_desc_set_, texture_sampler,
                         ao_raw_tex_.view, depth_view, ao_blurred_tex_.view);

    // ── Blur V: blurred → raw (ping-pong back) ──
    blur_v_desc_set_ = device->createDescriptorSets(
        descriptor_pool, blur_desc_set_layout_, 1)[0];
    writeBlurDescriptors(device, blur_v_desc_set_, texture_sampler,
                         ao_blurred_tex_.view, depth_view, ao_raw_tex_.view);

    // ── Apply: raw AO (after V blur) × color ──
    apply_desc_set_ = device->createDescriptorSets(
        descriptor_pool, apply_desc_set_layout_, 1)[0];
    writeApplyDescriptors(device, apply_desc_set_, texture_sampler,
                          ao_raw_tex_.view, hdr_color_view);
}

// ── Render ──────────────────────────────────────────────────────

void SSAO::render(
    const std::shared_ptr<renderer::CommandBuffer>& cmd_buf,
    const std::shared_ptr<renderer::DescriptorSet>& view_desc_set,
    const std::shared_ptr<renderer::Image>& hdr_color_image,
    const glm::uvec2& display_size) {

    if (!enabled) return;

    const glm::vec2 inv_size = {
        1.0f / float(display_size.x),
        1.0f / float(display_size.y) };
    const uint32_t gx = (display_size.x + 7) / 8;
    const uint32_t gy = (display_size.y + 7) / 8;

    // ── Pass 1: SSAO generation ──
    er::helper::transitMapTextureToStoreImage(
        cmd_buf, { ao_raw_tex_.image });

    cmd_buf->bindPipeline(er::PipelineBindPoint::COMPUTE, ssao_pipeline_);
    SSAOPushConstants pc;
    pc.inv_screen_size = inv_size;
    pc.radius      = radius;
    pc.bias        = bias;
    pc.power       = power;
    pc.intensity   = intensity;
    pc.kernel_size = kernel_size;
    pc.pad         = 0.0f;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        ssao_pipeline_layout_, &pc, sizeof(pc));
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::COMPUTE,
        ssao_pipeline_layout_,
        { ssao_desc_set_, view_desc_set });
    cmd_buf->dispatch(gx, gy, 1);

    er::helper::transitMapTextureFromStoreImage(
        cmd_buf, { ao_raw_tex_.image });

    // ── Pass 2: Bilateral blur (H) ──
    er::helper::transitMapTextureToStoreImage(
        cmd_buf, { ao_blurred_tex_.image });

    cmd_buf->bindPipeline(er::PipelineBindPoint::COMPUTE, blur_pipeline_);
    SSAOBlurPushConstants blur_pc;
    blur_pc.inv_screen_size = inv_size;
    blur_pc.blur_dir = glm::vec2(1.0f, 0.0f);
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        blur_pipeline_layout_, &blur_pc, sizeof(blur_pc));
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::COMPUTE,
        blur_pipeline_layout_,
        { blur_h_desc_set_ });
    cmd_buf->dispatch(gx, gy, 1);

    er::helper::transitMapTextureFromStoreImage(
        cmd_buf, { ao_blurred_tex_.image });

    // ── Pass 3: Bilateral blur (V) → writes back to ao_raw_tex_ ──
    er::helper::transitMapTextureToStoreImage(
        cmd_buf, { ao_raw_tex_.image });

    blur_pc.blur_dir = glm::vec2(0.0f, 1.0f);
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        blur_pipeline_layout_, &blur_pc, sizeof(blur_pc));
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::COMPUTE,
        blur_pipeline_layout_,
        { blur_v_desc_set_ });
    cmd_buf->dispatch(gx, gy, 1);

    er::helper::transitMapTextureFromStoreImage(
        cmd_buf, { ao_raw_tex_.image });

    // ── Pass 4: Apply AO to HDR color ──
    er::helper::transitMapTextureToStoreImage(
        cmd_buf, { hdr_color_image });

    cmd_buf->bindPipeline(er::PipelineBindPoint::COMPUTE, apply_pipeline_);
    SSAOApplyPushConstants apply_pc;
    apply_pc.inv_screen_size = inv_size;
    apply_pc.ao_strength = strength;
    apply_pc.pad = 0.0f;
    cmd_buf->pushConstants(
        SET_FLAG_BIT(ShaderStage, COMPUTE_BIT),
        apply_pipeline_layout_, &apply_pc, sizeof(apply_pc));
    cmd_buf->bindDescriptorSets(
        er::PipelineBindPoint::COMPUTE,
        apply_pipeline_layout_,
        { apply_desc_set_ });
    cmd_buf->dispatch(gx, gy, 1);

    er::helper::transitMapTextureFromStoreImage(
        cmd_buf, { hdr_color_image },
        er::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
}

// ── Destroy ─────────────────────────────────────────────────────

void SSAO::destroy(const std::shared_ptr<renderer::Device>& device) {
    if (ao_raw_tex_.image)     ao_raw_tex_.destroy(device);
    if (ao_blurred_tex_.image) ao_blurred_tex_.destroy(device);
    if (noise_tex_.image)      noise_tex_.destroy(device);

    if (ssao_pipeline_)        device->destroyPipeline(ssao_pipeline_);
    if (ssao_pipeline_layout_) device->destroyPipelineLayout(ssao_pipeline_layout_);
    if (ssao_desc_set_layout_) device->destroyDescriptorSetLayout(ssao_desc_set_layout_);

    if (blur_pipeline_)        device->destroyPipeline(blur_pipeline_);
    if (blur_pipeline_layout_) device->destroyPipelineLayout(blur_pipeline_layout_);
    if (blur_desc_set_layout_) device->destroyDescriptorSetLayout(blur_desc_set_layout_);

    if (apply_pipeline_)        device->destroyPipeline(apply_pipeline_);
    if (apply_pipeline_layout_) device->destroyPipelineLayout(apply_pipeline_layout_);
    if (apply_desc_set_layout_) device->destroyDescriptorSetLayout(apply_desc_set_layout_);
}

} // namespace scene_rendering
} // namespace engine
